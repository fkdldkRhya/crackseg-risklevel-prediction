# train_future_mask_unet_v2.py
"""
CrackSeq GT 마스크(multi_target_XXXX.npy)만 사용해서
현재 마스크(mask_t) -> 미래 마스크(mask_{t+Δ})를 예측하는
2D U-Net 학습 (클래스 불균형 / 너무 쉬운 샘플 제거 / pos_weight 자동 튜닝 버전).

예시 실행:

python train_future_mask_unet_v2.py \
  --data_root ./crackseq_dataset_multi_mono_patches/data/multi-temporal \
  --horizon_steps 4 \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_path future_mask_unet_h4_v2.pth
"""

import os
import argparse
from glob import glob
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# 0. 재현성 & 디바이스
# ============================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# 1. Dataset: (mask_t -> mask_{t+Δ}) 쌍 생성 + pos_weight 통계
# ============================================================

class FutureMaskDataset(Dataset):
    """
    multi_target_XXXX.npy 를 읽어서
    (mask_now, mask_future) 쌍을 만드는 Dataset.

    - 하나의 파일 vol: shape = (T, H, W) 또는 (T, H, W, 1)
    - horizon_steps=4 면, (t, t+4) 쌍을 만든다.
    - 너무 쉬운 샘플(미래 마스크의 foreground 비율이 너무 낮은 경우)은 제외 가능.
    - 전체 future 마스크의 foreground 비율을 기반으로 pos_weight 추천값 계산.
    """

    def __init__(
        self,
        data_root: str,
        horizon_steps: int = 4,
        binarize_threshold: float = 0.5,
        min_fg_ratio: float = 0.0005,  # 미래 마스크의 foreground 비율 최소값 (예: 0.05%)
    ):
        super().__init__()
        self.data_root = data_root
        self.horizon = horizon_steps
        self.binarize_threshold = binarize_threshold
        self.min_fg_ratio = min_fg_ratio

        target_dir = os.path.join(data_root, "targets")
        self.target_paths: List[str] = sorted(
            glob(os.path.join(target_dir, "multi_target_*.npy"))
        )

        if not self.target_paths:
            raise FileNotFoundError(
                f"targets 디렉토리에 multi_target_*.npy 가 없습니다: {target_dir}"
            )

        # (파일 경로, t_now, t_future)의 인덱스 리스트
        self.index_tuples: List[Tuple[str, int, int]] = []

        # pos_weight 계산을 위한 통계 (전체 future 마스크 기준)
        total_pixels = 0
        total_foreground = 0

        for path in self.target_paths:
            vol = np.load(path, mmap_mode="r")  # shape 확인 (메모리 절약)
            if vol.ndim == 4 and vol.shape[-1] == 1:
                T = vol.shape[0]
                H, W = vol.shape[1], vol.shape[2]
            elif vol.ndim == 3:
                T, H, W = vol.shape
            else:
                raise RuntimeError(
                    f"multi_target shape 예외: {path}, shape={vol.shape}"
                )

            for t in range(T):
                t_future = t + self.horizon
                if t_future >= T:
                    continue

                # 미래 마스크 하나를 읽어서 fg 비율 추정 (대략, binarize 적용)
                mask_future_raw = vol[t_future]
                mask_future = self._preprocess_mask(mask_future_raw)

                fg_count = float(mask_future.sum())
                pix_count = float(mask_future.size)
                fg_ratio = fg_count / (pix_count + 1e-6)

                # 너무 fg가 거의 없는 경우에는 학습에 큰 도움 안 되므로 스킵
                if fg_ratio < self.min_fg_ratio:
                    continue

                self.index_tuples.append((path, t, t_future))

                total_pixels += pix_count
                total_foreground += fg_count

        if not self.index_tuples:
            raise RuntimeError(
                "조건(min_fg_ratio 등) 때문에 (t, t+Δ) 유효 쌍이 하나도 없습니다."
            )

        # 전체 foreground 비율
        self.fg_ratio = total_foreground / (total_pixels + 1e-6)
        self.pos_weight = None
        if self.fg_ratio > 0.0:
            # BCEWithLogitsLoss의 pos_weight 권장값 ≈ (neg/pos) = (1-p) / p
            self.pos_weight = (1.0 - self.fg_ratio) / self.fg_ratio

        print(f"[INFO] Found {len(self.target_paths)} target volumes")
        print(f"[INFO] Created {len(self.index_tuples)} (now, future) pairs "
              f"(min_fg_ratio={self.min_fg_ratio})")
        print(f"[INFO] Global future-mask fg_ratio ≈ {self.fg_ratio:.6f}")
        if self.pos_weight is not None:
            print(f"[INFO] Suggested BCE pos_weight ≈ {self.pos_weight:.2f}")

    def __len__(self):
        return len(self.index_tuples)

    def _load_vol(self, path: str) -> np.ndarray:
        vol = np.load(path)  # (T,H,W) or (T,H,W,1) or (T,H,W,3?) 방어적 처리
        if vol.ndim == 4 and vol.shape[-1] == 1:
            vol = vol[..., 0]
        elif vol.ndim == 4 and vol.shape[-1] == 3:
            # target이 RGB로 저장됐다면, 일단 첫 채널만 사용
            vol = vol[..., 0]
        if vol.ndim != 3:
            raise RuntimeError(f"multi_target shape 예외: {path}, shape={vol.shape}")
        return vol  # (T,H,W)

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        mask: (H,W), 값이 0/1, 0/255, 기타 float 등일 수 있음
        -> 0~1 float로 정규화 후 threshold로 binarize.
        """
        m = mask.astype(np.float32)
        m_min, m_max = float(m.min()), float(m.max())
        if m_max > 1.0:
            # 0~255라고 가정
            m = m / 255.0

        m = (m >= self.binarize_threshold).astype(np.float32)
        return m  # 0.0 또는 1.0

    def __getitem__(self, idx: int):
        path, t_now, t_future = self.index_tuples[idx]
        vol = self._load_vol(path)  # (T,H,W)

        mask_now = vol[t_now]       # (H,W)
        mask_future = vol[t_future] # (H,W)

        mask_now = self._preprocess_mask(mask_now)
        mask_future = self._preprocess_mask(mask_future)

        # 채널 차원 추가: (1, H, W)
        mask_now = np.expand_dims(mask_now, axis=0)
        mask_future = np.expand_dims(mask_future, axis=0)

        # torch Tensor로 변환
        x = torch.from_numpy(mask_now)      # float32, 0/1
        y = torch.from_numpy(mask_future)   # float32, 0/1

        return x, y


# ============================================================
# 2. U-Net 2D 모델 정의 (1채널 in/out)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1: upsampled, x2: skip connection
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """
    간단한 2D U-Net:
      - 입력: (B,1,H,W)
      - 출력: (B,1,H,W) (logits, sigmoid 씌우면 0~1)
    """

    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)    # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 1024//factor

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        logits = self.outc(x)
        return logits  # (B,1,H,W), 아직 sigmoid 안 씌움


# ============================================================
# 3. Loss (BCE + Dice, BCE에 pos_weight 적용)
# ============================================================

def dice_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    logits: (B,1,H,W), 아직 sigmoid 안 씌움
    targets: (B,1,H,W), 0/1
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    dice = (2.0 * intersection + eps) / union
    dice_loss = 1.0 - dice  # (B,)
    return dice_loss.mean()


# ============================================================
# 4. 학습 루프
# ============================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: str,
    pos_weight: float = 1.0,
):
    # pos_weight 텐서화
    pos_w_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # -------------------- Train --------------------
        model.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)  # (B,1,H,W)
            y = y.to(device)  # (B,1,H,W)

            optimizer.zero_grad()
            logits = model(x)  # (B,1,H,W)

            loss_bce = bce(logits, y)
            loss_dice = dice_loss_with_logits(logits, y)
            # Dice 비중을 좀 더 크게 줌
            loss = 0.2 * loss_bce + 0.8 * loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))

        # -------------------- Validation --------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss_bce = bce(logits, y)
                loss_dice = dice_loss_with_logits(logits, y)
                loss = 0.2 * loss_bce + 0.8 * loss_dice
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}"
        )

        # 모델 저장 (val loss 기준 best)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> [BEST] 모델 갱신: {save_path} (val_loss={best_val_loss:.4f})")


# ============================================================
# 5. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CrackSeq GT 기반 future mask 예측 U-Net 학습 (pos_weight/fg filtering)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./crackseq_dataset_multi_mono_patches/data/multi-temporal",
        help="multi-temporal 폴더 (images, targets 있는 경로)",
    )
    parser.add_argument(
        "--horizon_steps",
        type=int,
        default=4,
        help="t_now -> t+Δ (future) 에서 Δ (기본: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="학습 epoch 수",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="배치 크기",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="train/val 분할 비율 (기본: 0.2)",
    )
    parser.add_argument(
        "--min_fg_ratio",
        type=float,
        default=0.0005,
        help="미래 마스크에서 foreground 비율이 이 값보다 낮으면 학습 샘플에서 제외 (기본: 0.0005 = 0.05%)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='"auto", "cuda", "cpu", "mps" 등',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="future_mask_unet_v2.pth",
        help="best 모델 가중치를 저장할 경로",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--pos_weight_override",
        type=float,
        default=None,
        help="BCE pos_weight를 강제로 지정하고 싶으면 값 입력 (예: 10.0). None이면 Dataset 통계 기반 자동.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Dataset 생성 (여기서 fg_ratio, pos_weight 후보도 계산됨)
    ds = FutureMaskDataset(
        data_root=args.data_root,
        horizon_steps=args.horizon_steps,
        binarize_threshold=0.5,
        min_fg_ratio=args.min_fg_ratio,
    )

    # train/val split
    n_total = len(ds)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[INFO] Train/Val split: {n_train} / {n_val}")
    print(f"[INFO] horizon_steps = {args.horizon_steps}")
    print(f"[INFO] min_fg_ratio  = {args.min_fg_ratio}")

    # pos_weight 결정
    if args.pos_weight_override is not None:
        pos_weight = float(args.pos_weight_override)
        print(f"[INFO] Using user override pos_weight = {pos_weight:.2f}")
    else:
        if ds.pos_weight is not None:
            pos_weight = float(ds.pos_weight)
        else:
            pos_weight = 1.0
        print(f"[INFO] Using pos_weight = {pos_weight:.2f}")

    # 모델 생성
    model = UNet2D(n_channels=1, n_classes=1, bilinear=True).to(device)

    # 학습
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        pos_weight=pos_weight,
    )

    print(f"[INFO] Training finished. Best model saved at: {args.save_path}")


if __name__ == "__main__":
    main()