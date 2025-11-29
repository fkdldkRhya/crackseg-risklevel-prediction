# main_yolo_unet_future_allinone.py
# - 1) YOLOv8 세그멘테이션으로 ROI 마스크 생성
# - 2) Vizuara UNet으로 전체 크랙 확률맵(prob_map) 생성
# - 3) YOLO ROI 마스크로 UNet 확률맵을 boost (신뢰도 기반 가중치 앙상블)
#      -> Final_Prob = clip(UNet_Prob + YOLO_BOOST_ALPHA * YOLO_Mask, 0, 1)
#      -> Final_Prob를 GMM에 넣어 이진화 → base now 마스크
# - 4) 원형/잡음 제거 → 최종 now 마스크
# - 5) 최종 now 마스크를 future-mask UNet에 넣어 미래 크랙 마스크 예측
# - 6) 미래 마스크에 대해: 글로벌 + 컴포넌트별 위험도 분석
# - 7) 미래 마스크들을 OR 합한 결과가 최종 미래 크랙 예측 마스크

import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Dict
from collections import defaultdict

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from sklearn.mixture import GaussianMixture
from ultralytics import YOLO


# ============================================================
# 0. GLOBAL CONFIG & CONSTANTS
# ============================================================

OUT_DIR: str = "out"
DEFAULT_MM_PER_PIXEL: float = 1.0

# ---- Vizuara UNet (현재 마스크용) ----
VIZUARA_REPO_ID = "Vizuara/unet-crack-segmentation"
VIZUARA_FILENAME = "unet_weights_v2.pth"
VIZUARA_IMG_H = 256
VIZUARA_IMG_W = 256

# ---- YOLOv8 (ROI 세그멘테이션용) ----
YOLO_REPO_ID = "OpenSistemas/YOLOv8-crack-seg"   # 필요 시 수정
YOLO_WEIGHT_FILENAME = "yolov8x/weights/best.pt"  # HF 안의 실제 파일명에 맞게 수정
YOLO_CONF = 0.25
YOLO_IMGSZ = 1024

# ---- YOLO → UNet 확률맵 boost 계수 (앙상블 강도) ----
YOLO_BOOST_ALPHA: float = 0.30   # 예: Final_Prob = UNET_PROB + 0.3 * YOLO_MASK

# ---- 미래 마스크 UNet (future_mask_unet_h4_v2.pth) ----
FUTURE_TILE_DEFAULT: int   = 128
FUTURE_STRIDE_DEFAULT: int = 64
FUTURE_PRED_THRESH_DEFAULT: float = 0.5

# ---- Severity 기준값 (논문 기반) ----
WIDTH_T0_MM: float = 0.30
WIDTH_T1_MM: float = 0.50

DENSITY_T0_PER_M: float = 0.5
DENSITY_T1_PER_M: float = 3.0

AREA_T0_MM2: float = 500.0
AREA_T1_MM2: float = 5000.0

FRACTAL_LOW: float = 1.20
FRACTAL_HIGH: float = 1.60

LENGTH_T0_M: float = 0.3
LENGTH_T1_M: float = 3.0

# ---- Risk level 구간 (R: 0~1) ----
RISK_THRESH_A: float = 0.2
RISK_THRESH_B: float = 0.4
RISK_THRESH_C: float = 0.7

# ---- Severity/Probability/Risk 가중치 ----
SEV_WEIGHT_WIDTH: float   = 1.0
SEV_WEIGHT_DENSITY: float = 1.0
SEV_WEIGHT_AREA: float    = 1.0
SEV_WEIGHT_FRACTAL: float = 1.0
SEV_WEIGHT_LENGTH: float  = 1.0

PROB_WEIGHT_TIP: float        = 1.0
PROB_WEIGHT_WIDTH_GRAD: float = 1.0
PROB_WEIGHT_ORIENT: float     = 1.0

RISK_WEIGHT_S: float = 0.65
RISK_WEIGHT_P: float = 0.35


# ---- 크랙 모양 필터링 기준 (원형/잡음 제거용) ----
MIN_COMP_AREA = 50          # 너무 작은 건 제거
MIN_LONG_AXIS = 20          # 가로/세로 중 긴 변 최소 길이
MIN_ASPECT_RATIO = 2.0      # aspect < 2 & 작으면 제거
MAX_CIRCULARITY = 0.75      # 4πA/P^2가 이 이상이면 거의 원형 → 제거 후보


# ============================================================
# 1. 기본 유틸
# ============================================================

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def out_path(image_path: str, postfix: str, ext: str = "png") -> str:
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    fname = f"{base_name}{postfix}.{ext}"
    return os.path.join(OUT_DIR, fname)


def preprocess_crack_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() > 1:
        m = (m > 0).astype(np.uint8)
    return m


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# 2. Vizuara UNet + GMM (현재 마스크)
# ============================================================

class VizuaraUNet(nn.Module):
    def __init__(self):
        super(VizuaraUNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        c4 = self.enc4(p3)
        p4 = self.pool(c4)

        bottleneck = self.bottleneck(p4)

        u4 = self.upconv4(bottleneck)
        u4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.upconv3(d4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)

        return torch.sigmoid(self.conv_last(d1))


def crack_edge_boost(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    sobel3 = cv2.Sobel(gray_clahe, cv2.CV_32F, 1, 0, ksize=3)
    sobel5 = cv2.Sobel(gray_clahe, cv2.CV_32F, 1, 0, ksize=5)
    sobel = cv2.addWeighted(sobel3, 0.5, sobel5, 0.5, 0)
    sobel = cv2.convertScaleAbs(sobel)

    lap = cv2.Laplacian(gray_clahe, cv2.CV_32F, ksize=3)
    lap = cv2.convertScaleAbs(lap)

    boost = cv2.addWeighted(gray_clahe, 0.6, sobel, 0.3, 0)
    boost = cv2.addWeighted(boost, 1.0, lap, 0.2, 0)
    boost_rgb = cv2.cvtColor(boost, cv2.COLOR_GRAY2BGR)
    return boost_rgb


def load_vizuara_unet(device: torch.device) -> VizuaraUNet:
    print(f"[INFO] Loading Vizuara UNet weights from HF: {VIZUARA_REPO_ID}/{VIZUARA_FILENAME}")
    weights_path = hf_hub_download(
        repo_id=VIZUARA_REPO_ID,
        filename=VIZUARA_FILENAME,
    )
    print(f"[INFO] UNet weights downloaded at: {weights_path}")
    model = VizuaraUNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_unet_prob_map(
    model: VizuaraUNet,
    image_bgr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    boosted = crack_edge_boost(image_bgr)
    image_rgb = cv2.cvtColor(boosted, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    transform = transforms.Compose(
        [
            transforms.Resize((VIZUARA_IMG_H, VIZUARA_IMG_W)),
            transforms.ToTensor(),
        ]
    )

    with torch.no_grad():
        inp = transform(pil_img).unsqueeze(0).to(device)
        pred = model(inp)
        pred = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    prob_map = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob_map


def keep_elongated_components(
    mask_u8: np.ndarray,
    min_area: float = 80.0,
    min_elongation: float = 2.5,
    keep_top_k: int = 5,
) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m

    h, w = m.shape
    img_diag = np.sqrt(h * h + w * w)

    comps = []
    for lbl in range(1, num_labels):
        x, y, ww, hh, area = stats[lbl]
        if area < min_area:
            continue

        aspect = max(ww, hh) / (min(ww, hh) + 1e-6)
        norm_len = np.sqrt(ww * ww + hh * hh) / (img_diag + 1e-6)
        score = aspect * (0.5 + 0.5 * norm_len)

        if aspect < min_elongation:
            continue

        comps.append((lbl, score, area))

    if not comps:
        return m

    comps.sort(key=lambda x: x[1], reverse=True)
    comps = comps[:keep_top_k]

    keep_labels = {lbl for (lbl, _, _) in comps}
    out = np.zeros_like(m, dtype=np.uint8)
    for lbl in keep_labels:
        out[labels == lbl] = 1

    return out


def fill_small_holes_by_contours(
    mask_u8: np.ndarray,
    max_hole_ratio: float = 0.02,
    max_hole_area: int = 2000,
) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(
        m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return (m > 0).astype(np.uint8)

    hierarchy = hierarchy[0]
    img_area = m.shape[0] * m.shape[1]

    for i, cnt in enumerate(contours):
        parent_idx = hierarchy[i][3]
        if parent_idx == -1:
            continue

        area = cv2.contourArea(cnt)
        if area > max_hole_area:
            continue
        if area > img_area * max_hole_ratio:
            continue

        cv2.drawContours(m, [cnt], -1, 255, thickness=-1)

    return (m > 0).astype(np.uint8)


def prob_to_binary_gmm(
    prob_map: np.ndarray,
    image_bgr: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """
    여기서 들어오는 prob_map은 이미 YOLO 기반 boost가 적용된 Final_Prob.
    """
    p_min = float(prob_map.min())
    p_max = float(prob_map.max())
    print(f"[INFO] prob_map(min,max) = ({p_min:.4f}, {p_max:.4f})")

    if p_max - p_min < 1e-6:
        print("[WARN] prob_map dynamic range 거의 없음 → 빈 마스크 반환")
        return np.zeros_like(prob_map, dtype=np.uint8)

    norm = (prob_map - p_min) / (p_max - p_min + 1e-6)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

    h, w = norm.shape

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(gx, gy)

    e_min, e_max = float(edge_mag.min()), float(edge_mag.max())
    if e_max - e_min < 1e-6:
        edge_norm = np.zeros_like(edge_mag, dtype=np.float32)
    else:
        edge_norm = (edge_mag - e_min) / (e_max - e_min + 1e-6)

    feat1 = norm.reshape(-1, 1)
    feat2 = edge_norm.reshape(-1, 1)
    features = np.concatenate([feat1, feat2], axis=1)

    total_px = features.shape[0]
    sample_pixels = 50000
    if total_px > sample_pixels:
        idx = np.random.choice(total_px, sample_pixels, replace=False)
        train_feat = features[idx]
    else:
        train_feat = features

    print(f"[INFO] Fitting GMM (n_components={n_components}) on {train_feat.shape[0]} samples...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42
    )
    gmm.fit(train_feat)

    labels = gmm.predict(features)
    labels_img = labels.reshape(h, w)

    cluster_means = []
    flat_norm = norm.reshape(-1)
    for k in range(n_components):
        mask_k = (labels == k)
        if mask_k.any():
            mean_k = float(flat_norm[mask_k].mean())
        else:
            mean_k = 0.0
        cluster_means.append(mean_k)
        print(f"[INFO] cluster {k}: mean_norm={mean_k:.4f}, count={int(mask_k.sum())}")

    cluster_means = np.array(cluster_means)
    main_k = int(cluster_means.argmax())
    print(f"[INFO] Main crack cluster = {main_k} (mean_norm={cluster_means[main_k]:.4f})")

    sorted_idx = np.argsort(-cluster_means)
    crack_clusters = [int(sorted_idx[0])]
    if n_components >= 2:
        k2 = int(sorted_idx[1])
        if cluster_means[main_k] - cluster_means[k2] < 0.15:
            crack_clusters.append(k2)
            print(f"[INFO] Also including cluster {k2} (mean_norm={cluster_means[k2]:.4f})")

    print(f"[INFO] crack_clusters = {crack_clusters}")

    crack_mask = np.isin(labels_img, crack_clusters).astype(np.uint8)
    crack_mask_u8 = crack_mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crack_mask_u8 = cv2.morphologyEx(crack_mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    crack_mask_u8 = cv2.morphologyEx(crack_mask_u8, cv2.MORPH_ERODE, k, iterations=1)

    crack_mask_elong = keep_elongated_components(
        crack_mask_u8,
        min_area=80.0,
        min_elongation=2.5,
        keep_top_k=5,
    )

    crack_mask_filled = fill_small_holes_by_contours(
        crack_mask_elong,
        max_hole_ratio=0.02,
        max_hole_area=2000,
    )

    return crack_mask_filled


def generate_crack_mask_unet_gmm(
    image_path: str,
    model: VizuaraUNet,
    device: torch.device,
    yolo_roi_mask: Optional[np.ndarray] = None,
    yolo_boost: float = YOLO_BOOST_ALPHA,
) -> np.ndarray:
    """
    UNet 확률맵 + (옵션) YOLO ROI 마스크로 boost 한 후
    GMM 이진화까지 수행해서 base now 마스크 생성.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {image_path}")

    h, w = image_bgr.shape[:2]
    print(f"[INFO] Image shape: {w}x{h} (WxH)")

    print("[INFO] Running Vizuara UNet inference (prob_map)...")
    prob_map_unet = run_unet_prob_map(model, image_bgr, device)

    # --- YOLO 기반 확률 boost (옵션 2: 신뢰도 기반 가중치 앙상블) ---
    if yolo_roi_mask is not None:
        print(f"[INFO] Applying YOLO-based prob boosting (alpha={yolo_boost:.3f})")
        # YOLO 마스크가 다른 크기면 resize
        if yolo_roi_mask.shape != prob_map_unet.shape:
            yolo_resized = cv2.resize(
                yolo_roi_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            yolo_resized = yolo_roi_mask.astype(np.uint8)

        yolo_float = (yolo_resized > 0).astype(np.float32)  # 0/1 mask
        prob_map_final = prob_map_unet + yolo_boost * yolo_float
        prob_map_final = np.clip(prob_map_final, 0.0, 1.0)
    else:
        print("[INFO] YOLO ROI mask is None → UNet prob_map만 사용")
        prob_map_final = prob_map_unet

    # 디버그용 확률맵 저장
    ensure_out_dir()
    cv2.imwrite(
        out_path(image_path, "_unet_prob_raw"),
        (np.clip(prob_map_unet, 0.0, 1.0) * 255).astype(np.uint8),
    )
    cv2.imwrite(
        out_path(image_path, "_unet_prob_boosted"),
        (np.clip(prob_map_final, 0.0, 1.0) * 255).astype(np.uint8),
    )

    print("[INFO] Running GMM clustering on [Final_Prob, edge] features...")
    bin_mask = prob_to_binary_gmm(
        prob_map_final,
        image_bgr,
        n_components=3,
    )

    bin_mask = (bin_mask > 0).astype(np.uint8)

    mask_path = out_path(image_path, "_now_base_mask")
    cv2.imwrite(mask_path, bin_mask * 255)
    print(f"[INFO] Saved base now crack mask (UNet+YOLO ensemble+GMM) to: {mask_path}")

    overlay = image_bgr.copy()
    overlay[bin_mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)
    overlay_path = out_path(image_path, "_now_base_overlay")
    cv2.imwrite(overlay_path, blended)
    print(f"[INFO] Saved base now overlay to: {overlay_path}")

    return bin_mask


# ============================================================
# 3. YOLOv8 세그멘테이션 기반 ROI 마스크
# ============================================================

def load_yolo_from_hf(device_str: str) -> YOLO:
    print(f"[INFO] Downloading YOLOv8 weights from HF: {YOLO_REPO_ID}/{YOLO_WEIGHT_FILENAME}")
    weight_path = hf_hub_download(
        repo_id=YOLO_REPO_ID,
        filename=YOLO_WEIGHT_FILENAME,
    )
    print(f"[INFO] YOLO weights at: {weight_path}")
    model = YOLO(weight_path)
    model.to(device_str)
    return model


def generate_yolo_roi_mask_seg(
    image_path: str,
    yolo_model: YOLO,
) -> np.ndarray:
    """
    YOLOv8-seg 결과의 mask들을 모두 OR한 ROI 마스크 생성 (0/1)
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
    H, W = img_bgr.shape[:2]

    print(f"[INFO] Running YOLOv8 segmentation (conf={YOLO_CONF}, imgsz={YOLO_IMGSZ})")
    results = yolo_model(
        image_path,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )
    if len(results) == 0:
        print("[INFO] YOLO 결과 없음 → ROI 마스크는 0")
        return np.zeros((H, W), dtype=np.uint8)

    r = results[0]
    if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
        print("[INFO] YOLO masks 없음 → ROI 마스크는 0")
        return np.zeros((H, W), dtype=np.uint8)

    masks = r.masks.data.cpu().numpy()  # (N, Hm, Wm), 0~1
    union = (masks.sum(axis=0) > 0.5).astype(np.uint8)
    # YOLO가 리사이즈 했을 수 있으니 원본 사이즈로 보간
    roi_mask = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST)

    ensure_out_dir()
    cv2.imwrite(out_path(image_path, "_yolo_roi_mask"), roi_mask * 255)

    overlay = img_bgr.copy()
    overlay[roi_mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
    cv2.imwrite(out_path(image_path, "_yolo_roi_overlay"), blended)
    print("[INFO] Saved YOLO ROI mask & overlay.")

    print(f"[INFO] YOLO ROI fg_ratio={float(roi_mask.mean()):.4f}")

    return roi_mask


# ============================================================
# 4. 크랙 모양 필터링 (원형/잡음 제거)
# ============================================================

def filter_crack_like_components(
    combined_mask: np.ndarray,
    image_path: str,
) -> np.ndarray:
    """
    GMM으로 얻은 base now 마스크에서:
      - 너무 작은 것
      - 거의 원형에 가까운 blob
      - 짧고 aspect 낮은 것
    제거해서 최종 now 마스크 생성
    """
    m = preprocess_crack_mask(combined_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = np.zeros_like(m, dtype=np.uint8)
    keep_count = 0

    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area < MIN_COMP_AREA:
            continue

        long_axis = max(w, h)
        short_axis = min(w, h) + 1e-6
        aspect = long_axis / short_axis

        comp = np.zeros_like(m, dtype=np.uint8)
        comp[labels == lbl] = 1

        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0.0
        for c in contours:
            perimeter += cv2.arcLength(c, True)

        if perimeter <= 0.0:
            continue

        circularity = 4.0 * np.pi * float(area) / (perimeter * perimeter + 1e-6)

        # 원형 blob: circularity가 크고, aspect도 낮으면 제거
        if circularity > MAX_CIRCULARITY and aspect < MIN_ASPECT_RATIO:
            continue

        # 너무 짧고 aspect가 낮은 직선/점은 제거
        if long_axis < MIN_LONG_AXIS and aspect < MIN_ASPECT_RATIO:
            continue

        out[labels == lbl] = 1
        keep_count += 1

    print(f"[INFO] filter_crack_like_components: {num_labels-1} comps -> {keep_count} kept")

    ensure_out_dir()
    cv2.imwrite(out_path(image_path, "_now_mask_raw"), m * 255)
    cv2.imwrite(out_path(image_path, "_now_mask_filtered"), out * 255)

    return out


# ============================================================
# 5. 크랙 기하/위험도 계산 모듈
# ============================================================

def skeletonize_cv(crack_mask: np.ndarray) -> np.ndarray:
    m = preprocess_crack_mask(crack_mask)

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skel = cv2.ximgproc.thinning(m)
        return (skel > 0).astype(np.uint8)

    skel = np.zeros_like(m)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False
    while not done:
        eroded = cv2.erode(m, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(m, temp)
        skel = cv2.bitwise_or(skel, temp)
        m = eroded.copy()
        if cv2.countNonZero(m) == 0:
            done = True

    return (skel > 0).astype(np.uint8)


def compute_w_max_mm(crack_mask: np.ndarray, mm_per_pixel: float) -> float:
    m = preprocess_crack_mask(crack_mask)
    if m.sum() == 0:
        return 0.0

    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    max_radius_px = float(dist.max())
    w_max_px = 2.0 * max_radius_px
    return w_max_px * mm_per_pixel


def compute_crack_length_and_density_per_m(
    crack_mask: np.ndarray,
    mm_per_pixel: float,
) -> Tuple[float, float]:
    m = preprocess_crack_mask(crack_mask)
    h, w = m.shape[:2]

    if m.sum() == 0:
        return 0.0, 0.0

    skel = skeletonize_cv(m)
    length_px = int(skel.sum())

    crack_length_m = (length_px * mm_per_pixel) / 1000.0
    width_m = (w * mm_per_pixel) / 1000.0
    if width_m == 0:
        return 0.0, crack_length_m

    rho_L = crack_length_m / width_m
    return rho_L, crack_length_m


def fractal_dimension_boxcount(
    crack_mask: np.ndarray,
    min_box_size: int = 2,
    max_box_size: Optional[int] = None,
    num_scales: int = 6,
) -> float:
    m = preprocess_crack_mask(crack_mask)
    if m.sum() == 0:
        return 1.0

    h, w = m.shape[:2]
    if max_box_size is None:
        max_box_size = max(min(h, w) // 2, min_box_size * 2)

    sizes = np.logspace(
        np.log2(min_box_size), np.log2(max_box_size),
        num=num_scales, base=2
    ).astype(int)

    sizes = np.unique(sizes)
    sizes = sizes[sizes > 0]

    Ns, inv_sizes = [], []
    for s in sizes:
        n_h = (h + s - 1) // s
        n_w = (w + s - 1) // s
        pad_h = n_h * s - h
        pad_w = n_w * s - w
        padded = np.pad(m, ((0, pad_h), (0, pad_w)), constant_values=0)
        blocks = padded.reshape(n_h, s, n_w, s).sum(axis=(1, 3))
        N_s = np.count_nonzero(blocks)
        if N_s > 0:
            Ns.append(N_s)
            inv_sizes.append(1.0 / s)

    if len(Ns) < 2:
        return 1.0

    log_inv_s = np.log(inv_sizes)
    log_Ns = np.log(Ns)
    coeffs = np.polyfit(log_inv_s, log_Ns, 1)
    return float(coeffs[0])


def compute_crack_area_mm2(crack_mask: np.ndarray, mm_per_pixel: float) -> float:
    m = preprocess_crack_mask(crack_mask)
    area_px = int(m.sum())
    return area_px * (mm_per_pixel ** 2)


def piecewise_score(x: float, t0: float, t1: float) -> float:
    if t0 <= 0 or t1 <= t0:
        raise ValueError("t0>0, t1>t0 이어야 합니다.")
    if x <= 0:
        return 0.0
    if x <= t0:
        return 0.5 * (x / t0)
    if x <= t1:
        return 0.5 + 0.5 * ((x - t0) / (t1 - t0))
    return 1.0


def score_width_continuous(w_max_mm: float) -> float:
    return piecewise_score(w_max_mm, WIDTH_T0_MM, WIDTH_T1_MM)


def score_density_continuous(rho_L_per_m: float) -> float:
    return piecewise_score(rho_L_per_m, DENSITY_T0_PER_M, DENSITY_T1_PER_M)


def score_area_continuous(area_mm2: float) -> float:
    return piecewise_score(area_mm2, AREA_T0_MM2, AREA_T1_MM2)


def score_fractal_continuous(D_f: float) -> float:
    if D_f <= FRACTAL_LOW:
        return 0.0
    if D_f >= FRACTAL_HIGH:
        return 1.0
    return (D_f - FRACTAL_LOW) / (FRACTAL_HIGH - FRACTAL_LOW)


def score_length_continuous(length_m: float) -> float:
    return piecewise_score(length_m, LENGTH_T0_M, LENGTH_T1_M)


def _find_skeleton_endpoints(skel: np.ndarray):
    sk = (skel > 0).astype(np.uint8)
    h, w = sk.shape
    endpoints = []
    for y in range(h):
        for x in range(w):
            if sk[y, x] == 0:
                continue
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            neighbor_cnt = int(sk[y0:y1, x0:x1].sum())
            if neighbor_cnt - 1 == 1:
                endpoints.append((y, x))
    return endpoints


def compute_tip_sharpness_score(
    crack_mask: np.ndarray,
) -> float:
    m = preprocess_crack_mask(crack_mask)
    if m.sum() == 0:
        return 0.0

    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    skel = skeletonize_cv(m)
    endpoints = _find_skeleton_endpoints(skel)
    if not endpoints:
        return 0.0

    h, w = m.shape[:2]
    R_pix = 10
    sharp_vals = []

    for (y, x) in endpoints:
        tip_r = float(dist[y, x])

        y0, y1 = max(0, y - R_pix), min(h, y + R_pix + 1)
        x0, x1 = max(0, x - R_pix), min(w, x + R_pix + 1)
        local_mask = m[y0:y1, x0:x1]
        local_dist = dist[y0:y1, x0:x1]

        local_r = local_dist[local_mask > 0]
        if local_r.size == 0:
            continue

        interior_r = float(local_r.max())
        if interior_r <= 0:
            continue

        val = 1.0 - (tip_r / interior_r)
        val = float(np.clip(val, 0.0, 1.0))
        sharp_vals.append(val)

    if not sharp_vals:
        return 0.0

    return float(np.mean(sharp_vals))


def compute_width_gradient_score(
    crack_mask: np.ndarray,
    mm_per_pixel: float,
) -> float:
    m = preprocess_crack_mask(crack_mask)
    if m.sum() == 0:
        return 0.0

    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    skel = skeletonize_cv(m)

    skel_idx = skel > 0
    if skel_idx.sum() < 2:
        return 0.0

    widths_mm = 2.0 * dist[skel_idx] * mm_per_pixel
    widths_mm = widths_mm[widths_mm > 0]
    if widths_mm.size < 2:
        return 0.0

    p10 = float(np.percentile(widths_mm, 10))
    p90 = float(np.percentile(widths_mm, 90))
    if p90 <= 0:
        return 0.0

    spread = (p90 - p10) / (p90 + 1e-6)
    spread = float(np.clip(spread, 0.0, 1.0))
    return spread


def compute_orientation_stats(
    crack_mask: np.ndarray,
) -> Tuple[float, Tuple[float, float], Tuple[float, float], np.ndarray]:
    m = preprocess_crack_mask(crack_mask)
    if m.sum() == 0:
        h, w = m.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        return 0.0, (cx, cy), (1.0, 0.0), np.zeros_like(m)

    skel = skeletonize_cv(m)
    ys, xs = np.where(skel > 0)
    if len(xs) < 2:
        h, w = m.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        return 0.0, (cx, cy), (1.0, 0.0), skel

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean_pt = pts.mean(axis=0)
    cx, cy = float(mean_pt[0]), float(mean_pt[1])

    pts_centered = pts - mean_pt
    cov = np.cov(pts_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    v = eigvecs[:, idx[0]]
    vx, vy = float(v[0]), float(v[1])

    angle_rad = float(np.arctan2(vy, vx))
    angle_deg = angle_rad * 180.0 / np.pi

    return angle_deg, (cx, cy), (vx, vy), skel


def save_skeleton_orientation_visuals(
    image_path: str,
    crack_mask: np.ndarray,
    skel: np.ndarray,
    angle_deg: float,
    center_xy: Tuple[float, float],
    axis_vec: Tuple[float, float],
    suffix: str,
) -> None:
    ensure_out_dir()
    skel_vis = (skel > 0).astype(np.uint8) * 255
    cv2.imwrite(out_path(image_path, f"_{suffix}_skeleton"), skel_vis)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return

    h, w = img.shape[:2]
    overlay = img.copy()

    yy, xx = np.where(skel > 0)
    overlay[yy, xx] = (0, 0, 255)

    cx, cy = center_xy
    vx, vy = axis_vec
    if vx == 0 and vy == 0:
        vx, vy = 1.0, 0.0

    length = min(h, w) / 3.0
    x1 = int(cx - vx * length)
    y1 = int(cy - vy * length)
    x2 = int(cx + vx * length)
    y2 = int(cy + vy * length)

    cv2.arrowedLine(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2, tipLength=0.1)

    text = f"{angle_deg:.1f} deg"
    cv2.putText(
        overlay, text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(out_path(image_path, f"_{suffix}_skeleton_orient"), overlay)


def compute_orientation_risk_base(angle_deg: float) -> float:
    angle_rad = angle_deg * np.pi / 180.0

    theta = abs(angle_rad)
    if theta > np.pi:
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    theta = abs(theta)
    if theta > np.pi / 2:
        theta = np.pi - theta

    s = np.sin(2.0 * theta)
    s = float(np.clip(s, 0.0, 1.0))
    return s


@dataclass
class SeverityFeatures:
    w_max_mm: float
    rho_L_per_m: float
    area_mm2: float
    D_f: float
    length_m: float


@dataclass
class SeverityScores:
    S_w: float
    S_density: float
    S_area: float
    S_fractal: float
    S_length: float
    S_total: float


@dataclass
class SeverityWeights:
    alpha: float = SEV_WEIGHT_WIDTH
    beta: float = SEV_WEIGHT_DENSITY
    gamma: float = SEV_WEIGHT_AREA
    delta: float = SEV_WEIGHT_FRACTAL
    epsilon: float = SEV_WEIGHT_LENGTH


@dataclass
class ProbabilityInputs:
    P_TipSharpness: float
    P_WidthGradient: float
    P_Orientation: float


@dataclass
class ProbabilityWeights:
    alpha: float = PROB_WEIGHT_TIP
    beta: float = PROB_WEIGHT_WIDTH_GRAD
    gamma: float = PROB_WEIGHT_ORIENT


@dataclass
class RiskWeights:
    wS: float = RISK_WEIGHT_S
    wP: float = RISK_WEIGHT_P


@dataclass
class RiskResult:
    image_path: str
    S: float
    P: float
    R: float
    level: Literal["A", "B", "C", "D"]
    severity_scores: Optional[SeverityScores]


def compute_severity(
    feats: SeverityFeatures,
    weights: SeverityWeights = SeverityWeights(),
) -> SeverityScores:
    S_w = score_width_continuous(feats.w_max_mm)
    S_density = score_density_continuous(feats.rho_L_per_m)
    S_area = score_area_continuous(feats.area_mm2)
    S_fractal = score_fractal_continuous(feats.D_f)
    S_length = score_length_continuous(feats.length_m)

    a, b, c, d, e = (
        weights.alpha,
        weights.beta,
        weights.gamma,
        weights.delta,
        weights.epsilon,
    )
    denom = a + b + c + d + e
    if denom <= 0:
        raise ValueError("SeverityWeights 합이 0 이하여서는 안 됩니다.")

    S_total = (a * S_w + b * S_density + c * S_area + d * S_fractal + e * S_length) / denom

    return SeverityScores(
        S_w=S_w,
        S_density=S_density,
        S_area=S_area,
        S_fractal=S_fractal,
        S_length=S_length,
        S_total=S_total,
    )


def compute_probability(
    prob: ProbabilityInputs,
    weights: ProbabilityWeights = ProbabilityWeights(),
) -> float:
    a, b, c = weights.alpha, weights.beta, weights.gamma
    denom = a + b + c
    if denom <= 0:
        raise ValueError("ProbabilityWeights 합이 0 이하여서는 안 됩니다.")
    return (
        a * prob.P_TipSharpness
        + b * prob.P_WidthGradient
        + c * prob.P_Orientation
    ) / denom


def classify_risk_level(R: float) -> Literal["A", "B", "C", "D"]:
    if R < RISK_THRESH_A:
        return "A"
    elif R < RISK_THRESH_B:
        return "B"
    elif R < RISK_THRESH_C:
        return "C"
    else:
        return "D"


def compute_risk(
    img_path: str,
    S: float,
    P: float,
    risk_weights: RiskWeights = RiskWeights(),
    mode: Literal["product", "weighted_sum"] = "weighted_sum",
) -> RiskResult:
    S_clamp = float(np.clip(S, 0.0, 1.0))
    P_clamp = float(np.clip(P, 0.0, 1.0))

    if mode == "product":
        R = S_clamp * P_clamp
    else:
        wS, wP = risk_weights.wS, risk_weights.wP
        s = wS + wP
        if s <= 0:
            raise ValueError("RiskWeights wS+wP > 0 이어야 합니다.")
        wS /= s
        wP /= s
        R = wS * S_clamp + wP * P_clamp

    R = float(np.clip(R, 0.0, 1.0))
    level = classify_risk_level(R)

    return RiskResult(
        image_path=img_path,
        S=S_clamp,
        P=P_clamp,
        R=R,
        level=level,
        severity_scores=None,
    )


def analyze_mask(
    image_path: str,
    crack_mask: np.ndarray,
    mm_per_pixel: float,
    tag: str,
) -> RiskResult:
    ensure_out_dir()
    m = preprocess_crack_mask(crack_mask)

    mask_vis = (m * 255).astype(np.uint8)
    cv2.imwrite(out_path(image_path, f"_{tag}_mask"), mask_vis)
    mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
    cv2.imwrite(out_path(image_path, f"_{tag}_mask_color"), mask_color)

    w_max_mm = compute_w_max_mm(m, mm_per_pixel)
    rho_L, length_m = compute_crack_length_and_density_per_m(m, mm_per_pixel)
    D_f = fractal_dimension_boxcount(m)
    area_mm2 = compute_crack_area_mm2(m, mm_per_pixel)

    feats = SeverityFeatures(
        w_max_mm=w_max_mm,
        rho_L_per_m=rho_L,
        area_mm2=area_mm2,
        D_f=D_f,
        length_m=length_m,
    )
    sev_scores = compute_severity(feats)

    angle_deg, center_xy, axis_vec, skel_for_vis = compute_orientation_stats(m)
    save_skeleton_orientation_visuals(
        image_path,
        m,
        skel_for_vis,
        angle_deg,
        center_xy,
        axis_vec,
        suffix=tag,
    )

    P_tip = compute_tip_sharpness_score(m)
    P_grad = compute_width_gradient_score(m, mm_per_pixel)
    orient_base = compute_orientation_risk_base(angle_deg)
    P_orient = orient_base * sev_scores.S_length

    prob_inputs = ProbabilityInputs(
        P_TipSharpness=P_tip,
        P_WidthGradient=P_grad,
        P_Orientation=P_orient,
    )
    P_val = compute_probability(prob_inputs)

    rr = compute_risk(
        img_path=image_path,
        S=sev_scores.S_total,
        P=P_val,
        mode="weighted_sum",
    )
    rr.severity_scores = sev_scores

    print(f"\n=== [{tag}] {image_path} ===")
    print(f"w_max_mm        : {w_max_mm:.4f}")
    print(f"rho_L_per_m     : {rho_L:.4f}")
    print(f"length_m        : {length_m:.4f}")
    print(f"D_f             : {D_f:.4f}")
    print(f"area_mm2        : {area_mm2:.4f}")
    print("--- Severity scores (0~1) ---")
    print(f"S_w             : {sev_scores.S_w:.3f}")
    print(f"S_density       : {sev_scores.S_density:.3f}")
    print(f"S_area          : {sev_scores.S_area:.3f}")
    print(f"S_fractal       : {sev_scores.S_fractal:.3f}")
    print(f"S_length        : {sev_scores.S_length:.3f}")
    print(f"S_total         : {sev_scores.S_total:.3f}")
    print("--- Probability components (0~1) ---")
    print(f"P_TipSharpness  : {prob_inputs.P_TipSharpness:.3f}")
    print(f"P_WidthGradient : {prob_inputs.P_WidthGradient:.3f}")
    print(f"P_Orientation   : {prob_inputs.P_Orientation:.3f}  "
          f"(angle={angle_deg:.1f}°, base={orient_base:.3f})")
    print(f"P               : {P_val:.3f}")
    print("--- Risk (0~1) ---")
    print(f"R               : {rr.R:.3f}, Level: {rr.level}")

    return rr


def analyze_components(
    image_path: str,
    mask: np.ndarray,
    mm_per_pixel: float,
    tag_prefix: str,
) -> List[RiskResult]:
    """
    최종 마스크를 컴포넌트별로 나눠서 각 크랙(가지)에 대해 위험도를 개별 산출
    + 각 크랙마다 ROI bounding box가 그려진 이미지를 저장
      - 개별 이미지: <base>_{tag}_bbox.png
      - 전체 bbox 모으는 이미지: <base>_{tag_prefix}_components_bbox_all.png
    """
    m = preprocess_crack_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    results: List[RiskResult] = []

    print(f"\n[INFO] Component-wise analysis ({tag_prefix}): {num_labels-1} components")

    base_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    combined_overlay = None
    if base_img is not None:
        combined_overlay = base_img.copy()

    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area < MIN_COMP_AREA:
            continue

        comp = (labels == lbl).astype(np.uint8)
        tag = f"{tag_prefix}_c{lbl}"
        rr = analyze_mask(image_path, comp, mm_per_pixel, tag=tag)
        results.append(rr)

        # ----- 각 크랙마다 ROI bbox 이미지 저장 -----
        if base_img is not None:
            comp_overlay = base_img.copy()

            # 크랙 픽셀 빨간색으로 표시
            yy, xx = np.where(comp > 0)
            comp_overlay[yy, xx] = (0, 0, 255)

            # bounding box (초록색)
            cv2.rectangle(
                comp_overlay,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            # 파일 저장: <이미지이름>_{tag}_bbox.png
            cv2.imwrite(out_path(image_path, f"_{tag}_bbox"), comp_overlay)

            # 전체 bbox 한 장에 모으는 overlay에도 bbox 추가
            if combined_overlay is not None:
                cv2.rectangle(
                    combined_overlay,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    2,
                )

    # 전체 컴포넌트 bbox만 모은 이미지도 저장
    if combined_overlay is not None:
        cv2.imwrite(
            out_path(image_path, f"_{tag_prefix}_components_bbox_all"),
            combined_overlay,
        )

    return results


# ============================================================
# 6. 미래 마스크 예측용 UNet
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FutureMaskUNet(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(512 + 512, 256)
        self.up2 = Up(256 + 256, 128)
        self.up3 = Up(128 + 128, 64)
        self.up4 = Up(64 + 64, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def load_future_unet(model_path: str, device: torch.device) -> FutureMaskUNet:
    print(f"[INFO] Loading future-mask UNet from: {model_path}")
    state = torch.load(model_path, map_location=device)
    model = FutureMaskUNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_future_mask_tiled(
    model: FutureMaskUNet,
    now_mask: np.ndarray,
    device: torch.device,
    pred_thresh: float = FUTURE_PRED_THRESH_DEFAULT,
    tile: int = FUTURE_TILE_DEFAULT,
    stride: int = FUTURE_STRIDE_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    m = preprocess_crack_mask(now_mask)
    h, w = m.shape[:2]

    prob_full = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    m_u8 = (m * 255).astype(np.uint8)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = m_u8[y:y+tile, x:x+tile]
            if patch.size == 0:
                continue

            ph, pw = patch.shape[:2]
            pad_bottom = max(0, tile - ph)
            pad_right = max(0, tile - pw)

            patch_pad = cv2.copyMakeBorder(
                patch,
                0, pad_bottom,
                0, pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )

            inp = patch_pad.astype(np.float32) / 255.0
            inp = torch.from_numpy(inp)[None, None, ...].to(device)

            with torch.no_grad():
                logits = model(inp)
                prob_small = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)

            prob_patch = prob_small[:ph, :pw]

            prob_full[y:y+ph, x:x+pw] += prob_patch
            weight[y:y+ph, x:x+pw] += 1.0

    weight[weight == 0] = 1.0
    prob_full /= weight

    future_bin = (prob_full >= pred_thresh).astype(np.uint8)

    mean_prob = float(prob_full.mean())
    fg_ratio = float(future_bin.mean())

    return future_bin, prob_full, mean_prob, fg_ratio


# ============================================================
# 7. 메인 파이프라인 (하드코딩 버전)
# ============================================================

def run_pipeline():
    # ======== 여기를 너 환경에 맞게 수정 =========
    IMAGE_PATH = "a.png"                    # 분석할 이미지
    MM_PER_PIXEL = DEFAULT_MM_PER_PIXEL     # mm/px
    FUTURE_MODEL_PATH = "future_mask_unet_h4_v2.pth"  # 미래 UNet 가중치 경로
    # ======================================

    ensure_out_dir()

    device = infer_device()
    device_str = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

    # 1) 모델 로딩 (YOLO, Vizuara UNet, Future UNet)
    yolo_model = load_yolo_from_hf(device_str=device_str)
    viz_unet = load_vizuara_unet(device)
    future_unet = load_future_unet(FUTURE_MODEL_PATH, device)

    # 2) YOLO ROI 마스크
    yolo_roi_mask = generate_yolo_roi_mask_seg(IMAGE_PATH, yolo_model)

    # 3) UNet+YOLO 확률 앙상블 + GMM → base now 마스크
    unet_now_mask = generate_crack_mask_unet_gmm(
        IMAGE_PATH,
        viz_unet,
        device=device,
        yolo_roi_mask=yolo_roi_mask,
        yolo_boost=YOLO_BOOST_ALPHA,
    )

    # 4) 원형/잡음 제거 → 최종 now 마스크
    final_now_mask = filter_crack_like_components(unet_now_mask, IMAGE_PATH)

    # 최종 now 오버레이 저장
    img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    now_overlay = img_bgr.copy()
    now_overlay[final_now_mask > 0] = (0, 0, 255)
    blended_now = cv2.addWeighted(img_bgr, 0.7, now_overlay, 0.3, 0)
    cv2.imwrite(out_path(IMAGE_PATH, "_now_final_overlay"), blended_now)

    # 5) 글로벌 now 위험도 분석
    now_global = analyze_mask(
        IMAGE_PATH,
        final_now_mask,
        mm_per_pixel=MM_PER_PIXEL,
        tag="now_global",
    )

    # 6) 컴포넌트별 now 위험도 분석 (+ bbox 이미지)
    now_components = analyze_components(
        IMAGE_PATH,
        final_now_mask,
        mm_per_pixel=MM_PER_PIXEL,
        tag_prefix="now_comp",
    )

    # 7) 미래 마스크 예측
    future_bin, future_prob, mean_prob, fg_ratio = predict_future_mask_tiled(
        future_unet,
        final_now_mask,
        device=device,
        pred_thresh=FUTURE_PRED_THRESH_DEFAULT,
        tile=FUTURE_TILE_DEFAULT,
        stride=FUTURE_STRIDE_DEFAULT,
    )

    cv2.imwrite(out_path(IMAGE_PATH, "_future_prob"), (future_prob * 255).astype(np.uint8))
    cv2.imwrite(out_path(IMAGE_PATH, "_future_mask"), future_bin * 255)

    future_overlay = img_bgr.copy()
    future_overlay[future_bin > 0] = (255, 0, 0)
    blended_future = cv2.addWeighted(img_bgr, 0.7, future_overlay, 0.3, 0)
    cv2.imwrite(out_path(IMAGE_PATH, "_future_final_overlay"), blended_future)

    print("\n[INFO] 미래 예측 요약")
    print(f"  - mean_prob (평균 예측 확률)     : {mean_prob:.4f}")
    print(f"  - fg_ratio (미래 크랙 비율, 0~1) : {fg_ratio:.4f}")

    # 8) 글로벌 future 위험도 분석
    future_global = analyze_mask(
        IMAGE_PATH,
        future_bin,
        mm_per_pixel=MM_PER_PIXEL,
        tag="future_global",
    )

    # 9) 컴포넌트별 future 위험도 분석 (+ bbox 이미지)
    future_components = analyze_components(
        IMAGE_PATH,
        future_bin,
        mm_per_pixel=MM_PER_PIXEL,
        tag_prefix="future_comp",
    )

    # 10) 요약 출력
    print("\n===== SUMMARY (GLOBAL) =====")
    print(f"Image         : {IMAGE_PATH}")
    print(f"mm_per_pixel  : {MM_PER_PIXEL}")
    print("---- NOW GLOBAL ----")
    print(f"S_now = {now_global.S:.3f}")
    print(f"P_now = {now_global.P:.3f}")
    print(f"R_now = {now_global.R:.3f}, Level={now_global.level}")
    print("---- FUTURE GLOBAL ----")
    print(f"S_future = {future_global.S:.3f}")
    print(f"P_future = {future_global.P:.3f}")
    print(f"R_future = {future_global.R:.3f}, Level={future_global.level}")
    print("---- Δ(변화량) ----")
    print(f"ΔS = {future_global.S - now_global.S:+.3f}")
    print(f"ΔR = {future_global.R - now_global.R:+.3f}")

    print("\n===== SUMMARY (COMPONENT COUNT) =====")
    print(f"now_components    : {len(now_components)}")
    print(f"future_components : {len(future_components)}")
    print("\n[INFO] 모든 결과 이미지는 out/ 폴더에 저장됨.")


if __name__ == "__main__":
    run_pipeline()