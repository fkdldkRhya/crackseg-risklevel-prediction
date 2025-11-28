# Crack Future Risk Analysis Pipeline

<div align="left">
스마트 건물 안전 관리를 위해 Vision & AI 기술을 활용한 콘크리트 크랙 탐지에 관한 연구
</div>

<div align="left">
|                                              최시훈                                              |
| :----------------------------------------------------------------------------------------------:
|                  <img width="160px" src="https://github.com/fkdldkrhya.png" />                  |
</div>

---

**Affiliation**

- Department of Smart Convergence Engineering, Hanyang Univ. ERICA

**Contact**

- 📧 fkdldkrhya@hanyang.ac.kr

**Research Interests**

- AI
- Computer Vision

---

(now + future U-Net, CrackSeq 기반)

콘크리트 크랙 이미지를 입력으로 받아서:

1. **현재(now) 크랙 마스크**를 만들고
2. 현재 크랙에 대한 **Severity(S)**, **Probability(P)**, **Risk(R)**를 계산한 뒤
3. CrackSeq GT 마스크로 학습된 **Future-mask U-Net**으로  
   **미래(future) 크랙 마스크**를 예측하고
4. 동일한 방식으로 미래 크랙의 S, P, R을 계산
5. 마지막으로 **현재 vs 미래 위험도 변화를 비교 (ΔS, ΔR)**

까지 수행하는 전체 파이프라인입니다.

구성 파일:

- 학습 스크립트: `future_unet_train.py.py`
- 추론/위험도 평가 스크립트: `risklevel_prediction.py`

---

## 1. 설치 및 요구 사항

### 1.1 Python 및 라이브러리

- Python 3.8+
- PyTorch
- NumPy
- OpenCV (`opencv-python`)
- scikit-learn (GaussianMixture)
- Pillow
- torchvision
- huggingface_hub

```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python scikit-learn pillow huggingface_hub
```

---

## 2. 데이터 및 디렉토리 구조

### 2.1 학습용 데이터 구조 (`future_unet_train.py.py`)

```text
data_root/
  ├─ images/           # (현재 버전에서는 사용하지 않아도 되지만, 확장 가능)
  └─ targets/
       ├─ multi_target_0001.npy
       ├─ multi_target_0002.npy
       └─ ...
```

- `targets/multi_target_*.npy` 만 사용
- 각 `multi_target_XXXX.npy`:
  - shape: `(T, H, W)` 또는 `(T, H, W, 1)` (또는 방어적으로 `(T, H, W, 3)`이면 첫 채널 사용)
  - `T`: multi-temporal 시점 수
  - `vol[t]` = 시점 `t`의 GT 크랙 마스크

### 2.2 마스크 전처리 규칙 (공통)

모든 마스크는 다음 규칙을 따릅니다.

1. `float32`로 변환
2. max 값이 1보다 크면 `0~255`라고 가정하고 `/255.0` 으로 정규화
3. `binarize_threshold` (기본 0.5) 이상이면 1, 미만이면 0으로 이진화
4. 최종 마스크는 **0.0 또는 1.0 값만 가지는 이진 마스크**

이 규칙은:

- 학습용 GT (`FutureMaskDataset`)
- 현재/미래 위험도 분석 (`analyze_mask`)
- 타일링 예측 (`predict_future_mask_tiled`)

에서 일관되게 적용됩니다.

---

## 3. Future-mask U-Net 학습 (`future_unet_train.py.py`)

### 3.1 목적

- CrackSeq GT 마스크만 사용해서:

  - 입력: 현재 마스크 `mask_t`
  - 출력: Δ 스텝 이후 마스크 `mask_{t+Δ}`

  를 예측하는 **2D U-Net**을 학습합니다.

- 원본 intensity 이미지는 사용하지 않고, **마스크→마스크 순수 변환**에 집중합니다.
- 크랙 픽셀 비율이 매우 낮은 불균형 데이터 상황을 고려하여:
  - **너무 쉬운 샘플 제거 (future mask foreground 비율이 너무 낮은 경우)**
  - **foreground 비율 기반 `pos_weight` 자동 추천**
  - **Dice 비중을 높인 BCE+Dice 복합 손실 사용**

### 3.2 FutureMaskDataset (mask_now, mask_future 쌍 생성)

클래스: `FutureMaskDataset`

- 입력: `data_root`, `horizon_steps`, `min_fg_ratio`, `binarize_threshold`
- 주요 로직:

1. `targets/multi_target_*.npy` 전체 스캔
2. 각 파일에 대해:

   - `vol` shape: `(T, H, W)` 형태로 정규화
   - 시간축 `t = 0 .. T-1` 순회
   - `t_future = t + horizon_steps`
   - `t_future < T` 인 경우만 후보

3. `mask_future = vol[t_future]` 를 전처리 후 foreground 비율(`fg_ratio`) 계산
4. `fg_ratio < min_fg_ratio`(기본 0.0005 = 0.05%)이면 **해당 쌍 제외**
   - 거의 올 배경인 trivial sample 제거
5. 조건을 통과한 `(path, t_now, t_future)` 쌍만 `self.index_tuples`에 저장
6. 동시에 전체 future 마스크에 대해 `total_foreground`, `total_pixels`를 누적하여  
   전역 `fg_ratio` 계산

#### pos_weight 자동 추정

Dataset 생성 시:

```text
fg_ratio = total_foreground / total_pixels
pos_weight ≈ (1 - fg_ratio) / fg_ratio
```

이 값은 `BCEWithLogitsLoss(pos_weight=...)`에 사용될 **권장값**입니다.

- `--pos_weight_override` 가 `None`이면 Dataset에서 계산한 값 사용
- 직접 수동으로 지정하고 싶으면 `--pos_weight_override` 인자로 덮어쓰기

### 3.3 U-Net2D 구조 (학습용)

`UNet2D` (학습용) 요약:

- 입력: `(B, 1, H, W)` (현재 마스크)
- 출력: `(B, 1, H, W)` (Δ 이후 미래 마스크 logits)
- 인코더:
  - `DoubleConv(1→64)`
  - `Down(64→128)`
  - `Down(128→256)`
  - `Down(256→512)`
  - `Down(512→1024//factor)`
- 디코더:
  - `Up(1024→512//factor)`
  - `Up(512→256//factor)`
  - `Up(256→128//factor)`
  - `Up(128→64)`
- 마지막: `OutConv(64→1)`로 채널 1개 출력 (logits)

Upsample은 bilinear + conv 구조를 사용합니다.

### 3.4 Loss 구성 (BCE + Dice)

- `BCEWithLogitsLoss(pos_weight)`
- `DiceLoss` (logits → sigmoid 후 Dice)

최종 손실:

```text
loss = 0.2 * BCEWithLogitsLoss + 0.8 * DiceLoss
```

의도:

- BCE: 픽셀 단위 분류 안정화
- Dice: 희소한 크랙 마스크의 전체 overlap을 잘 맞춤

### 3.5 학습 루프 요약

1. Train:
   - `model.train()`
   - 각 배치에 대해 forward → loss → backward → `optimizer.step()`
2. Validation:
   - `model.eval()` + `torch.no_grad()`
   - 같은 loss 조합으로 평균 val_loss 계산
3. Best 모델 저장:
   - `avg_val_loss`가 최소일 때마다 `save_path`에 `state_dict` 저장

### 3.6 실행 예시 및 인자

```bash
python future_unet_train.py.py \
  --data_root ./crackseq_dataset_multi_mono_patches/data/multi-temporal \
  --horizon_steps 4 \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_path future_mask_unet_h4_v2.pth
```

주요 인자:

- `--data_root`: `images/`, `targets/`가 포함된 루트 경로
- `--horizon_steps`: Δ (t_now → t+Δ)
- `--epochs`: epoch 수
- `--batch_size`: 배치 크기
- `--lr`: learning rate
- `--val_ratio`: train/val 비율 (기본 0.2)
- `--min_fg_ratio`: future mask foreground 비율 하한
- `--device`: `"auto"`, `"cuda"`, `"cpu"`, `"mps"` 등
- `--save_path`: best 모델 가중치 저장 경로
- `--seed`: random seed
- `--pos_weight_override`: pos_weight 수동 지정 시 사용

---

## 4. Future-mask UNet (추론용, `risklevel_prediction.py`)

`risklevel_prediction.py` 안의 `FutureMaskUNet`은 학습용 `UNet2D`와 채널 구성이 맞도록 정의되어 있습니다.

- 입력: `(B, 1, H, W)` (현재 now_mask 패치)
- 출력: `(B, 1, H, W)` (미래 mask logits)

구성 요약:

- 인코더:
  - `inc = DoubleConv(1, 64)`
  - `down1 = Down(64, 128)`
  - `down2 = Down(128, 256)`
  - `down3 = Down(256, 512)`
  - `down4 = Down(512, 512)` (bottleneck)
- 디코더:
  - `up1 = Up(512+512 → 256)`
  - `up2 = Up(256+256 → 128)`
  - `up3 = Up(128+128 → 64)`
  - `up4 = Up(64+64 → 64)`
- 출력: `outc = OutConv(64, 1)`

`load_future_unet(model_path, device)`로 가중치를 로드합니다.

---

## 5. 현재(now) 마스크 생성 (`risklevel_prediction.py`)

### 5.1 Otsu 기반 모드

```bash
--now_mask_mode otsu
```

로 설정할 때 사용됩니다.

- `load_crack_mask_from_image(image_path, invert, blur_ksize)`:
  - BGR → Gray 변환
  - 선택적으로 GaussianBlur
  - Otsu 이진화 (`THRESH_BINARY` 또는 `THRESH_BINARY_INV`)
- 결과:
  - `now_mask`: `(H, W)` 0/1 마스크 (uint8)
- 디버그 출력:
  - `out/<basename>_gray.png`
  - `out/<basename>_thresh.png`

### 5.2 Vizuara UNet + GMM 기반 모드

```bash
--now_mask_mode unet
```

로 설정할 때 사용됩니다.

#### 5.2.1 VizuaraUNet

- HuggingFace Hub:
  - repo: `Vizuara/unet-crack-segmentation`
  - weights: `unet_weights.pth`
- 3채널 입력(256x256), 1채널 출력(sigmoid) 구조

#### 5.2.2 Edge boost 전처리

함수: `crack_edge_boost(image_bgr)`

- Gray 변환
- CLAHE로 대비 향상
- Sobel(3,5) 및 Laplacian 결합
- edge + intensity 정보를 강화한 gray 이미지를 RGB로 변환해 UNet에 입력

#### 5.2.3 UNet 확률맵 생성

함수: `run_unet_prob_map(model, image_bgr, device)`

- boosted 이미지를 256x256으로 resize
- UNet forward → sigmoid → 확률맵
- 원본 해상도로 bilinear resize → `prob_map` (0~1 float)

#### 5.2.4 GMM 기반 이진화

함수: `prob_to_binary_gmm(prob_map, image_bgr, n_components, save_debug)`

- `prob_map`을 [0,1]로 정규화
- Gray에서 Sobel로 edge magnitude 계산 후 [0,1]로 정규화
- 각 픽셀의 feature = `[prob_norm, edge_norm]`
- `GaussianMixture(n_components=3)`로 학습 후 전체 픽셀에 대해 클러스터 예측
- 각 클러스터에 대해 평균 prob_norm 계산 →
  - 가장 큰 평균값을 갖는 클러스터를 main crack cluster로 선택
  - 필요 시 두 번째 cluster까지 포함
- 후처리:
  - Morphological open/erode
  - `keep_elongated_components`: 세장비/면적 기준으로 가늘고 긴 컴포넌트만 유지
  - `fill_small_holes_by_contours`: 작은 홀 영역 채우기
- 결과:
  - `0/1` now_mask (uint8)

#### 5.2.5 마스크 및 오버레이 저장

함수: `generate_crack_mask_unet_gmm(...)`

- `out/<basename>_now_unet_mask.png`
- `out/<basename>_now_unet_overlay.png` (원본 위에 붉은 크랙 영역)

---

## 6. 타일링 기반 미래 마스크 예측

함수: `predict_future_mask_tiled(model, now_mask, device, pred_thresh, tile, stride)`

- 입력:
  - `now_mask`: `(H, W)` 0/1
  - `tile`: 타일 크기 (기본 128)
  - `stride`: 슬라이딩 간격 (기본 64, overlap 존재)
- 절차:

  1. `now_mask`를 0/255 uint8로 변환
  2. `y = 0..H, x = 0..W`에 대해 stride 간격으로 타일을 잘라냄
  3. 타일 크기가 작으면 zero padding으로 (tile, tile)까지 확장
  4. 0~1 float로 정규화 후 `(1,1,tile,tile)` 텐서로 모델에 입력
  5. sigmoid(logits)로 얻은 확률맵에서 ph,pw 범위만 잘라 원위치에 더함
  6. 동일 위치에 여러 타일이 겹친 경우 `weight`를 누적해서  
     마지막에 `prob_full /= weight`로 평균

- 결과:

  - `future_prob`: 전체 이미지 크기의 0~1 확률맵
  - `future_bin`: `future_prob >= pred_thresh` 이면 1, 아니면 0
  - `mean_prob`: 확률맵의 전체 평균
  - `fg_ratio`: 미래 바이너리 마스크에서 foreground 비율

- 저장:
  - `out/<basename>_future_prob.png`
  - `out/<basename>_future_mask.png`

---

## 7. 위험도 분석 (`analyze_mask`)

현재(now)와 미래(future) 모두 같은 함수를 사용합니다.

### 7.1 Severity Features

`SeverityFeatures`:

- `w_max_mm` : 최대 크랙 폭 (mm)
- `rho_L_per_m` : 단위 폭당 크랙 길이 (1/m, 밀도)
- `area_mm2` : 크랙 면적 (mm²)
- `D_f` : 프랙탈 차원
- `length_m` : 총 크랙 길이 (m)

각 물리량은:

- distance transform, skeletonization, box-counting 등으로 계산됩니다.

### 7.2 Severity Score (0~1)

함수: `compute_severity(feats, weights)`

각 feature를 piecewise 함수로 0~1로 정규화:

- `S_w` : 폭 기준 (`WIDTH_T0_MM`, `WIDTH_T1_MM`)
- `S_density` : 밀도 기준 (`DENSITY_T0_PER_M`, `DENSITY_T1_PER_M`)
- `S_area` : 면적 기준 (`AREA_T0_MM2`, `AREA_T1_MM2`)
- `S_fractal` : 프랙탈 기준 (`FRACTAL_LOW`, `FRACTAL_HIGH`)
- `S_length` : 길이 기준 (`LENGTH_T0_M`, `LENGTH_T1_M`)

가중 평균:

```text
S_total = (α*S_w + β*S_density + γ*S_area + δ*S_fractal + ε*S_length) / (α+β+γ+δ+ε)
```

기본적으로 α~ε는 모두 1.0입니다.

### 7.3 Probability 구성요소

`ProbabilityInputs`:

- `P_TipSharpness` : tip sharpness 점수 (0~1)
- `P_WidthGradient` : 폭 변화 gradient 점수 (0~1)
- `P_Orientation` : orientation 기반 점수 (0~1)

1. **Tip sharpness**:
   - skeleton endpoint 주변과 내부 폭을 비교하여 얼마나 뾰족한지 평가
2. **Width gradient**:
   - skeleton 상의 폭 분포에서 10 percentile vs 90 percentile을 사용해  
     상대적인 폭 변화량을 정규화
3. **Orientation**:
   - skeleton 좌표에 PCA 적용 → 주 방향 벡터, 각도(angle_deg)
   - 각도로부터 orientation risk base (sin(2θ) 형태) 계산
   - 여기에 `S_length`를 곱해 `P_Orientation` 구성

최종 Probability:

```text
P = (α*P_TipSharpness + β*P_WidthGradient + γ*P_Orientation) / (α+β+γ)
```

기본적으로 α=β=γ=1.0입니다.

### 7.4 Risk 계산 및 Level 분류

함수: `compute_risk(img_path, S, P, risk_weights, mode, normalize_weights)`

두 가지 모드:

1. `mode="product"`:
   ```text
   R = S * P
   ```
2. `mode="weighted_sum"` (기본):
   ```text
   R = wS * S + wP * P   (wS + wP = 1)
   ```

S, P, R은 모두 0~1 범위로 클램프합니다.

Risk Level 분류:

- R < 0.2 → A
- R < 0.4 → B
- R < 0.7 → C
- 그 이상 → D

최종 결과는 `RiskResult`에 담겨 반환되며, 콘솔에는 상세 로그를 출력합니다.

또한, `analyze_mask` 실행 시:

- 마스크: `<basename>_<tag>_mask.png`
- 컬러맵: `<basename>_<tag>_mask_color.png`
- skeleton: `<basename>_<tag>_skeleton.png`
- skeleton + 주 방향: `<basename>_<tag>_skeleton_orient.png`

등이 `out/`에 저장됩니다.

---

## 8. 메인 파이프라인 (`risklevel_prediction.py`)

메인 함수 흐름:

1. 인자 파싱
   - `image_path`
   - `mm_per_pixel`
   - `model_path` (future-mask U-Net 가중치)
   - `pred_thresh`, `tile`, `stride`
   - `now_mask_mode` ("otsu" 또는 "unet")
   - `gmm_components`, `save_debug`
2. 디바이스 선택 (`cuda` / `mps` / `cpu`)
3. 현재(now) 크랙 마스크 생성
   - `now_mask_mode=="otsu"` → Otsu
   - `now_mask_mode=="unet"` → Vizuara UNet + GMM
4. `analyze_mask(..., tag="now")`로 현재 위험도 분석
5. `load_future_unet(model_path, device)`로 future-mask U-Net 로딩
6. `predict_future_mask_tiled(...)`로 미래 확률맵/마스크 예측
7. `analyze_mask(..., tag="future")`로 미래 위험도 분석
8. 콘솔에 요약 출력:

```text
===== SUMMARY =====
Image: <image_path>
mm_per_pixel: <val>
---- NOW ----
S_now = ...
P_now = ...
R_now = ..., Level=...
---- FUTURE ----
S_future = ...
P_future = ...
R_future = ..., Level=...
---- Δ(변화량) ----
ΔS = ...
ΔR = ...
```

---

## 9. 실행 예시

### 9.1 Future-mask U-Net 학습

```bash
python future_unet_train.py.py \
  --data_root ./crackseq_dataset_multi_mono_patches/data/multi-temporal \
  --horizon_steps 4 \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_path future_mask_unet_h4_v2.pth
```

### 9.2 현재+미래 위험도 평가 (UNet+GMM 기반 now-mask)

```bash
python risklevel_prediction.py G.png \
  --mm_per_pixel 0.1 \
  --model_path future_mask_unet_h4_v2.pth \
  --now_mask_mode unet \
  --tile 128 \
  --stride 64 \
  --pred_thresh 0.5 \
  --save_debug
```

### 9.3 현재+미래 위험도 평가 (Otsu 기반 now-mask)

```bash
python risklevel_prediction.py G.png \
  --mm_per_pixel 0.1 \
  --model_path future_mask_unet_h4_v2.pth \
  --now_mask_mode otsu
```

---

## 10. 튜닝 팁

- `horizon_steps`
  - 더 큰 Δ: 더 먼 미래 예측 → 난이도/불확실성 증가
- `min_fg_ratio`
  - 너무 크면 샘플 수가 줄어들고, 너무 작으면 trivial 샘플이 많아짐
- `pos_weight`
  - 자동값을 기본으로 사용하되, 필요 시 `--pos_weight_override`로 조정
- `tile`, `stride`
  - 작은 stride (overlap 많음): 매끄러운 예측 but 느림
  - 큰 stride: 빠르지만 타일 경계 아티팩트 가능
- Severity/Probability/Risk 가중치
  - 도메인 관점에서 폭/길이/밀도/각도 등에 가중치를 다르게 줄 수 있음

---

이 README는 `future_unet_train.py.py`와 `risklevel_prediction.py`의 전체 동작을  
한 번에 이해하고 재현할 수 있도록 정리한 문서입니다.

# LICENSE

BSD 2-Clause License

Copyright (c) 2023, CHOI SI-HUN

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
