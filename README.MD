# UNet-Base Crack Future Risk Analysis Pipeline

_(YOLOv8 Segmentation + Vizuara UNet + Future-mask UNet All-in-one)_

<div align="left">
스마트 건물 안전 관리를 위해 Vision &amp; AI 기술을 활용한 콘크리트 크랙 탐지 및 미래 위험도 예측 파이프라인
</div>

**Affiliation**

- 최시훈 (CHOI SIHUN)
- Department of Smart Convergence Engineering, Hanyang Univ. ERICA

**Contact**

- 📧 fkdldkrhya@hanyang.ac.kr

**Research Interests**

- AI / Computer Vision / Structural Health Monitoring

---

## 0. 전체 파이프라인 개요

이 리포지토리는 **단일 스크립트** `risklevel_prediction.py` 안에서  
다음 과정을 **end-to-end**로 수행하는 파이프라인을 구현합니다.

1. 입력 콘크리트 이미지에서 **YOLOv8 세그멘테이션**으로 크랙 후보 **ROI 마스크** 생성
2. 전체 이미지를 **Vizuara UNet**에 넣어 **크랙 확률맵(UNet_Prob)** 생성
3. YOLO ROI 정보를 이용해 **확률맵을 boost** 하는 가중치 앙상블

   \[
   \text{Final_Prob} = \mathrm{clip}(\text{UNet_Prob} + \alpha \cdot \text{YOLO_Mask}, 0, 1)
   \]

   - `YOLO_BOOST_ALPHA = α` (기본 0.30)

4. **Final_Prob + 에지(edge) 특징**을 사용해 **GMM 클러스터링**으로 이진화 → **base now 마스크**
5. 너무 작은 blob / 원형에 가까운 blob / aspect가 작은 잡음을 제거 → **최종 now 마스크**
6. 최종 now 마스크를 **future-mask UNet**에 넣어, 타일링 기반으로 **미래 크랙 마스크** 예측
7. now / future 마스크 각각에 대해:

   - **글로벌 위험도 분석** (Severity S, Probability P, Risk R, Level A~D)
   - **컴포넌트별(가지별) 위험도 분석** 및 bounding box 시각화

8. 마지막으로 **현재 vs 미래 위험도 변화** (ΔS, ΔR)를 계산하고 콘솔에 요약 출력

---

## 1. 파일 구성

현재 버전의 핵심은 다음 한 파일입니다.

- `risklevel_prediction.py`
  - YOLOv8 ROI 세그멘테이션
  - Vizuara UNet 확률맵 + YOLO boost + GMM 이진화
  - 원형/잡음 제거 및 컴포넌트 분리
  - future-mask UNet 타일링 추론
  - now/future 글로벌 & 컴포넌트별 위험도 분석
  - 결과 이미지 저장 & 콘솔 요약 출력

추가로 필요한 모델 가중치:

- `future_mask_unet_h4_v2.pth`
  - CrackSeq 기반 **미래 크랙 마스크** 예측용 U-Net 가중치
  - 로딩 위치는 스크립트 상단의 `FUTURE_MODEL_PATH` 설정을 따릅니다.

> **참고**: 미래 U-Net 학습 스크립트(`future_unet_train.py.py`)는 별도 파일로 존재하며,  
> CrackSeq multi-temporal GT 마스크를 이용해 `future_mask_unet_h4_v2.pth`를 학습합니다.  
> (이 README에서는 **추론 파이프라인** 위주로 설명합니다.)

---

## 2. 설치 및 요구 사항

### 2.1 Python 버전

- Python **3.8+** 권장

### 2.2 필수 의존성

```bash
pip install torch torchvision torchaudio

pip install   numpy   opencv-python   scikit-learn   pillow   huggingface_hub   ultralytics
```

선택(있으면 더 좋음):

```bash
# skeletonize를 위해 ximgproc thinning을 쓰고 싶다면
pip install opencv-contrib-python
```

> `opencv-contrib-python`이 없더라도, 스크립트 내부에서 **fallback skeletonization** 로직을 사용하므로 동작은 가능합니다.

---

## 3. 사용 모델 & 체크포인트

### 3.1 Vizuara UNet (현재 마스크용)

- Hugging Face Hub: `Vizuara/unet-crack-segmentation`
- 가중치 파일명: `unet_weights_v2.pth`
- 스크립트 내에서 `hf_hub_download`로 **자동 다운로드**됩니다.

역할:

- 크랙 edge에 특화된 전처리(`crack_edge_boost`)를 거친 이미지를 입력으로 받아,
- 픽셀 단위 크랙 확률맵 `prob_map_unet (H, W, 0~1)` 생성

### 3.2 YOLOv8 Segmentation (ROI 마스크용)

- Hugging Face Hub: `OpenSistemas/YOLOv8-crack-seg`
- 기본 가중치 경로: `"yolov8x/weights/best.pt"`
- 스크립트 내부:

  ```python
  YOLO_REPO_ID = "OpenSistemas/YOLOv8-crack-seg"
  YOLO_WEIGHT_FILENAME = "yolov8x/weights/best.pt"
  ```

역할:

- 입력 이미지를 YOLOv8 세그멘테이션 모델에 통과시켜,
- 검출된 모든 마스크를 OR한 결과를 **ROI 마스크(0/1)**로 사용
- 이 ROI가 Vizuara UNet 확률맵을 **boost**하는 데 사용됨

### 3.3 Future-mask U-Net (미래 마스크 예측용)

- 로컬 파일: `future_mask_unet_h4_v2.pth`
- 네트워크 구조: 스크립트 내 `FutureMaskUNet`
  - 입력: `(1, 1, H, W)` 형태의 **now 마스크 patch**
  - 출력: `(1, 1, H, W)` 형태의 **미래 마스크 logits**

역할:

- 현재(now) 마스크를 타일 단위로 입력받아,
- CrackSeq multi-temporal GT를 통해 학습된 미래 크랙 분포를 예측

---

## 4. 파이프라인 상세 설명

### 4.1 YOLOv8 ROI 마스크 생성

함수: `generate_yolo_roi_mask_seg(image_path, yolo_model)`

- YOLOv8 segmentation으로 inference 수행
- 결과 마스크들을 모두 OR하여 단일 ROI 마스크 생성
- ROI 마스크를 원본 해상도로 리사이즈 (`(H, W)` 0/1)
- 저장 파일:
  - `<base>_yolo_roi_mask.png`
  - `<base>_yolo_roi_overlay.png` (원본 위에 ROI 영역을 녹색으로 표시)

### 4.2 Vizuara UNet 확률맵 + YOLO boost + GMM 이진화

#### 4.2.1 Edge-boosted 입력 생성

함수: `crack_edge_boost(image_bgr)`

- Gray 변환 → CLAHE로 대비 향상
- Sobel + Laplacian 결합으로 edge 강화
- edge + intensity를 혼합한 gray 이미지를 RGB로 변환하여 UNet 입력으로 사용

#### 4.2.2 UNet 확률맵 생성

함수: `run_unet_prob_map(model, image_bgr, device)`

- 256×256으로 resize → Vizuara UNet 추론 → sigmoid
- 원본 해상도로 bilinear resize하여 `prob_map_unet (H, W, 0~1)` 생성

#### 4.2.3 YOLO boost 기반 앙상블

함수: `generate_crack_mask_unet_gmm(...)`

- YOLO ROI가 있다면:

  ```python
  prob_map_final = prob_map_unet + YOLO_BOOST_ALPHA * YOLO_Mask
  prob_map_final = np.clip(prob_map_final, 0.0, 1.0)
  ```

- `YOLO_BOOST_ALPHA`를 조절하여:
  - YOLO에 더 의존 (↑): ROI 안에서 확률이 강하게 상승
  - UNet에 더 의존 (↓): 전역 세그멘테이션에 더 의존

디버그 출력:

- `<base>_unet_prob_raw.png`
- `<base>_unet_prob_boosted.png`

#### 4.2.4 GMM 이진화 + 길게 뻗은 컴포넌트 유지

함수: `prob_to_binary_gmm(prob_map_final, image_bgr, n_components)`

- 특징 벡터: `[Final_Prob_norm, Edge_norm]`
- `GaussianMixture(n_components=3)`로 클러스터링
- 평균 확률이 높은 클러스터를 크랙 클러스터로 선택 (필요 시 2개까지)
- Morphological open / erode로 가벼운 잡음 제거
- `keep_elongated_components`:
  - 세장비(aspect ratio), 정상화 길이(norm_len), 최소 면적 기준으로  
    **가늘고 긴 컴포넌트**만 유지
- `fill_small_holes_by_contours`:
  - 전체 이미지 대비 작은 비율의 홀만 채워 내부를 메움

최종 결과:

- `unet_now_mask (H, W, 0/1)`

저장:

- `<base>_now_base_mask.png`
- `<base>_now_base_overlay.png`

### 4.3 원형/잡음 제거 → 최종 now 마스크

함수: `filter_crack_like_components(combined_mask, image_path)`

- 각 connected component에 대해:
  - 면적(area) < `MIN_COMP_AREA` → 제거
  - 긴 변(long_axis) < `MIN_LONG_AXIS` & aspect < `MIN_ASPECT_RATIO` → 제거
  - **원형성(circularity)**:
    - \(\mathcal{C} = 4\pi A / P^2\)
    - \(\mathcal{C} > \text{MAX_CIRCULARITY}\) & aspect < `MIN_ASPECT_RATIO` → 제거
- 나머지를 모두 합쳐 최종 now 마스크 생성

저장:

- `<base>_now_mask_raw.png`
- `<base>_now_mask_filtered.png`
- `<base>_now_final_overlay.png` (원본 위에 최종 now 마스크 표시)

### 4.4 타일링 기반 미래 마스크 예측

함수: `predict_future_mask_tiled(model, now_mask, device, pred_thresh, tile, stride)`

- now 마스크를 `tile × tile` 크기 패치로 슬라이딩 (stride 간격)
- 패치가 작으면 zero padding
- 각 패치를 `[0,1]`로 정규화 후 Future U-Net에 통과
- 확률맵을 원위치에 accumulate 후 `weight`로 정규화
- 최종:

  - `future_prob (H, W, 0~1)`
  - `future_bin = (future_prob >= pred_thresh)`

저장:

- `<base>_future_prob.png`
- `<base>_future_mask.png`
- `<base>_future_final_overlay.png`

콘솔:

- `mean_prob`: 전체 평균 예측 확률
- `fg_ratio`: 미래 마스크 foreground 비율

### 4.5 글로벌 & 컴포넌트별 위험도 분석

공통 함수: `analyze_mask(image_path, crack_mask, mm_per_pixel, tag)`

#### 4.5.1 Severity (S, 0~1)

`SeverityFeatures`:

- `w_max_mm` : 최대 크랙 폭 (mm)
- `rho_L_per_m` : 단위 폭당 크랙 길이 (선형 밀도, 1/m)
- `area_mm2` : 크랙 면적 (mm²)
- `D_f` : 프랙탈 차원
- `length_m` : 총 크랙 길이 (m)

각 feature는 아래 임계값을 기준으로 0~1로 정규화됩니다.

- 폭: `WIDTH_T0_MM`, `WIDTH_T1_MM`
- 밀도: `DENSITY_T0_PER_M`, `DENSITY_T1_PER_M`
- 면적: `AREA_T0_MM2`, `AREA_T1_MM2`
- 프랙탈: `FRACTAL_LOW`, `FRACTAL_HIGH`
- 길이: `LENGTH_T0_M`, `LENGTH_T1_M`

`SeverityScores`:

- `S_w`, `S_density`, `S_area`, `S_fractal`, `S_length`
- 가중 평균:

  \[
  S*\text{total} = \frac{\alpha S_w + \beta S*{density} + \gamma S*{area} + \delta S*{fractal} + \epsilon S\_{length}}{\alpha+\beta+\gamma+\delta+\epsilon}
  \]

#### 4.5.2 Probability (P, 0~1)

`ProbabilityInputs`:

- `P_TipSharpness` : tip sharpness 기반 (끝단이 얼마나 뾰족한지)
- `P_WidthGradient` : skeleton 상 폭 변화(gradient) 기반
- `P_Orientation` : 크랙 방향(angle) 및 길이에 따른 위험도

계산 방식:

1. skeleton endpoint 주변/내부 폭을 비교하여 tip이 가늘수록 점수 ↑
2. skeleton 폭 분포의 10/90 percentile 차이를 통해 폭 변화량 정규화
3. PCA 기반 주 방향을 구해 angle → orientation risk base (sin(2θ) 형태)  
   여기에 `S_length`를 곱해 `P_Orientation` 구성

최종:

\[
P = \frac{\alpha P*{Tip} + \beta P*{Grad} + \gamma P\_{Orient}}{\alpha+\beta+\gamma}
\]

기본적으로 α=β=γ=1입니다.

#### 4.5.3 Risk (R, 0~1) 및 Level (A~D)

함수: `compute_risk(img_path, S, P, mode="weighted_sum")`

- `mode="product"`: \(R = S \cdot P\)
- `mode="weighted_sum"` (기본):

  \[
  R = w_S S + w_P P,\quad w_S + w_P = 1
  \]

Risk Level 분류:

- `R < 0.2` → Level A
- `0.2 ≤ R < 0.4` → Level B
- `0.4 ≤ R < 0.7` → Level C
- `0.7 ≤ R` → Level D

#### 4.5.4 컴포넌트별 분석 & BBox 시각화

함수: `analyze_components(image_path, mask, mm_per_pixel, tag_prefix)`

- connected component를 하나의 크랙 가지로 보고 각 가지에 대해 `analyze_mask` 수행
- 각 컴포넌트에 대해:
  - 개별 마스크 & skeleton & orientation 이미지
  - 원본 위에 해당 컴포넌트만 붉게 표시 + bounding box(초록색) 표시
  - 파일명 예: `<base>_now_comp_c3_bbox.png`
- 전체 컴포넌트 bbox를 한 장에 모은 이미지도 생성:
  - `<base>_now_comp_components_bbox_all.png`
  - `<base>_future_comp_components_bbox_all.png`

---

## 5. 메인 실행 방법

### 5.1 스크립트 내 하드코딩 파라미터

`run_pipeline()` 상단에서 다음을 수정합니다.

```python
def run_pipeline():
    # ======== 여기를 너 환경에 맞게 수정 =========
    IMAGE_PATH = "a.png"                    # 분석할 이미지 경로
    MM_PER_PIXEL = DEFAULT_MM_PER_PIXEL     # 이미지 스케일 (mm/px)
    FUTURE_MODEL_PATH = "future_mask_unet_h4_v2.pth"  # 미래 UNet 가중치 경로
    # ======================================
```

- `IMAGE_PATH` : 분석할 콘크리트 이미지 파일
- `MM_PER_PIXEL` : 픽셀당 mm (없으면 1.0으로 두고 상대 비교용으로 사용)
- `FUTURE_MODEL_PATH` : future-mask U-Net 가중치 파일 위치

### 5.2 실행

```bash
python risklevel_prediction.py
```

### 5.3 출력 결과

- 모든 결과 이미지는 **`out/` 폴더**에 저장됩니다.
- 콘솔에는 다음과 같은 요약이 출력됩니다.

```text
===== SUMMARY (GLOBAL) =====
Image         : a.png
mm_per_pixel  : 0.10
---- NOW GLOBAL ----
S_now = ...
P_now = ...
R_now = ..., Level=...
---- FUTURE GLOBAL ----
S_future = ...
P_future = ...
R_future = ..., Level=...
---- Δ(변화량) ----
ΔS = ...
ΔR = ...

===== SUMMARY (COMPONENT COUNT) =====
now_components    : N_now
future_components : N_future

[INFO] 모든 결과 이미지는 out/ 폴더에 저장됨.
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
