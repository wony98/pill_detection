# 3팀 모델 비교 실험 설계서 v4

> 팀원 각자 환경(GPU, 데이터 경로 등)은 자유롭게 구성하고, **결과 비교용 핵심 조건**만 통일합니다.

**v4 변경점**: 증강 Top-3 확정 (3-1 섹션), 프레임워크별 구현 코드 추가 (3-2 섹션)

---

## 0. 배경

팀원이 각자 모델 2개씩(총 10개)을 선정하여 학습하고, 실사용 평가셋 33장을 기준으로 최고 성능 모델 1개를 선발합니다.

---

## 1. 공통 입력 데이터

- **dataset_yolo v2** (팀 전체 동일 사용, 이미 공유됨)
  - train 8,302장 / val 2,090장 / 118 classes
  - Split: GroupShuffleSplit by combo, val_ratio=0.2, seed=42
- **eval_package.zip** (팀 공유됨)
  - 실사용 33장 + answer_key.json

---

## 2. 학습 공통 조건 (필수 고정)

```
epochs        = 15
imgsz         = 640
batch         = 16        (GPU 메모리 맞춰 조정 가능, 실제 값 기록 필수)
seed          = 42        (재현성 검증 시 seed=100 추가)
pretrained    = True      (각 모델 공식 기본 가중치)
```

- Optimizer / learning rate / scheduler: **각 프레임워크 기본값을 사용합니다** (통일하지 않습니다)
- 학습 환경(GPU, 로컬/Colab/Kaggle)은 **팀원별 환경에 맞춥니다**

---

## 3. 증강 (필수 고정)

### 3-1. 적용할 증강 Top-3

11개 기법 ablation(Baseline + B1~B3/C1~C3/D1~D2/E1~E2/F) 결과, Baseline 대비 Δ Top-1 기준 상위 3개 기법을 모든 모델에 공통 적용:

| 순위 | 기법 | 파라미터 | Δ Top-1 | Δ Top-3 |
|---|---|---|---|---|
| 🥇 | **CLAHE** (E1) | `clip_limit=4.0, tile_grid_size=(8,8), p=0.5` | +0.121 | +0.242 |
| 🥈 | **GaussianBlur** (C1) | `blur_limit=7, p=0.3` | +0.121 | +0.182 |
| 🥉 | **MotionBlur** (C2) | `blur_limit=7, p=0.3` | +0.121 | +0.182 |

Baseline (증강 없음 기본): Top-1 0.091 / Top-3 0.091

선정 근거: Δ Top-1 동점 3개 발생 → Δ Top-3로 tie-break. 11개 전부 양의 기여지만 Top-3가 가장 큰 차이(절대 수치 33장 중 4장 추가 적중, 9.1% → 21.2%).

### 3-2. 프레임워크별 구현

**원칙**: 수학적 완전 동일 불가, "같은 방향의 증강"만 보장. bbox/mask 함께 변환되는 구현만 사용.

#### ultralytics (YOLO / RT-DETR) — Albumentations 몽키패치

```python
import sys
# 이전 패치 흔적 제거 (재실행 대응)
mods = [k for k in sys.modules.keys() if k.startswith('ultralytics')]
for m in mods:
    del sys.modules[m]

from ultralytics.data.augment import Albumentations
import albumentations as A

_original_init = Albumentations.__init__

def _patched_init(self, p=1.0, transforms=None):
    custom_transforms = [
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.GaussianBlur(blur_limit=7, p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
    ]
    _original_init(self, p=p, transforms=custom_transforms)

Albumentations.__init__ = _patched_init
```

그 후 기본 방식대로 `model.train(data='pill.yaml', epochs=15, imgsz=640, batch=16, seed=42)`.

#### torchvision (Faster R-CNN / RetinaNet / SSD) — Albumentations 래핑

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    A.GaussianBlur(blur_limit=7, p=0.3),
    A.MotionBlur(blur_limit=7, p=0.3),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Dataset __getitem__ 안에서:
transformed = train_transform(image=img_np, bboxes=boxes, labels=labels)
img_tensor = transformed['image']
boxes = transformed['bboxes']
labels = transformed['labels']
```

#### rfdetr — albumentations 직접 적용

rfdetr 공식 API가 albumentations 지원. 구현 방식은 위 torchvision 방식과 동일 (bbox_params는 COCO 포맷 사용).

#### segmentation (SAM / YOLOv11-seg)

mask/bbox 같이 변환되는 것만 사용. albumentations 사용 시 `A.Compose(..., additional_targets={'mask': 'mask'})` 로 mask도 함께 변환.

---

## 4. 평가 공통 조건 (필수 고정)

### 4-1. 평가 대상
- 33장 전부, 이미지당 bbox + class Top-3 예측

### 4-2. 채점 로직 (이름 기반 필수)

**핵심**: class_idx 매핑이 모델마다 다르므로 **`dl_name` 문자열 매칭**으로 채점.

```python
# 의사코드
for image in test_images:
    gt_name = answer_key[image]['dl_name']
    predictions = model.predict(image)  # top-3 class names

    if gt_name in predictions[0]:        # Top-1 hit
        top1_hit += 1
    if gt_name in predictions[0:3]:      # Top-3 hit
        top3_hit += 1

top1_accuracy = top1_hit / 33
top3_accuracy = top3_hit / 33
detection_rate = detected_images / 33
```

### 4-3. Segmentation 모델 특수 처리
- mask → bbox 변환 (`cv2.boundingRect` 또는 mask xyxy 추출)
- 변환된 bbox로 동일 채점 로직 적용

---

## 5. 선발 기준 (필수 고정)

### 5-1. 1차 필터 (자격 요건)
- ☑ 학습 완료 (학습 커브 스크린샷 제출)
- ☑ 33장 Detection rate ≥ 0.8
- ☑ 33장 미탐지 ≤ 5장

### 5-2. 2차 주 지표
- **Top-1 정확도 (33장)** 최고

### 5-3. 3차 근접 처리
Top-1 차이 **±6%p (±2장)** 이내면:
1. Top-3 정확도로 tie-break

### 5-4. 4차 재현성 검증
- 상위 2-3개 후보 → **seed=100으로 추가 학습 1회**
- seed=42와 seed=100 **모두 상위권**이어야 통과

### 5-5. 5차 최종
- 4차 통과 모델 중 **2회 Top-1 평균 최고**

---

## 6. 산출물 (필수 제출)

### 6-1. 필수 제출
1. **결과 JSON** (아래 형식 준수)
2. **학습 커브 스크린샷** (loss, mAP 등)
3. **학습 설정 기록** (args.yaml 또는 하이퍼파라미터)

### 6-2. JSON 형식 (필수 준수)
```json
{
  "exp_name": "faster_rcnn",
  "author": "wony98",
  "experiment_type": "model",
  "model_framework": "torchvision",
  "model_params_M": 41.3,
  "batch_size_used": 8,
  "num_images": 33,
  "detection_rate": 0.85,
  "top1_accuracy": 0.52,
  "top3_accuracy": 0.73,
  "seed": 42,
  "results": [
    {
      "filename": "image1.jpg",
      "gt_dl_name": "타이레놀정500mg",
      "num_detections": 1,
      "detections": [
        {
          "bbox": [x1, y1, x2, y2],
          "top3_class_name": ["타이레놀정500mg", "...", "..."],
          "top3_confidence": [0.89, 0.06, 0.03]
        }
      ]
    }
  ]
}
```

---

## 7. 금지 사항 (필수 준수)

- ❌ **epochs, imgsz, seed 임의 변경** (batch만 GPU 메모리 맞춰 조정 가능)
- ❌ **증강 Top-3 외 임의 추가** (CLAHE / GaussianBlur / MotionBlur만)
- ❌ **Top-3 파라미터 임의 변경** (3-1 표의 값 그대로)
- ❌ **평가셋 이미지를 학습에 포함**
- ❌ **평가 로직 임의 변경** (이름 기반 채점 필수)

---

## 8. 체크리스트

- [ ] epochs=15 / imgsz=640 / seed=42 설정 확인
- [ ] 증강 Top-3 (CLAHE + GaussianBlur + MotionBlur) 파라미터 정확히 적용
- [ ] bbox(또는 mask) 함께 변환되는 구현인지 확인
- [ ] 평가 스크립트 준비 (본인 프레임워크용)
- [ ] JSON 형식 출력 준비
- [ ] 학습 커브 스크린샷 저장 준비

---

**본 설계서의 모든 조건은 공정 비교를 위한 최소 통일 조건입니다.**
