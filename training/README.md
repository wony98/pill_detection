# Training

학습 노트북 + 학습 config 박혀있는 폴더입니다.

## 구조

```
training/
├── notebooks/              학습 노트북 (Colab 실행 기준)
└── configs/                mmdet 학습 config (.py)
```

## 학습 환경

- Colab Pro A100 (40GB) 권장
- PyTorch 2.1.0 + cu118
- mmengine 0.10.4 (★ 0.10.5+ ASGD KeyError 박힘)
- mmcv 2.1.0 / mmdet 3.3.0
- numpy<2 (1.26.4) / albumentations 2.0.8 (`--no-deps` 설치)

## 모델 학습 이력 (RTMDet-tiny)

| 버전 | best epoch | 33장 Top-1 | 주요 처방 | 학습 산출물 (Hugging Face Hub) |
|---|---|---|---|---|
| baseline | 15 (best e13) | 29.8% | 합성 통합본 학습 | `wony98/healtheat-pill-rtmdet-baseline` |
| v2 | 14 | 31.9% | 합성 v2 (격자/배경 단축경로 차단) | (학습 백업, HF 미공개) |
| v3 | 11 | 29.8% | 합성 v3 (palm 마스크) | (학습 백업, HF 미공개) |
| **v4** | **12** | **46.8%** ★ | **imgsz 1536 + RandomCrop + Albu 4종 + EMA 0.0001** | `wony98/healtheat-pill-rtmdet-v4` |

> v4 결과 JSON 파일명에 박힌 `e13`은 라벨 표기 오류 — 실제 best epoch = 12.

## v4 처방 (최종 모델)

학습 사양 변경 7개 (vs v3):

1. imgsz 1280 → **1536** (각인 보존)
2. RandomCrop crop_size = imgsz - 256 = (1280, 1280)
3. EMA momentum 0.0002 → **0.0001** (일반화 강화)
4. CachedMixUp prob 0.5 → **0.7**
5. Albu 추가: CLAHE / RandomBrightnessContrast / GaussNoise / MotionBlur (각 p=0.3)
6. batch 16 (A100 fallback)
7. lr 0.00025 (batch 비례)

유지: max_epochs 13 / num_last_epochs 3 / switch_epoch 10 / HSV value_delta 60 / seed 100 / num_classes 118

## 노트북

| 파일 | 모델 | 비고 |
|---|---|---|
| `rtmdet_train_seed100.ipynb` | RTMDet-tiny seed=100 | 메인 학습 |
| `rtmdet_train_seed42.ipynb` | RTMDet-tiny seed=42 | 재현성 검증 |
| `yolov8n_no_aug.ipynb` | YOLOv8n | 모델 비교 |
| `faster_rcnn_train_eval.ipynb` | Faster R-CNN | 모델 비교 |
| `rfdetr_train_eval.ipynb` | RF-DETR | 모델 비교 |
| `ablation_11_experiments.ipynb` | RTMDet 증강 ablation | 11개 실험 |

## 가중치 다운로드

가중치는 깃에 박지 않습니다 (24~44 MB). Hugging Face Hub 사용:

```bash
huggingface-cli download wony98/healtheat-pill-rtmdet-baseline  # baseline 43 MB
huggingface-cli download wony98/healtheat-pill-rtmdet-v4         # v4 44 MB
```

## 재현 절차 (v4 기준)

1. `huggingface-cli download wony98/healtheat-pill-rtmdet-v4` 로 v4 산출물 다운로드
2. config = `configs/v4_rtmdet_tiny_pill.py` (imgsz 1536, Albu, EMA 0.0001 박힘)
3. 데이터 = `dataset_yolo.zip` (팀 내부 공유)
4. Colab A100 + 위 환경 박힌 그대로
5. ETA ~3시간 30분 (epoch 13)
