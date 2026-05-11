# Pill Detection (HealthEat)

Codeit AI 부트캠프 초급 프로젝트 — **3팀 HealthEat**.
이미지에서 최대 4개 알약을 탐지하고 118종 클래스로 분류하는 object detection 모델.

---

## 결과 비교 (4-way)

| 측정 (33장 / 47 GT) | **baseline (배포)** | v2 | v3 | **v4 (분석 우위)** |
|---|---|---|---|---|
| Top-1 정답률 | 29.8% | 31.9% | 29.8% | **46.8%** ★ |
| Top-3 정답률 | 40.4% | 53.2% | 55.3% | **59.6%** |
| Detection rate | 95.7% | 95.7% | 93.6% | **95.7%** |
| False Positive (conf 0.05) | 259 | 100 | 99 | **84** |
| False Positive (conf 0.25) | - | - | - | **10** |
| 모델 파라미터 | 4.9M | 4.9M | 4.9M | **4.9M** |
| GPU 추론 (ms/img) | 125.5 | 128.6 | 132.1 | **132.1** |
| CPU 추론 (s/img) | ~1.0 | ~1.0 | ~1.0 | **~1.0** (HF Spaces 검증) |

**평가 기준**: 실사용자 알약 사진 33장 / 47 GT bbox, IoU 0.5 매칭, 박스별 Top-3 그룹핑.
**일반화 목표**: 임의 사용자 스마트폰 사진. Kaggle 점수는 보조 지표 (도메인 동일).

### 환경별 정답률 (v4)

| 환경 | n | Top-1 |
|---|---|---|
| palm (손바닥) | 20 | 50.0% |
| dark (어두운 배경) | 2 | 50.0% |
| paper (종이) | 7 | 42.9% |
| desk (책상) | 5 | 40.0% |
| cloth (천) | 10 | 40.0% |
| bottle (약통) | 2 | 100.0% |
| box (약상자) | 1 | 0.0% |

## Quick Start

```bash
# 1. 클론
git clone https://github.com/wony98/pill_detection.git
cd pill_detection

# 2. RTMDet 환경 (venv 권장)
pip install torch==2.1.0 torchvision==0.16.0 "numpy<2"
pip install -U openmim
mim install "mmengine==0.10.4"   # 0.10.5+ ASGD 충돌
mim install "mmcv==2.1.0"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && pip install -e . && cd ..

# 3. 가중치 다운로드 (Hugging Face Hub — "산출물" 섹션 참조)
huggingface-cli download wony98/healtheat-pill-rtmdet-baseline --local-dir demo/baseline/
#    v4 (선택)
huggingface-cli download wony98/healtheat-pill-rtmdet-v4 --local-dir demo/v4/

# 4. 추론 (default = baseline)
python inference.py --img sample.png

# 5. v4 모델 사용 (분석 단계 우위 검증용)
python inference.py --img sample.png --model v4

# 6. 옵션
python inference.py --img sample.png --device cuda:0 --out result.json
python inference.py --help
```

**모델 선택**:
- `baseline` (default) — 현재 배포 모델, 웹 데모 후처리 호환
- `v4` — 분석 단계 우위 (Top-1 46.8%), 후처리 호환 작업 시 교체 가능

---

## 폴더 구조

```
pill_detection/
├── docs/                    발표자료 / 보고서 / 협업일지
├── data/
│   ├── preprocessing/       팀 공통 전처리
│   └── synthetic/           합성 데이터 생성 (v2/v3, palm 마스크, dark surface)
├── eda/                     데이터 분석 + 클래스 매핑 (01~04b)
├── training/
│   ├── notebooks/           학습 노트북 (RTMDet/YOLOv8/Faster R-CNN/RF-DETR/ablation)
│   └── configs/             baseline / v4 학습 config
├── analysis/                평가 분석 (4-way 비교 + 진단 노트북 10+개)
├── eval/                    평가셋 33장 + 결과 JSON + Kaggle 제출
├── demo/
│   ├── baseline/            현재 배포 모델 (웹 데모 후처리 호환)
│   ├── v4/                  분석 단계 우위 모델 (후처리 호환 작업 시 교체 가능)
│   └── api_matching.ipynb   식약처 API 매칭 (baseline/v4 공통)
├── results/                 모델 결과 JSON / 보고서 산출물
├── requirements-*.txt       환경별 패키지 (eda/train/rtmdet/all)
└── README.md
```

각 폴더의 상세는 폴더 내 README.md 참조.

---

## 환경 세팅 상세

추론만 할 경우 위의 **Quick Start** 섹션으로 충분합니다. 학습/EDA/실험까지 직접 돌려보려면 환경별로 분리된 requirements를 사용합니다.

```bash
# EDA + 일반 학습 (YOLOv8 / RF-DETR / Faster R-CNN)
pip install -r requirements-all.txt
```

환경별 패키지 분리 의도:
- `requirements-eda.txt` — 데이터 분석 / 노트북 실행
- `requirements-train.txt` — YOLOv8 / RF-DETR / Faster R-CNN 학습
- `requirements-rtmdet.txt` — RTMDet 학습 + 추론 (mmdet 스택, Quick Start에 동일 명령)
- `requirements-all.txt` — eda + train 통합 (RTMDet은 별도)

**RTMDet venv 분리 권장 이유**: mmdet 스택은 PyTorch 2.1.0 + numpy<2 + mmengine 0.10.4 핀이 필요하고, mim 경유 설치라 일반 학습 환경과 분리하면 충돌이 줄어듭니다.

---

## 학습 / 평가 흐름

1. **EDA + 전처리** — `eda/01_merge.ipynb` → `02_eda` → `03_preprocessing` → `04_split`
2. **합성 데이터 생성** — `data/synthetic/synthetic_images_v2.ipynb` / `v3.ipynb` / `palm_mask_rembg_colab.ipynb`
3. **학습** — `training/notebooks/rtmdet_train_seed100.ipynb` (메인)
4. **평가** — `eval/` 안 평가 노트북 + `analysis/03_v4_evaluation_analysis.ipynb`
5. **추론** — `python inference.py --img <이미지경로>` (루트 진입점, default = baseline)

---

## 데이터

- 학습 클래스: 118종 (`pill.yaml` — YOLO/RTMDet 학습용 클래스 이름 리스트, 클래스 ID 진짜 출처)
- 공통 전처리 결과: 39,864 ann / 10,392 img
- Split: **Combo Group** (같은 조합 사진은 train/val 한쪽에만 들어가는 분할 방식), val_ratio=0.2, seed=42 (train 8,302 / val 2,090)
- 합성 데이터 v3: 7,994장 (palm 마스크 + dark surface + **격자 단축경로 차단** — 학습 데이터의 4모서리 격자 배치를 모델이 외워버리는 단축 학습을 막기 위한 설계)
- 평가셋: 실사용 이미지 33장 / 47 GT bbox

**Kaggle 제출 주의**: `category_id = dl_idx` (250부터 시작) — 학습용 0~117 인덱스 박으면 점수 0점.
> `dl_idx`는 Kaggle 메타데이터의 클래스 인덱스로, 학습 코드의 0~117 인덱스와 다릅니다.

---

## v4 처방 (최종 모델)

학습 사양 변경 7개 (vs v3):
1. imgsz 1280 → **1536** (각인 보존)
2. RandomCrop crop_size = (1280, 1280)
3. EMA momentum 0.0002 → **0.0001** (일반화 강화)
4. CachedMixUp prob 0.5 → **0.7**
5. Albu 추가 (CLAHE / RandomBrightnessContrast / GaussNoise / MotionBlur, 각 p=0.3)
6. batch 16 (A100 fallback)
7. lr 0.00025

유지: max_epochs 13 / num_last_epochs 3 / switch_epoch 10 / HSV value_delta 60 / seed 100

**conf threshold sweet spot**: 0.25 (Top-1 손해 0, FP 84 → 10, 88% 감소)

---

## 산출물 (Hugging Face Hub)

가중치 / 합성 데이터는 깃에 박지 않고 Hugging Face Hub에 별도 공개해 두었습니다.

| 항목 | Hugging Face 레포 |
|---|---|
| baseline 모델 (배포) | [`wony98/healtheat-pill-rtmdet-baseline`](https://huggingface.co/wony98/healtheat-pill-rtmdet-baseline) |
| v4 모델 (분석 우위) | [`wony98/healtheat-pill-rtmdet-v4`](https://huggingface.co/wony98/healtheat-pill-rtmdet-v4) |
| 합성 데이터 baseline | [`wony98/healtheat-pill-synthetic-baseline`](https://huggingface.co/datasets/wony98/healtheat-pill-synthetic-baseline) |
| 합성 데이터 v2 | [`wony98/healtheat-pill-synthetic-v2`](https://huggingface.co/datasets/wony98/healtheat-pill-synthetic-v2) |
| 합성 데이터 v3 | [`wony98/healtheat-pill-synthetic-v3`](https://huggingface.co/datasets/wony98/healtheat-pill-synthetic-v3) |

```bash
# 모델 가중치 다운로드 예시
huggingface-cli download wony98/healtheat-pill-rtmdet-baseline
huggingface-cli download wony98/healtheat-pill-rtmdet-v4
```

---

## 발표 자료 / 보고서

- [`docs/healtheat_presentation.pptx`](docs/healtheat_presentation.pptx) — 발표 자료 (PPT, 30 MB)
- [`docs/보고서.md`](docs/보고서.md) — 본문 (마크다운)
- [`docs/보고서.pdf`](docs/보고서.pdf) — PDF 출력본
- [`docs/협업일지/`](docs/협업일지/) — 작업 진행 일지 12편 (2026-04-20 ~ 2026-05-08)
