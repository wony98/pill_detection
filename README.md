# Pill Detection (HealthEat)

Codeit AI 부트캠프 초급 프로젝트, 3팀 HealthEat.
이미지에서 최대 4개 알약을 탐지하고 118종 클래스로 분류하는 object detection 모델.

[MLflow로 기존 실험 기록 사후 정리](experiments/mlflow/README.md): 학습 로그와 평가 결과를 원본 파일 해시와 함께 정리하고, 평가와 체크포인트의 연결은 미확정으로 표시합니다.

정답 라벨 오기 1건(두 박스)을 고친 GT v2로 과거 v4 저장 예측을 재채점해 **24/47(51.1%)**로 정정했으며, 저장 가중치의 별도 실행에서는 **25/47**을 확인했습니다. GT v2는 모델명이 아닌 정답 버전이며, [정오표](docs/errata_label_v2_20261024/README.md)와 [별도 실행 근거](experiments/forward_eval_20261025/README.md)에서 각 결과를 확인할 수 있습니다.

---

## 결과 비교 (4-way, GT v2 재채점)

| 측정 (33장 / 47 GT) | baseline (당시 배포) | v2 | v3 | v4 (평가 JSON e13) |
|---|---|---|---|---|
| Top-1 정답률 | 29.8% (14/47) | 36.2% (17/47) | 29.8% (14/47) | 51.1% (24/47) |
| Top-3 정답률 | 40.4% (19/47) | 57.4% (27/47) | 59.6% (28/47) | 63.8% (30/47) |
| Detection rate | 95.7% | 95.7% | 93.6% | 95.7% |
| False Positive (conf 0.05) | 259 | 100 | 99 | 84 |
| False Positive (저장 점수 0.25 사후 필터) | 26 | 19 | 16 | 10 |
| 모델 파라미터 | 4.9M | 4.9M | 4.9M | 4.9M |
| GPU 추론 (ms/img, 원본 JSON 기록) | 125.5 | 128.6 | 132.1 | 132.1 |
| CPU 추론 (s/img, 당시 HF Spaces 기록) | ~1.0 | ~1.0 | ~1.0 | ~1.0 |

**평가 기준**: 사진33장·정답 박스47개, IoU 0.5 매칭. 네 모델의 저장 예측을 같은 GT v2로 재채점했으며 원본 예측은 바꾸지 않았습니다. 이 평가셋은 모델·점수 기준 선택에도 사용했습니다.

별도 새 실행은 원시 예측에 점수 기준0.25를 적용했을 때 Top-1 25/47, Top-3 28/47, 미매칭 박스10개였습니다. 과거 예측의 사후 필터와 처리 조건이 달라 위 표의 모델 성능 개선으로 해석하지 않습니다. 새 실행의 입력·설정·가중치·결과는 별도 MLflow 기록에 연결했습니다.

### 환경별 정답률 (v4, 당시 GT v1 분석)

아래 환경별 집계는 당시 기록을 보존한 값입니다. 이번 GT v2 재채점의 환경별 결과로 바꾸어 표시하지 않았습니다.

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
- `baseline` (default): 당시 웹 데모에 사용한 모델, 웹 데모 후처리 호환
- `v4`: 저장 예측의 GT v2 재채점 Top-1 24/47(51.1%). 웹 후처리 호환 작업은 별도

---

## 폴더 구조

```
pill_detection/
├── docs/                    보고서 / 협업일지 / 모델 선정 / 합성 진화 / 진단 자료
├── data/synthetic/          합성 데이터 생성 노트북 (v2/v3, palm 마스크, dark surface)
├── eda/                     데이터 분석 + 클래스 매핑 (01~04b)
├── training/
│   ├── notebooks/           학습 노트북 (RTMDet/YOLOv8/Faster R-CNN/RF-DETR/ablation)
│   └── configs/             baseline / v4 학습 config
├── analysis/                평가 분석 (4-way 비교 + 진단 노트북 10+개)
├── eval/                    평가셋 33장 + Kaggle 제출 노트북
├── demo/
│   ├── baseline/            현재 배포 모델 (웹 데모 후처리 호환)
│   ├── v4/                  분석 단계 우위 모델 (후처리 호환 작업 시 교체 가능)
│   └── api_matching.ipynb   식약처 API 매칭 (baseline/v4 공통)
├── results/reports/         보고서용 표 / 그림 / CSV
├── assets/                  README / 보고서용 그림
├── ppt_figures/             PPT 슬라이드 그림 산출물
├── inference.py             추론 진입점
├── requirements.txt         통합 패키지 목록 (RTMDet은 별도 venv 안내)
├── LICENSE
└── README.md
```

각 폴더의 상세는 폴더 내 README.md 참조.

---

## 환경 세팅 상세

추론만 할 경우 위의 Quick Start 섹션으로 충분합니다. 학습/EDA/실험까지 직접 돌려보려면 통합 requirements를 사용합니다.

```bash
# EDA + 일반 학습 (YOLOv8 / RF-DETR / Faster R-CNN)
pip install -r requirements.txt
```

**RTMDet venv 분리 권장**: mmdet 스택은 PyTorch 2.1.0 + numpy<2 + mmengine 0.10.4 핀이 필요하고 mim 경유 설치라 별도 venv에서 진행해주세요 (`requirements.txt` 상단 주석에 설치 흐름 정리되어 있습니다).

---

## 학습 / 평가 흐름

1. **EDA + 전처리**: `eda/01_merge.ipynb` → `02_eda` → `03_preprocessing` → `04_split`
2. **합성 데이터 생성**: `data/synthetic/synthetic_images_v2.ipynb` / `v3.ipynb` / `palm_mask_rembg_colab.ipynb`
3. **학습**: `training/notebooks/rtmdet_train_seed100.ipynb` (메인)
4. **평가**: `eval/` 안 평가 노트북 + `analysis/03_v4_evaluation_analysis.ipynb`
5. **추론**: `python inference.py --img <이미지경로>` (루트 진입점, default = baseline)

---

## 데이터

- 학습 클래스: 118종 (`pill.yaml`: YOLO/RTMDet 학습용 클래스 이름 목록, 클래스 ID 기준 출처)
- 공통 전처리 결과: 39,864 ann / 10,392 img
- Split: Combo Group (같은 조합 사진은 train/val 한쪽에만 들어가는 분할 방식), val_ratio=0.2, seed=42 (train 8,302 / val 2,090)
- 합성 데이터 v3: 7,994장 (palm 마스크, dark surface, 격자 단축경로 차단. 학습 데이터의 4모서리 격자 배치를 모델이 외우는 단축 학습을 막기 위한 설계)
- 평가셋: 실사용 이미지 33장 / 47 GT bbox

**Kaggle 제출 주의**: `category_id = dl_idx` (250부터 시작). 학습용 0~117 인덱스를 넣으면 점수 0점.
> `dl_idx`는 Kaggle 메타데이터의 클래스 인덱스로, 학습 코드의 0~117 인덱스와 다릅니다.

---

## v4 학습 설정 변경 (최종 모델)

학습 사양 변경 7개 (vs v3):
1. imgsz 1280 → 1536 (각인 보존)
2. RandomCrop crop_size = (1280, 1280)
3. EMA momentum 0.0002 → 0.0001 (일반화 강화)
4. CachedMixUp prob 0.5 → 0.7
5. Albu 추가 (CLAHE / RandomBrightnessContrast / GaussNoise / MotionBlur, 각 p=0.3)
6. batch 16 (A100 fallback)
7. lr 0.00025

유지: max_epochs 13 / num_last_epochs 3 / switch_epoch 10 / HSV value_delta 60 / seed 100

**점수 기준0.25**: 과거 저장 예측의 사후 필터에서는 GT v2 Top-1 24/47·Top-3 30/47을 유지하며 미매칭 박스가84개에서10개로 줄었습니다. 새 원시 예측의 후처리에서는 Top-1 25/47은 유지됐지만 Top-3는29/47에서28/47로 달라졌습니다. 새로운 사진에서도 같은 결과가 유지되는지는 별도 평가가 필요합니다.

---

## 산출물 (Hugging Face Hub)

가중치와 합성 데이터는 용량이 커서 깃 저장소에 포함하지 않고 Hugging Face Hub에 별도 공개해 두었습니다.

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

- [`docs/healtheat_presentation.pptx`](docs/healtheat_presentation.pptx): 발표 자료 (PPT, 30 MB)
- [`docs/report.md`](docs/report.md): 본문 (마크다운)
- [`docs/journal/`](docs/journal/): 작업 진행 일지 12편 (2026-04-20 ~ 2026-05-08)

---

## License

본 레포지토리는 [MIT License](LICENSE)를 따릅니다.
