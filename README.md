# Pill Detection (HealthEat)

Codeit AI 부트캠프 초급 프로젝트 — **3팀 HealthEat**.
이미지에서 최대 4개 알약을 탐지하고 118종 클래스로 분류하는 object detection 모델.

---

## 최종 결과 (RTMDet-tiny v4)

| 측정 (33장 / 47 GT) | baseline | v2 | v3 | **v4 (최종)** |
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
│   ├── baseline/            1차 deploy 정본 (지우님 웹 후처리 호환)
│   ├── v4/                  추후 적용 여지 (후처리 호환 후 교체)
│   └── api_matching.ipynb   식약처 API 매칭 (baseline/v4 공통)
├── results/                 모델 결과 JSON / 보고서 산출물
├── requirements-*.txt       환경별 패키지 (eda/train/rtmdet/all)
└── README.md
```

각 폴더 상세는 안 README 참조.

---

## 환경 세팅

```bash
# 1. 통합 환경 (EDA + 일반 학습)
pip install -r requirements-all.txt

# 2. RTMDet 전용 (별도 venv 권장)
pip install torch==2.1.0 torchvision==0.16.0 "numpy<2"
pip install -U openmim
mim install "mmengine==0.10.4"   # ★ 0.10.5+ ASGD 충돌
mim install "mmcv==2.1.0"
git clone https://github.com/open-mmlab/mmdetection.git && cd mmdetection && pip install -e .
```

환경별 패키지 분리:
- `requirements-eda.txt` — 데이터 분석 / 노트북 실행
- `requirements-train.txt` — YOLOv8 / RF-DETR / Faster R-CNN 학습
- `requirements-rtmdet.txt` — RTMDet 학습 + 추론 (mmdet 스택)
- `requirements-all.txt` — eda + train 통합

---

## 학습 / 평가 흐름

1. **EDA + 전처리** — `eda/01_merge.ipynb` → `02_eda` → `03_preprocessing` → `04_split`
2. **합성 데이터 생성** — `data/synthetic/synthetic_images_v2.ipynb` / `v3.ipynb` / `palm_mask_rembg_colab.ipynb`
3. **학습** — `training/notebooks/rtmdet_train_seed100.ipynb` (메인)
4. **평가** — `eval/` 안 평가 노트북 + `analysis/03_v4_evaluation_analysis.ipynb`
5. **추론 데모** — `demo/baseline/inference_demo.py` (1차 deploy 정본)

---

## 데이터

- 학습 클래스: 118종 (`pill.yaml` names 리스트 = 클래스 ID 진짜 출처)
- 공통 전처리 결과: 39,864 ann / 10,392 img
- Split: Combo Group, val_ratio=0.2, seed=42 (train 8,302 / val 2,090)
- 합성 데이터 v3: 7,994장 (palm 마스크 + dark surface + 격자 단축경로 차단)
- 평가셋: 실사용 이미지 33장 / 47 GT bbox

**Kaggle 제출 주의**: `category_id = dl_idx` (250부터 시작) — 학습용 0~117 인덱스 박으면 점수 0점.

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

## 산출물 (Drive 백업)

가중치 / 학습 로그는 깃에 박지 않습니다. Drive 사용:

| 항목 | Drive 위치 | 사이즈 |
|---|---|---|
| baseline 학습 백업 | `pill_detection_3팀/seungwon/seungwon_work_dir.zip` | 206 MB |
| v2 학습 백업 | `pill_detection_3팀/seungwon/seungwon_v6_work_dir.zip` | 207 MB |
| v3 학습 백업 | `pill_detection_3팀/seungwon/seungwon_v3_work_dir.zip` | 281 MB |
| v4 학습 백업 | `pill_detection_3팀/seungwon/seungwon_v4_work_dir.zip` | 281 MB |
| 데이터셋 | Drive ID `1xeE2mlCVcaxlhc98icP_Di7FKQfgAtlj` | 17.5 GB |

---

## 보고서 / 발표자료 / 협업일지

- 발표자료 PPT: `docs/HealthEat_3팀_발표자료.pptx` (TBD — 박힘 예정)
- 보고서 PDF: `docs/HealthEat_3팀_보고서.pdf` (TBD)
- 협업일지: `docs/협업일지/` (5/4 ~ 5/8, 일자별 .txt)
