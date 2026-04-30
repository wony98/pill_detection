# Pill Detection (HealthEat)

Codeit AI 부트캠프 초급 프로젝트 — 3팀.
이미지에서 최대 4개 알약을 탐지하고 118종 클래스로 분류하는 object detection 모델.

---

## 현재 상태

**최종 모델: RTMDet-tiny** (seed=42 / seed=100 평균)

| 항목 | 값 |
|---|---|
| Top-1 정답률 | 31.9% |
| Top-3 정답률 | 42.6% |
| Detection rate | 88.3% |
| 파라미터 | 4.91M |
| CPU 추론 속도 | 1.03초 / 장 (HF Spaces 환경 근사) |

평가 기준: 33장 / 47 bbox, IoU 0.5 매칭, 박스별 Top-3 정답률.

---

## 폴더 구조

```
codeit10_pj1/
├── eda/         # 01~04b 데이터 분석 + 전처리 노트북
├── train/       # 모델별 학습 노트북 (faster_rcnn / rfdetr / rtmdet / yolov8n / ablation)
├── eval/        # 모델 비교 / 채점 / CPU 추론 측정 노트북
├── demo/        # 추론 데모 스크립트 (RTMDet-tiny)
├── results/     # 모델 결과 JSON / 보고서 / 백업 가중치
├── data/        # 데이터셋 (gitignore)
└── README.md
```

각 폴더 상세는 폴더 안 README 참고 (있는 경우).

---

## 학습 / 평가 흐름

1. **EDA + 전처리** — `eda/01_merge.ipynb` → `02_eda` → `03_preprocessing` → `04_split`
2. **학습** — `train/<모델>_train_*.ipynb`
3. **평가** — `eval/model_selection.ipynb` (개별 채점) → `eval/model_comparison.ipynb` (모델 비교 + 1위 결정)
4. **추론 데모** — `demo/inference_demo.py`

---

## 환경 (박제값)

```
Python 3.11
PyTorch 2.1.0 + cu118  (CPU 추론은 +cpu)
NumPy <2

# RTMDet 전용 (mmdet 스택)
mmengine 0.10.4   ★ 0.10.5+는 PyTorch 2.1 ASGD 충돌
mmcv 2.1.0
mmdet 3.3.0
```

상세 세팅은 `demo/README.md` 참고.

---

## 데이터

- 학습 타겟: 118종 (`pill.yaml` names 리스트가 진짜 출처 — CSV sorted 추측은 함정)
- 공통 전처리 결과: 39,864 ann / 10,392 img
- Split: Combo Group, val_ratio=0.2, seed=42 (train 8,302 / val 2,090)
- 평가셋: 실사용 이미지 33장 / 47 bbox

> Kaggle 제출 시 `category_id`는 학습용 class_idx (0~117)가 아니라 **원본 dl_idx**.
