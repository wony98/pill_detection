# data/

학습/평가용 데이터 위치 (raw 자료는 깃에 박지 않습니다 — `.gitignore` 처리).

## 깃에 트래킹되는 내용

```
data/
└── synthetic/        합성 데이터 생성 노트북
    ├── synthetic_images_v2.ipynb          v2 합성 (격자/배경 단축경로 차단)
    ├── synthetic_images_v3.ipynb          v3 합성 (palm 마스크 + dark surface)
    ├── palm_mask_rembg_colab.ipynb        손바닥 마스크 추출 (rembg)
    └── generate_dark_surface.ipynb        어두운 배경 합성
```

## 깃에 박지 않는 (gitignored) 내용

- `train_images/`, `train_annotations/`, `test_images/` — Kaggle/AI Hub 원본
- `preprocessing/`, `team_preprocessed/` — 공통 전처리 산출물
- `synthetic/`, `synthetic_combined/`, `coco/`, `dataset_yolo/`, `yolo/` — 학습용 가공본
- `merged_*.csv`, `final_*.csv` — 병합 메타데이터
- `pill.yaml` — 클래스 리스트 (실제 파일은 `demo/baseline/pill.yaml` 등에 사본 보관)

## 합성 데이터셋 (Hugging Face Hub)

학습에 쓴 합성 자료는 별도 공개:

- [`wony98/healtheat-pill-synthetic-baseline`](https://huggingface.co/datasets/wony98/healtheat-pill-synthetic-baseline)
- [`wony98/healtheat-pill-synthetic-v2`](https://huggingface.co/datasets/wony98/healtheat-pill-synthetic-v2)
- [`wony98/healtheat-pill-synthetic-v3`](https://huggingface.co/datasets/wony98/healtheat-pill-synthetic-v3)
