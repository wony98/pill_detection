# RTMDet 알약 탐지 데모 패키지 (baseline)

3팀 HealthEat 프로젝트 — 웹 데모 1차 deploy용 baseline 모델 패키지.

## 모델 사양

- RTMDet-tiny (4.9M params)
- best epoch 15 (best 가중치 = epoch 13에서 박힘)
- imgsz 1280 / batch 16 / lr 0.00025
- 합성 통합본 학습 (synthetic v1, K-code 56종 + AI Hub 추가 학습)

## 33장 평가 성능 (47 GT 단위)

| 측정 | baseline (e15) |
|---|---|
| Top-1 | 29.8% |
| Top-3 | 40.4% |
| Detection | 95.7% |
| FP (conf 0.05) | 259 |
| ms/image | 125.5 |

## 패키지 구성

| 파일 | 설명 |
|---|---|
| `best_coco_bbox_mAP_epoch_15.pth` | baseline 가중치 (43 MB, Drive 백업, 깃 미포함) |
| `rtmdet_tiny_pill.py` | 학습 config (35 KB) |
| `pill.yaml` | 118 클래스 매핑 (★ 클래스 ID 진짜 출처) |
| `inference_demo.py` | Top-3 그룹핑 추론 스크립트 (CONF_THRESH=0.05) |
| `README.md` | 이 문서 |

## 가중치 다운로드

Hugging Face Hub 권장: `huggingface-cli download wony98/healtheat-pill-rtmdet-baseline`
- 정본: `best_coco_bbox_mAP_epoch_15.pth` (43 MB)
- 호환 사본: `baseline_v1_e15_best_e13.pth` (24 MB, EMA-only)

## 환경 세팅

```bash
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" -q
pip install "mmengine==0.10.4" "mmcv==2.1.0" "mmdet==3.3.0" -q
pip install pyyaml opencv-python pillow -q
```

## 사용법 (CLI)

```bash
python inference_demo.py --image path/to/image.jpg --config rtmdet_tiny_pill.py --weights best_coco_bbox_mAP_epoch_15.pth
```

## 추론 동작 방식

1. 이미지 로드 (한글 경로 안전 처리: `cv2.imdecode(np.fromfile(...))`)
2. RTMDet 추론 → bbox + score + label 출력
3. NMS (IoU 0.5)
4. CONF_THRESH 박힘 = 0.05 (baseline 권장값, 표시 박스 필터)
5. Top-3 그룹핑 (같은 bbox에 대한 상위 3개 클래스)

## 주의 (클래스 매핑)

**`pill.yaml`이 클래스 ID의 진짜 출처입니다.**
Kaggle 제출 시 `category_id = dl_idx` (250부터 시작) — 학습용 0~117 인덱스 박으면 점수 0점.

## 트러블슈팅

- **mmengine ASGD KeyError**: 0.10.5+ 버그. `pip install "mmengine==0.10.4"` 명시 박음 + 런타임 재시작
- **numpy ABI 충돌**: albumentations 등 설치 시 numpy 2.x 끌려옴 → `pip install "numpy<2" --no-deps` 우회
- **한글 경로 cv2.imread 실패**: `cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)` 패턴 사용
- **mmcv NMS sm_90 미지원**: H100에서 RuntimeError. mmcv 2.1.0 cu118 wheel은 sm_80까지만 빌드 → A100 사용 권장

## 참고

- v4 모델 (Top-1 46.8%)은 `demo/v4/`에 박혀있음. 후처리 호환 작업 후 deploy 예정.
- baseline 학습 진행: 본 레포 `training/notebooks/rtmdet_train_seed100.ipynb`
