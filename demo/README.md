# Demo

웹 데모 및 HF Spaces 등 배포용 데모 패키지를 모아둔 폴더입니다.

## 구조

```
demo/
├── baseline/             ← 1차 deploy 정본 (웹 후처리 호환)
├── v4/                   ← 추후 적용 여지 (후처리 호환 후 교체)
└── api_matching.ipynb    ← 식약처 API 매칭 (118종, baseline/v4 공통)
```

## 모델 선택 가이드

| 항목 | baseline | v4 |
|---|---|---|
| 33장 Top-1 | 29.8% | **46.8%** |
| 33장 Top-3 | 40.4% | **59.6%** |
| Detection | 95.7% | 95.7% |
| 가중치 사이즈 | 43 MB | 44 MB |
| imgsz | 1280 | 1536 |
| 추론 시간 (CPU) | ~1초 | ~1초 |
| 후처리 호환 | ✓ (현 웹 데모 호환) | ✗ (추후 작업) |
| 1차 deploy | ★ | - |

**1차 deploy = baseline** — 팀에서 운영 중인 웹 데모의 후처리 코드가 baseline 사양에 맞춰져 있어 즉시 호환됩니다.
**v4는 후처리 호환 작업 완료 후 교체 예정** (성능 +17pp, conf 0.25 권장, bimodal 분포 대응 필요).

## 가중치 다운로드

가중치는 깃에 포함되지 않습니다 (24~44 MB). Hugging Face Hub 권장:

- baseline: `huggingface-cli download wony98/healtheat-pill-rtmdet-baseline` → `best_coco_bbox_mAP_epoch_15.pth` (43 MB)
- v4: `huggingface-cli download wony98/healtheat-pill-rtmdet-v4` → `best_coco_bbox_mAP_epoch_12.pth` (44 MB)

## 사용법

각 모델 폴더 안 README 참조:
- `baseline/README.md` — baseline deploy 환경 + 사용법
- `v4/README.md` — v4 사양 + 후처리 호환 주의사항

## 식약처 API 매칭 (`api_matching.ipynb`)

118종 알약에 대한 효능/용법/주의사항 매핑 노트북 (data.go.kr 공개 API).
공개 API 키 필요 (환경변수 `DATA_GO_KR_KEY`).
출력: `pill_info_master.json` (118/118 클래스 100% 매핑 박힘).
위치: `demo/pill_info_master.json` (118종 효능/용법/주의 + 식약처 image_url).
