# CPU 추론 속도 측정 (v3.7 6단계)

측정: 2026-04-29 14:45
환경: Colab CPU runtime (2 vCPU, HF Spaces 무료 환경 근사)
모델: RTMDet-tiny (seed=100, best_coco_bbox_mAP_epoch_15.pth)
imgsz: 640

## 결과

| 지표 | 값 |
|---|---|
| 측정 횟수 | 33장 × 3 round = 99회 |
| 평균 | 1031ms (1.03초) |
| 표준편차 | 289ms |
| min / median / max | 760 / 941 / 2784ms |

## 판정 (v3.7 룰)

- 2초 이내 → 직접 배포 가능
- 2~5초 → 데모 가능, 답답함, 최적화 권장
- 5초 초과 → 부적합

**판정: 직접 배포 가능**
