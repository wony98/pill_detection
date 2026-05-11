# analysis/data/

분석 노트북이 반복 호출하는 예측 캐시 / 정답 메타데이터.

| 파일 | 내용 |
|---|---|
| `val_kcode_coco.json` | K-code val 분할 (2,090장) COCO 포맷 GT |
| `predictions_baseline_kcode_val_2090.json` | baseline 모델의 val 2,090장 예측 결과 |
| `predictions_v2_kcode_val_2090.json` | v2 모델의 val 2,090장 예측 결과 |

용도: `analysis/` 안 노트북들이 4-way 비교, ablation 효과 측정 시 이 캐시를 읽어 재계산을 줄입니다.
