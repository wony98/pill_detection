# analysis/figures_v4_eval/

`03_v4_evaluation_analysis.ipynb` 산출물: v4 최종 모델 4-way 비교 (baseline/v2/v3/v4).

| 파일 | 내용 |
|---|---|
| `4way_quantitative_comparison.png` | 4모델 정량 비교 (Top-1/Top-3/Detection/FP) |
| `4way_qualitative_낙소졸정_500_20mg.png` | 정성 비교: 낙소졸정 |
| `4way_qualitative_놀텍정_10mg.png` | 정성 비교: 놀텍정 |
| `4way_qualitative_비모보정_500_20mg.png` | 정성 비교: 비모보정 |
| `conf_threshold_sweep_3way.png` | confidence threshold 스윕 (v2/v3/v4), v4의 최적 임계값 0.25 |
| `ensemble_4way_comparison.png` | 4모델 ensemble 비교 |

결과: v4 Top-1 46.8% (baseline +17.0pp), conf threshold 0.25에서 FP 88% 감소.
