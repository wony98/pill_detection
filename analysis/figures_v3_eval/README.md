# analysis/figures_v3_eval/

`v3_evaluation_analysis.ipynb` 산출물 — v3(palm 마스크 + dark surface) 모델 평가 + 3-way(baseline/v2/v3) 비교.

| 파일 | 내용 |
|---|---|
| `3way_comparison.png` | baseline/v2/v3 33장 정답률 종합 비교 |
| `conf_threshold_sweep.png` | confidence threshold 스윕 (FP 감소 vs Top-1 손실) |
| `ensemble_comparison.png` | 단일 vs ensemble(IoU 매칭 + conf 평균) 비교 |
| `hallucination_change.png` | hallucination(없는 알약 탐지) 변화 |
| `palm_change.png` | palm(손바닥) 환경 정답률 변화 |
| `paper_change.png` / `paper_regression.png` | paper(종이) 환경 정답률 + 회귀 케이스 |
| `pin_class_change.png` | pin 클래스 변화 |

핵심 관찰: v3는 palm 환경 개선했으나 paper에서 일부 회귀 — v4의 imgsz 1536 + Albu 도입 동기.
