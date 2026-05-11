# analysis/figures_generalization/

`generalization_analysis.ipynb` 산출물 — Phase A → B → C 단계별 일반화 진척 분석.

| 파일 | 내용 |
|---|---|
| `08_phase_a23_summary.png` | Phase A2, A3 정답률 변화 요약 |
| `08_training_curves.png` / `08_training_curves_v2.png` | 학습 커브 (loss / mAP) |
| `08_map50_vs_map5095.png` | mAP@0.5 vs mAP@0.5:0.95 — Top-1 정답률과 박스 정확도 분리 시각화 |

관찰: 박스 정확도(mAP@0.5)는 일찍 포화되지만 클래스 정확도(Top-1)는 후반 epoch에서 계속 개선.
