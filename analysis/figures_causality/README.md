# analysis/figures_causality/

`data_model_causality.ipynb` 산출물: imgsz(입력 해상도)가 baseline → v4 정답률 격차의 주 원인인지 검증한 인과성 분석.

| 파일 | 내용 |
|---|---|
| `10_pair1_original.png` / `10_pair1_imgsz_compare.png` | 페어 1: 원본 vs imgsz 변경 시 결과 비교 |
| `10_pair2_*` / `10_pair3_*` | 페어 2, 3 (서로 다른 K-code 약품 케이스) |
| `10_summary.png` | 3개 페어 종합 요약 |

결론: imgsz 1280 → 1536 변경이 각인(imprint) 보존을 통해 정답률에 직접 기여한다는 정량 근거 확보.
