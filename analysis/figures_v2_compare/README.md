# analysis/figures_v2_compare/

`v2_detail_analysis.ipynb` 산출물: baseline vs v2 모델의 33장 평가 결과 측면별 비교.

| 파일 | 내용 |
|---|---|
| `01_grid_baseline.png` / `02_grid_v2.png` | 33장 그리드 시각화 (baseline vs v2) |
| `03_regression_cases.png` | v2에서 회귀(악화)된 케이스 |
| `04_conf_distribution.png` | confidence 분포 변화 |
| `05_env_change.png` | 환경별(palm/dark/paper/desk/cloth/bottle/box) 정답률 변화 |
| `06_count_group.png` | 알약 개수별 정답률 변화 |
| `07_pin_class_pool.png` | pin 클래스(특정 외형) 분리 |
| `08_shape_distribution.png` | 형태별 분포 |

관찰: v2는 격자/배경 단축경로 차단 효과로 Top-3 정답률이 +12.8pp 올랐지만 Top-1은 거의 동률이다. v3/v4가 필요한 근거다.
