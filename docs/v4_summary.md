# v4 학습 결과 분석 요약

## GT v2 재채점과 별도 실행 확인

정답 라벨 오기1건(두 박스)을 수정한 GT v2로 저장 예측을 재채점했습니다. 과거 v4 예측은 Top-1 22→**24/47(51.1%)**, Top-3 28→**30/47(63.8%)**로 정정됐습니다. 같은 가중치로 새로 만든 결과라는 뜻은 아닙니다([정오표](errata_label_v2_20261024/README.md)).

저장 가중치의 별도 실행에서는 Top-1 **25/47**을 확인했습니다. 점수기준0.25에서 Top-3는28/47, 미매칭 박스는10개입니다. 원시 출력·설정·가중치·GT·환경과 결과를 연결한 [실행 근거](../experiments/forward_eval_20261025/README.md)를 별도로 남겼습니다. 과거 평가 JSON과 이 체크포인트의 연결은 여전히 미확정입니다.

당시 선택은 v4 단독과 점수기준0.25였습니다. 아래 환경별·앙상블·전이 분석 및 그림은 GT v1 당시 기록으로 보존하고, 대표 비교표만 GT v2로 갱신했습니다.

---

## 메인 비교표 (47 GT, GT v2 재채점)

| 저장 예측 | Top-1 | Top-3 | Detection | FP (0.05) | JSON best_epoch |
|---|---|---|---|---|---|
| baseline (e15) | 29.8% (14/47) | 40.4% (19/47) | 95.7% | 259 | 15 |
| v2 (e14) | 36.2% (17/47) | 57.4% (27/47) | 95.7% | 100 | 14 |
| v3 (e11) | 29.8% (14/47) | 59.6% (28/47) | 93.6% | 99 | 11 |
| v4 (평가 JSON e13) | 51.1% (24/47) | 63.8% (30/47) | 95.7% | 84 | 13 |

v4는 같은 GT v2에서 baseline보다10개(+21.3pp), v2보다7개(+14.9pp)를 더 맞혔습니다. 가중치의 epoch12와 원본 평가 JSON의 e13은 구분합니다. 여러 학습 조건을 함께 바꾼 비교이므로 한 가지 변경의 효과로 단정하지 않습니다.

### Ensemble 비교 (conf 0.2, 당시 GT v1 분석)

| 구성 | Top-1 | Top-3 | FP |
|---|---|---|---|
| v4 단독 | 46.8% | 59.6% | 14 |
| v3+v4 | 44.7% | 61.7% | 24 |
| v2+v4 | 44.7% | 57.4% | 28 |
| v2+v3+v4 | 42.6% | 57.4% | 35 |
| v2+v3 (기존 최고) | 36.2% | 55.3% | 33 |

ensemble 방식: IoU 매칭 후 conf 평균 (방식 A). 모든 모델의 prediction을 IoU 0.5 기준으로 매칭, 매칭된 그룹은 bbox 평균 + top3 클래스 conf 합산 후 재선정합니다.

---

## 환경별 효과 (Top-1, 당시 GT v1 분석)

| 환경 | n | v2 | v3 | v4 | v4-v2 |
|---|---|---|---|---|---|
| palm | 20 | 30.0% | 35.0% | 50.0% | +20.0pp |
| dark | 2 | 0.0% | 0.0% | 50.0% | +50.0pp |
| paper | 7 | 28.6% | 14.3% | 42.9% | +14.3pp |
| desk | 5 | 20.0% | 20.0% | 40.0% | +20.0pp |
| cloth | 10 | 40.0% | 30.0% | 40.0% | +0.0pp |
| bottle | 2 | 100.0% | 100.0% | 100.0% | +0.0pp |
| box | 1 | 0.0% | 0.0% | 0.0% | +0.0pp |

palm/dark/paper 모든 가설 환경에서 목표치 달성. dark 환경은 합성 v3 단독으로 해소 못 한 영역인데 학습 사양 7가지 변경으로 +50pp 회복했습니다.

---

## conf threshold sweep (당시 GT v1 분석)

| conf | v4 Top-1 | v4 FP |
|---|---|---|
| 0.05 (default) | 46.8% | 84 |
| 0.10 | 46.8% | 29 |
| 0.20 | 46.8% | 14 |
| 0.25 | 46.8% | 10 |
| 0.30 | 44.7% | 7 |

위 표는 당시 점수 기준을 선택할 때 쓴 GT v1 결과입니다. GT v2로 재채점한0.05·0.25에서도 Top-1 24/47·Top-3 30/47은 같고 미매칭 박스는84→10입니다. 별도 새 실행에서는 두 기준의 Top-1 25/47은 같지만 Top-3가29→28/47로 바뀌었습니다.

시각화: `analysis/figures_v4_eval/conf_threshold_sweep_3way.png`

---

## v4 학습 사양 변경 7가지

| # | 항목 | v3 | v4 |
|---|---|---|---|
| 1 | batch size | 16 | 16 (A100 fallback) |
| 2 | imgsz | 1280 | 1536 |
| 3 | lr | 0.00025 | 0.00025 |
| 4 | EMA momentum | 0.0002 | 0.0001 |
| 5 | RandomCrop | 없음 | crop_size=1280 |
| 6 | CachedMixUp | p=0.5 | p=0.7 |
| 7 | Albumentations | 없음 | CLAHE / Brightness / Noise / MotionBlur (p=0.3) |

기타 통제: max_epochs=13, num_last=3, switch=10, HSV value_delta=60, seed=100, num_classes=118, 합성 v3 그대로.

---

## v2 → v4 verdict transition (당시 GT v1 분석)

47 GT 단위 회복/후퇴 분석:

- 회복 13건 / 후퇴 6건 / 순증 7건
- top1 유지 13건 / wrong→top1 3건 / top3→top1 6건
- det_fail 2건 변화 없음

특정 케이스의 회복이 아니라 전반적 분류 능력 향상으로 해석합니다.

---

## Hallucination 3-way 분포 (당시 GT v1 분석)

- v2 91회 → v3 분포 재배치만 (수치 비슷) → v4 71회 (-20)
- 합성 다양화(v3)는 핀 클래스 분포 재배치만 했고, augmentation 다양성(v4)은 hallucination 양 자체를 감소시켰습니다.

시각화: `analysis/figures_v4_eval/hallucination_3way.png`

---

## 4-way 통합 시각화 (당시 GT v1 분석)

기존 그림과 아래 설명은 GT v1 기준입니다. 정정된 대표 수치는 위 GT v2 메인 비교표를 참고하세요.

### 정량 비교 (`analysis/figures_v4_eval/4way_quantitative_comparison.png`)

Top-1 / Top-3 / Detection / FP 4지표 막대그래프 4 subplot:
- **Top-1**: v4 46.8%로 baseline 29.8%, v2 31.9%, v3 29.8% 대비 우위
- **Top-3**: v4 59.6% (baseline 40.4% → v2 53.2% → v3 55.3% → v4 59.6%로 점진적 증가)
- **Detection**: baseline / v2 / v4 모두 95.7%로 동일, v3만 93.6%로 소폭 하락
- **FP**: baseline 259 → v4 84로 -68% 감소

### 정성 비교 (`analysis/figures_v4_eval/4way_qualitative_*.png`)

v2 → v4 회복 케이스 10건 자동 추출 후 상위 3건 시각화:

| 케이스 | 변화 | figure |
|---|---|---|
| 낙소졸정 500/20mg (palm 환경) | v2 top3 → v4 top1, baseline 4박스 → v4 2박스로 FP 감소 + Top-1 정답 | `4way_qualitative_낙소졸정_500_20mg.png` |
| 놀텍정 10mg | v2 wrong → v4 top1 | `4way_qualitative_놀텍정_10mg.png` |
| 비모보정 500/20mg | v2 top3 → v4 top1 | `4way_qualitative_비모보정_500_20mg.png` |

색 코드: GT 초록 점선 / baseline 회색 / v2 파랑 / v3 보라 / v4 빨강 (conf ≥ 0.25 박스만 표시)

### Ensemble 비교 (`analysis/figures_v4_eval/ensemble_4way_comparison.png`)

v2+v3 / v2+v4 / v3+v4 / v2+v3+v4 ensemble 4-way 비교 figure. 모든 ensemble 구성이 v4 단독보다 Top-1에서 후퇴함을 시각화합니다.

---

## 학습 곡선 분석

val mAP@50:95는 epoch 11→12에서 +0.003으로 둔화, val mAP@50은 epoch 7부터 천장 도달했습니다. 학습이 거의 천장에 닿았으므로 max_epochs 추가 연장은 의미 제한적이라 판단했습니다.

---

## 당시 결정 근거 (GT v1, v4 단독 + conf 0.25 채택)

1. **Top-1 우위**: baseline 대비 +17.0pp / v2 단독 대비 +14.9pp / v2+v3 ensemble 대비 +10.6pp
2. **Ensemble 후퇴**: v3+v4, v2+v4 모두 -2.1pp / v2+v3+v4 -4.2pp, v4가 강해 약한 모델이 끌어내림
3. **Top-3 trade-off**: v3+v4 ensemble 61.7%로 +2.1pp 이득이지만 추론 비용 2배 → 단독 채택이 합리적
4. **conf 0.25 최적 임계값**: Top-1 손해 0, FP 14→10 추가 감소, Top-3 손해 0
5. **conf 0.30 손해 발생**: Top-1 -2.1pp 발생으로 0.25가 한계
6. **배포 비용 동일**: ms/image 132.1 (v3와 동일)
7. **환경별 가설 충족**: palm 50% / dark 50% / paper 42.9% (목표 40/20/30 모두 달성)
8. **안정적 개선**: v2 → v4 verdict 순증 7건 + hallucination -20건

---

## 추론 사양 (최종)

- 모델: RTMDet-tiny (4.9M 파라미터)
- 입력: 1536px (학습 시 사용한 값)
- conf threshold: 0.25
- Top-3 박스 그룹핑 IoU: 0.5
- ms/image: 132.1ms (GPU) / 1.03s (CPU, HF Spaces 무료 환경)

---

## 참조

| 자료 | 위치 |
|---|---|
| figure 7장 | `analysis/figures_v4_eval/` |
| v4 학습 사양 + 가중치 | HF Hub `wony98/healtheat-pill-rtmdet-v4` 모델 카드 |
| v4 데모 패키지 | `demo/v4/` (가중치 + config + inference) |
| 합성 진화 흐름 | `docs/synthesis_evolution/synthesis_v1_v2_v3.md` |
| 진단 / 개선안 | `docs/diagnosis_issues.md`, `docs/diagnosis_solutions.md` |
