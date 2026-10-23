# 모델선정기준 v3.7

> v3.6 → v3.7 갱신: 셀프체크 통과한 seed만 평균에 포함 룰 추가 + 베이스라인 역할 명확화 (자기 충족 예언 회피, 팀 협업 원칙)

---

## 평가 지표 정의

- **Top-1 정답률**: 모델이 1순위로 뽑은 답이 정답일 확률
- **Top-3 정답률**: 모델이 뽑은 1·2·3순위 중에 정답이 들어 있을 확률

평가셋: 33장 / GT: 47 bbox / 매칭 IoU: 0.5

---

## 1단계: 1차 필터 (자격)

- 라이선스 (Apache/MIT/BSD/AGPL 공개 OK)
- 모델 사이즈 ≤ 100MB
- PyTorch 호환
- 학습 imgsz=640 호환

---

## 2단계: 임계값 (합격선)

Top-3 정답률 ≥ 27.7% (yolov8n 베이스라인 conf 스캔 best값, seed=42 단독 기준)

미달 → 탈락

### 임계값 고정 원칙 (v3.7 추가)
- 임계값은 베이스라인 컷 기준 역할 전용 (5단계 후보 비교에 다시 들고 오지 않음)
- 베이스라인 점수의 산출도 같은 룰 적용:
  - seed=42 + seed=100 두 결과 모두 셀프체크 v3.1 통과 시 → 두 seed 평균
  - 한쪽만 통과 시 → 통과한 seed 단독값
  - 둘 다 시그널 → 베이스라인 자체 신뢰성 의심 (재학습)
- 현 시점 베이스라인 seed=100에서 셀프체크 시그널 (Detection rate 76.6%) → seed=42 단독값 27.7%로 고정합니다.

---

## 3단계: Top-1 상위 3개 (압축)

- 임계값 통과 모델 중 Top-1 정답률 상위 3개로 좁힘
- 3개로 자른 이유: 5단계 점수차 분기 판단에 필요한 최소 단위

---

## 4단계: seed=100 재학습 (안정화)

### 대상 (v3.7 명확화)
- 상위 3개 후보만 seed=100 추가 학습
- 베이스라인은 4단계 대상 아님 (자기가 자기 컷이라 자기 충족 예언 발생, 팀 협업 원칙)

### 점수 산출 룰 (v3.7 추가)
각 후보의 최종 점수 산출:
- 두 seed 모두 셀프체크 v3.1 통과 → 두 seed 평균 사용
- seed=42만 통과 → seed=42 단독값 (seed=100 셀프체크 실패로 평균에서 제외)
- seed=100만 통과 → seed=100 단독값
- 두 seed 다 시그널 → 모델 자체 신뢰성 의심 → 추가 seed로 재학습 또는 후보 탈락

33장 small sample 환경의 우연성을 걸러내고, 셀프체크 실패 결과는 신뢰성이 낮아 평균에 포함하지 않습니다.

---

## 5단계: 후보 비교 테스팅 (1위 결정 + 분석)

### 비교 대상 (v3.7 명확화)
- 후보들끼리만 비교 (베이스라인 무관)
- 베이스라인은 1~2단계에서 컷 기준 역할로 끝났음

### (a) 1위 결정

**상위 3개 점수차 > 5%p (격차 큼)**

→ 1위 후보 신뢰성 검증 3중 통과 (seed=42 + seed=100 둘 다 검증)
1. 정답 분포: 특정 알약에 쏠림 없는지 (mode collapse)
2. 오답 패턴: 의미 있는 혼동인지 (같은 성분/유사 외형끼리)
3. 학습 데이터 누수: 평가셋이 학습/검증셋에 들어가지 않았는지

→ 통과 시 1위 확정

**상위 3개 점수차 ≤ 5%p (격차 작음)**

→ Kaggle 제출로 외부 검증 (33장 small sample 한계 우회)
→ ±2%p 이내 동률 → CPU 추론 속도 빠른 쪽 (HealthEat 배포 컨셉 = CPU 환경)
→ 그래도 동률 → 모델 사이즈 작은 쪽

### (b) 깊이 비교 분석 (보고서/발표 콘텐츠)

후보 3개의 약점/강점 비교. (a)와 병행 진행.

**케이스 묶음 분석 (절대 개수 표시)**
- 33장을 시각적 특성으로 묶어서 모델별 정답 개수 비교
- 묶음 기준: 알약 개수 / 색상 / 모양 / 각인 명확도
- % 표시 안 함 (33장 small sample이라 오해 유발), 절대 개수만

**혼동 시각화**
- 오답 케이스 매트릭스 테이블 (정답 / 예측 / 혼동 유형 분류)
- Sankey diagram (정답 → 예측 흐름, 색상 묶음으로 의미 있는 혼동 가시화)

---

## 6단계: 배포 적합성 (1위 확정 후)

HealthEat 웹 데모 컨셉 = 무료 CPU 환경에서 동작 가능해야 함.

### 측정 환경 (v3.7 실측 확인)
**Hugging Face Spaces 무료 환경**:
- 2 vCPU 코어
- 16GB RAM
- 50GB 디스크 (비영구, 48시간 미사용 시 sleep)
- GPU 없음
- 비용 무료

(공식 문서: huggingface.co/docs/hub/en/spaces-overview, 2026-04-29 확인)

33장 평균 추론 시간 측정.

### 판정 기준

부트캠프 발표 데모 컨셉 + 사용자 UX 한계 기반:

- CPU 2초 이내 → 직접 배포 가능 (Streamlit/Gradio + best.pt 그대로)
- CPU 2~5초 → 데모 가능하지만 사용자 경험 저하, 최적화 권장
- CPU 5초 초과 → 부적합 → 대응 진입

### 부적합 시 대응

- 1순위: ONNX 변환 (속도 2~5배 향상 가능)
- 2순위: int8 양자화
- 3순위: 차순위 후보 모델로 교체 (4단계 2위 모델 검증)

---

## 셀프 체크 (결과 JSON 제출 전 필수)

### 박스 sanity
- [ ] 알약 여러 개 들어있는 이미지에 prediction 박스가 GT 개수와 비슷하게 나왔는가
- [ ] pred 박스 mean이 GT mean(1.4) 근처인가 (1로 고정이거나 3 이상으로 고정이면 박스 컷 의심)
- [ ] 박스 1개 컷, 박스 3개 강제 컷 둘 다 비정상

### Top-3 sanity
- [ ] 박스당 top3_class_name 리스트 길이가 3인가
- [ ] top3_confidence도 길이 3인가

### Detection rate self-check
- [ ] Detection rate(IoU 0.5 매칭) 80% 이상인가
- [ ] 80% 미만이면 학습 부족 or 추론 박스 변환 문제 의심

### Conf 스캔
- [ ] conf 단일값 X, 0.01/0.05/0.1/0.2 스캔 후 본인 모델 best값 사용

### 자체 검증 셀

```python
import json

with open('YOUR_RESULT.json') as f:
    d = json.load(f)

# 1. 박스당 top3 길이
top3_lens = [len(p['top3_class_name']) for item in d['results'] for p in item['predictions']]
if top3_lens:
    print(f'박스당 top3 길이: min={min(top3_lens)}, mean={sum(top3_lens)/len(top3_lens):.2f}')
    print('  → 평균 1에 가까우면 후처리 미완. 3에 가까워야 정상.')

# 2. 이미지당 박스 수
pred_counts = [len(item['predictions']) for item in d['results']]
gt_counts = [len(item['gt_bboxes']) for item in d['results']]
print(f'\n이미지당 pred 박스 수: min={min(pred_counts)}, max={max(pred_counts)}, mean={sum(pred_counts)/len(pred_counts):.1f}')
print(f'이미지당 GT 박스 수:   min={min(gt_counts)}, max={max(gt_counts)}, mean={sum(gt_counts)/len(gt_counts):.1f}')
print('  → pred mean이 GT mean(1.4) 근처여야 정상. 1로 고정이거나 3 이상 고정이면 박스 컷 의심.')

# 3. Detection rate
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    aA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    aB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    u = aA + aB - inter
    return inter/u if u > 1e-9 else 0

matched, total = 0, 0
for item in d['results']:
    for gt in item['gt_bboxes']:
        total += 1
        if any(iou(gt['bbox'], p['bbox']) >= 0.5 for p in item['predictions']):
            matched += 1
print(f'\nDetection rate (IoU>=0.5): {matched}/{total} = {matched/total*100:.1f}%')
print('  → 80% 미만이면 학습/추론 어딘가 문제.')
```

---

## 현 시점 통과 모델 (2026-04-29 v3.7 기준)

### 1~2단계 임계값 통과 모델

| 모델 | 담당 | params | seed=42 Top-1 | seed=42 Top-3 | 임계값 통과 |
|---|---|---|---|---|---|
| RTMDet-tiny | wony98 | 4.91M | 34.0% | 46.8% | 통과 (1.69배) |
| YOLOv8s | 팀원 A | 11.18M | 21.3% | 29.8% | 통과 (1.08배) |
| RT-DETR | 팀원 D | 33.0M | 21.3% | 27.7% | 동률 |
| YOLOv8n 베이스라인 | 팀원 A | 3.35M | 14.9% | 27.7% | (기준선, 후보 X) |
| YOLOX-S | 팀원 C | 9.0M | 21.3% | 23.4% | 미달 |
| FRCNN MobileNetV3 | 팀원 A | 19.53M | 6.4% | 14.9% | 미달 |
| YOLO11n | 팀원 D | 2.6M | - | 27.7% | (Top-1 낮음) |
| YOLO11s | 팀원 D | - | - | 29.8% | (Top-1 낮음) |

### 3단계: Top-1 상위 3개 (후보)

1. RTMDet-tiny (Top-1 34.0%)
2. YOLOv8s (Top-1 21.3%)
3. RT-DETR (Top-1 21.3%): 동률

### 4단계: seed=100 재현성 검증 결과

| 후보 | seed=42 (det/Top-3) | seed=100 (det/Top-3) | 셀프체크 | 평균 Top-3 | 산출 기준 |
|---|---|---|---|---|---|
| RTMDet-tiny | 97.9% / 46.8% | 95.7% / 38.3% | 둘 다 통과 | 42.6% | 두 seed 평균 |
| YOLOv8s | 91.5% / 29.8% | 93.6% / 19.1% | 둘 다 통과 | 24.5% | 두 seed 평균 |
| RT-DETR | 91.5% / 27.7% | (대기) | (대기) | (대기) | (대기) |

**참고: 베이스라인 재현성 (4단계 대상 아님, 부록 정보)**

| | seed=42 (임계값 산출 기준) | seed=100 |
|---|---|---|
| Detection | 91.5% (통과) | 76.6% (실패) |
| Top-3 | 27.7% | 29.8% |

→ seed=100 셀프체크 실패. 임계값 산출에서 제외. 임계값 = seed=42 단독 27.7% 유지.

### 5단계: 후보 비교 (RT-DETR 대기 중 시뮬)

격차: RTMDet 42.6% - YOLOv8s 24.5% = 18.1%p (>5%p)

→ 격차 큼 분기 → RTMDet 신뢰성 검증 3중 → 통과 시 1위 확정

---

## 변경 이력

- v3.0: 임계값 베이스라인 Top-3 × 1.0으로 고정
- v3.1: 셀프 체크 섹션 신설
- v3.2: 임계값 25.5% → 27.7% 갱신 + tie-break 룰 명문화 + 재현성 검증 단계 신설
- v3.3: 통과 모델 갱신 + 비교 테스팅 분석 방식 신설
- v3.4: 6단계 배포 적합성 검증 (CPU 추론 속도) 신설
- v3.5: 5단계 격차 작음 분기에서 동률 깨는 기준을 GPU → CPU 추론 속도로 변경
- v3.6: 5단계 구조 통합, "1위 결정"과 "비교 테스팅 분석"을 5단계 안에 (a)/(b)로 묶음
- v3.7: 셀프체크 통과한 seed만 평균에 포함 룰 추가 + 베이스라인 역할 명확화 (4단계 대상 아님, 5단계 비교 대상 아님, 팀 협업 원칙 반영)
