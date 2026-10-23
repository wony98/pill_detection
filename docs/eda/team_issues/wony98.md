# 3팀 EDA: 데이터 품질 이슈 리스트

**검토 대상**: Kaggle train + AI Hub TL/TS 시리즈 1, 3, 4, 5, 6, 7, 8
**데이터 규모**: 41,225 annotation / 10,697 이미지 / 118 클래스

**상태 표기**:
- 확증: 실측으로 결함 확인됨
- 기각: 의심했으나 실측 결과 문제 없음
- 정보성: 결함은 아니지만 기록해둔 관찰 사항

---

## 이슈 목록

### [A] bbox 경계 초과 (확증)
- **정의**: `x < 0 or y < 0 or x+w > 976 or y+h > 1280`
- **실측**: 58 annotation (AI Hub 57 + Kaggle 1)

---

### [B] area 불일치 (확증)
- **정의**: `|area_raw - bbox_w * bbox_h| / (bbox_w * bbox_h) > 500%` (극단)
- **실측**: 극단 40건 / 비극단 216건

---

### [C] IoU=1.0 완전 중복 (확증)
- **정의**: 같은 이미지 내 bbox IoU ≥ 0.999
- **실측**: 71쌍 / 142 annotation

---

### [D] 음수 bbox 좌표 (기각)
- **정의**: bbox 좌표에 음수 값 존재
- **실측**: 0건

---

### [G] 각도 커버리지 부족 (기각)
- **정의**: Kaggle train에 특정 각도(90도) 없을 가능성
- **실측**: Kaggle train 90도 36.3% 존재

---

### [H] 메타 필드 누락 (정보성)
- **정의**: 학습 무관 메타 필드 누락
- **실측**: 18 annotation

---

### [I] bbox 중심 4코너 쏠림 (확증)
- **정의**: bbox 중심 좌표의 2D 분포가 4코너에 집중, 중앙(MC) 공백
- **실측**: MC 0.87% (균등 11.1% 대비 0.08x), 4코너 합 83.67%
- **참고**: 사각형 고정 지그 구조 (산점도 4-클러스터)

---

### [J] test 라벨 익명화 (정보성)
- **정의**: Kaggle test 라벨이 공개되지 않아 성능 분석 제한
- **참고**: 대회 규칙상 불가피

---

### [K] test 배경색 shift (기각)
- **정의**: Kaggle test 배경이 train(흰색)과 다름 (파랑/초록) 의심
- **실측**: 육안 확증 결과 train과 동일

---

### [L] iscrowd 이상치 (확증)
- **정의**: `iscrowd == 58548` (비정상 값)
- **실측**: 1 annotation

---

### [M] area_raw list 타입 (확증)
- **정의**: `area_raw` 필드가 숫자가 아닌 list
- **실측**: 1 annotation

---

### [N] 이미지당 객체 수 초과 (정보성)
- **정의**: 이미지당 annotation 수가 기대값(4) 초과
- **실측**: 90 annotation / 17 이미지
- **참고**: 실측 규명 결과 Q(near-dup) 이슈로 전부 설명됨

---

### [O] bbox 크기 극단 (정보성)
- **정의**: bbox 크기가 극단적으로 작거나 큼
- **실측**: 29 annotation
- **참고**: 대부분 A(경계 초과)와 중첩

---

### [P] 라벨 누락 (확증)
- **정의**: combo 폴더명에 명시된 알약 종 수보다 실제 JSON annotation이 적음
- **탐지**: `len(combo.split('-')) - 1 > 이미지당 annotation 수`
- **실측**: 324장 / 921 annotation (AI Hub 298장 + Kaggle train 26장)

---

### [Q] Near-duplicate 라벨 (확증)
- **정의**: 같은 이미지에 같은 클래스 bbox 2개, IoU > 0.5
- **탐지**: 이미지 내 bbox pair 전수 IoU 계산 → `same_class AND IoU > 0.5`
- **실측**: 16장 / 18 annotation
- **참고**: 다른 클래스 + IoU>0.5는 정상 겹침 (알약이 물리적으로 붙어있는 경우) 15쌍 발견

---

### [R] 라벨 오류 (확증)
- **정의**: 아래 중 하나라도 해당
  - (a) Q 처리 후에도 `actual > expected` (과다 잔여)
  - (b) 같은 이미지에 같은 `img_dl_name` 2번 이상
  - (c) Q/R 처리 후 `expected > actual` (누락 잔여)
- **실측**: 6장 / 27 annotation (전부 `aihub_ts1_tl1`)
- **참고**: 숫자상 정상(`expected == actual`)인데 클래스가 꼬인 "숨은 R" 1장 존재. 오직 "같은 클래스 2번" 규칙으로만 탐지됨

---

### [좌표 정확 일치 중복] (확증)
- **정의**: `(image_file, round(bbox))` 키 정확 일치
- **실측**: 18 annotation
- **참고**: 다른 클래스끼리 좌표 일치 발생 가능 → 라벨 오류일 수 있음 (실제로 이슈 R에서 1장 발견)

---

### [parse_error] JSON 손상 / 이미지-라벨 매칭 실패 (확증)
- **정의**: JSON 파일 깨짐 또는 이미지-라벨 짝 안 맞음
- **실측**: 324 annotation (JSON 손상 1, orphan_label 120, orphan_image 48, bbox_format_error 155)

---

### [cat_name 단일값] 학습 타겟 부적절 (확증)
- **정의**: AI Hub의 `cat_name` 필드가 전부 "Drug" 단일값
- **참고**: 학습 타겟으로 `img_dl_name` 사용 필수

---

### [bbox scale source 차이] (정보성)
- **정의**: Kaggle과 AI Hub 간 bbox 크기 분포 차이
- **실측**: Kaggle bbox 중앙값 53,600 vs AI Hub 82,000

---

### [dl_idx source별 분리] (확증)
- **정의**: `dl_idx` 값이 Kaggle과 AI Hub에서 같은 약품에 대해 다르게 발급됨 (Kaggle = AI Hub + 1)
- **실측**: 54종 공통 약품 전부 diff=1 패턴
- **참고**: `dl_idx` 기준 병합 시 같은 약이 source별로 분리됨 → `img_dl_name` / `img_dl_mapping_code` / `img_drug_N` 3중 키 병합 필수

---

## 요약

| 상태 | 이슈 수 | 코드 |
|---|---|---|
| 확증 | 13 | A, B, C, I, L, M, P, Q, R, 좌표중복, parse_error, cat_name, dl_idx |
| 기각 | 3 | D, G, K |
| 정보성 | 5 | H, J, N, O, bbox scale |

---

## 검토 방법론

1. 숫자 기반 탐지 → outlier 전수 시각화 → 육안 확증
2. "숫자 OK ≠ 데이터 OK": 이슈 R "숨은 1장"은 숫자로는 정상이지만 클래스가 꼬인 케이스. 시각 확증 없었으면 못 찾음.

---

## 참조 시각 자료

`cell20_issue_PQR_confirmation.png`: P / Q / R 대표 샘플 + 정상 겹침 1장 (3x3 그리드)
