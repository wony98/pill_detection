# EDA 이슈 리스트 (경구약제 객체검출)

> **용도**: 가공된 데이터 손상/오류/이상치 탐지 결과. 팀미팅에서 전처리 방향 결정용.
> **최종 갱신**: 2026-04-21
> **근거 노트북**: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb)
> **이슈 포맷**: 심각도(🔴🟡🟠🟢) · 카테고리 · 제안 해결책 옵션 A/B/C

---

## 요약 테이블

| ID | 제목 | 심각도 | 카테고리 | 건수 | 근거 셀 | 상태 |
|---|---|---|---|---|---|---|
| A | bbox 이미지 범위 초과 | 🔴 Blocker | Annotation Quality | 1 | Cell 7 | open |
| B | area_meta ≠ w×h | 🟢 Info | Annotation Quality | 6 | Cell 7 | open |
| C | 동일 좌표 이중 라벨링 (IoU=1.0) | 🟡 Major | Annotation Quality | 3쌍 | Cell 6, 17 | open |
| D | 누락 어노테이션 (3경로 교차) | 🟡 Major | Data Integrity | 8 | Cell 8 | open |
| F | 파일명 pos 2='1' 이상 | 🟠 Minor | Data Integrity | 4 | Cell 10 | open |
| G | 각도 커버리지 부족 | 🟡 Major | Metadata & Bias | 74조합 | Cell 11 | open |
| H | 가바토파정 100mg 부가 메타 전량 누락 | 🟠 Minor | Annotation Quality | 18 | Cell 15, 15-2 | open |

> Cell 19~ 진행 중 신규 이슈 **I, J, K...** 추가 예정.

**심각도 기준**
- 🔴 **Blocker** — 해결 전 모델링 불가 / 평가 무의미
- 🟡 **Major** — 성능에 직접 영향, 반드시 대응
- 🟠 **Minor** — 개선하면 좋음, 우선순위 낮음
- 🟢 **Info** — 공유 목적, 해결 대상 아님

---

## 상세 이슈

### [A. Annotation Quality] bbox 이미지 범위 초과

**발견자**: 승원
**심각도**: 🔴 Blocker
**발견일**: 2026-04-20
**카테고리**: Annotation Quality

**내용**:
한 어노테이션의 bbox x 좌표가 이미지 크기를 크게 벗어남. `bbox_x=6567`이지만 이미지 너비 `img_w=976`으로 6.7배.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 7
- 영향 범위: 1건 / 763 어노테이션 (0.13%)
- 재현 방법: `df[(df['bbox_x'] + df['bbox_w']) > df['img_w']]`

**가능한 영향**:
- 학습 시 좌표 클리핑 없으면 `NaN`/loss 폭주 가능
- 해당 이미지 학습 불가 → 클래스 손실

**제안 해결책**:
1. **Option A**: 해당 어노테이션 제거 — 간단, 데이터 1건 손실
2. **Option B**: JSON 원본 재확인 후 좌표 복구 — 복구 가능 시 베스트
3. **Option C**: 좌표 클리핑 (x = min(x, img_w - w)) — 학습엔 돌아가지만 라벨 부정확

→ **추천**: B 시도 후 실패 시 A

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

### [B. Annotation Quality] area_meta ≠ w×h

**발견자**: 승원
**심각도**: 🟢 Info
**발견일**: 2026-04-20
**카테고리**: Annotation Quality

**내용**:
JSON의 `area` 필드 값이 `bbox_width × bbox_height`와 불일치하는 케이스 6건. 10% 이상 오차.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 7
- 영향 범위: 6건 / 763 어노테이션 (0.79%)
- 재현 방법: `|area_meta - bbox_w*bbox_h| / (bbox_w*bbox_h) > 0.1`

**가능한 영향**:
- 실제 bbox 좌표는 정상 — 메타 필드만 오차
- COCO 평가에서 `area`는 small/medium/large 버킷 구분용. 본 데이터는 전부 large라 영향 거의 없음

**제안 해결책**:
1. **Option A**: 무시 — 평가 영향 없음
2. **Option B**: `w*h`로 재계산하여 덮어쓰기 — 혹시 모를 하류 오류 차단

→ **추천**: A (현재 범위에서 무시)

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

### [C. Annotation Quality] 동일 좌표 이중 라벨링 (IoU=1.0)

**발견자**: 승원
**심각도**: 🟡 Major
**발견일**: 2026-04-20
**카테고리**: Annotation Quality

**내용**:
같은 이미지 내 bbox 2개가 완전히 동일한 좌표 (IoU=1.0) 쌍이 3건. 부분 overlap(0.5 < IoU < 1)은 0건.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 6, 17
- 영향 범위: 3쌍 (6개 어노테이션) / 763
- 재현 방법: 같은 `image_id` 내 IoU 전쌍 계산 → IoU==1.0 필터
- 특이: 2/3 쌍이 `id=3351` 일양하이트린정 2mg 집중 (D, F와 공통)

**가능한 영향**:
- 중복 학습 → 해당 샘플 가중치 2배 → 클래스 편향
- 진짜 동일 약이 겹쳐 놓인 건지, 라벨링 오류인지 구분 필요

**제안 해결책**:
1. **Option A**: 3쌍 수작업 검수 — 이미지 열어서 실제 알약 개수 확인
2. **Option B**: 자동으로 중복 제거 (dedup) — 편하지만 진짜 2개인 경우 정보 손실
3. **Option C**: 유지 — 아무 것도 안 함, 학습 노이즈 감수

→ **추천**: A (수작업 3장만 확인)

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

### [D. Data Integrity] 누락 어노테이션 (3경로 교차)

**발견자**: 승원
**심각도**: 🟡 Major
**발견일**: 2026-04-20
**카테고리**: Data Integrity

**내용**:
이미지-JSON 교차검증(3경로 — 폴더 구조 / 파일명 토큰 / JSON image_id)에서 어노테이션이 누락된 케이스 8건. `id=3351` 일양하이트린정 2mg에 4건 집중.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 8
- 영향 범위: 8건 / 763 (1.05%)
- 재현 방법: `K-<조합>_json` 폴더 내 예상 JSON 수 vs 실제 JSON 수

**가능한 영향**:
- 학습 시 해당 이미지에서 bbox가 있어야 할 위치가 background로 학습됨 → False Negative 유도, recall 저하
- `id=3351`은 C 2/3 + D 4건 + F 4건 **모두 집중** → 이 클래스 데이터 신뢰도 의심

**제안 해결책**:
1. **Option A**: `id=3351` 전수 재검토 후 원본 JSON 복구 — 근본 해결, 시간 소요
2. **Option B**: 누락 8건에 해당하는 이미지 제외 — 간단, 데이터 손실
3. **Option C**: 누락 인정하고 학습 — recall 저하 감수

→ **추천**: A → 실패 시 B

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

### [F. Data Integrity] 파일명 pos 2='1' 이상

**발견자**: 승원
**심각도**: 🟠 Minor
**발견일**: 2026-04-20
**카테고리**: Data Integrity

**내용**:
파일명 변형 토큰 두 번째 위치가 '1'인 케이스 4건. 다른 파일은 모두 '0'. 전부 `K-003351` 조합.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 10
- 영향 범위: 4건 / 232 이미지 (1.72%)
- 재현 방법: 파일명 파싱 후 pos 2 토큰 value_counts

**가능한 영향**:
- 의미 미확인 — 재촬영본인지, 오류본인지, 다른 배치인지 불명
- 학습 영향은 메타 의미에 달려있음 → **정보 확보 후 판단**

**제안 해결책**:
1. **Option A**: 대회 주최측/데이터 제공자에게 파일명 규칙 문의
2. **Option B**: 4장 직접 열어보고 다른 파일들과 시각적 차이 확인
3. **Option C**: 무시 — 4건이라 영향 미미

→ **추천**: B (일단 눈으로 확인 → 특이점 없으면 C)

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

### [G. Metadata & Bias] 각도 커버리지 부족

**발견자**: 승원
**심각도**: 🟡 Major
**발견일**: 2026-04-20
**카테고리**: Metadata & Bias

**내용**:
각도 variation(파일명 E 토큰: 70/75/90) 중 2각도 이하로만 촬영된 조합 74개. 전체 조합 중 상당 비율.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 11
- 영향 범위: 74조합 (전체 조합 수 대비 비율은 Cell 재확인 필요)
- 재현 방법: 조합별 angle_set_counter에서 `len(angles) <= 2` 필터

**가능한 영향**:
- 회전/시점 변화에 취약한 모델이 될 가능성
- test 이미지에서 학습 못 본 각도로 찍힌 약 → 검출 실패

**제안 해결책**:
1. **Option A**: Augmentation(회전, perspective warp)으로 보완 — 가장 현실적
2. **Option B**: 누락 각도 조합 패턴 분석 후 특정 클래스에 대해 보강 촬영 — 불가능(외부 데이터 금지 아님, 실행 비용 높음)
3. **Option C**: 유지 — 모델이 알아서 일반화할 거라 기대

→ **추천**: A (albumentations `Rotate`, `Perspective` 적용)

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

### [H. Annotation Quality] 가바토파정 100mg 부가 메타 전량 누락

**발견자**: 승원
**심각도**: 🟠 Minor
**발견일**: 2026-04-20
**카테고리**: Annotation Quality

**내용**:
`id=16548` 가바토파정 100mg 클래스의 18개 어노테이션 **전부** `drug_shape`, `color_class1` 필드가 None.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 15, 15-2
- 영향 범위: 18건 (이 클래스 100%) / 763 (전체 2.36%)
- 재현 방법: `df[(df['drug_shape'].isna()) | (df['color_class1'].isna())]`

**가능한 영향**:
- **메인 bbox + 클래스 학습에는 무관** — 부가 피처 사용 시에만 영향
- 부가 피처를 multi-task learning으로 쓸 경우 이 클래스만 공백 → 마스킹 필요

**제안 해결책**:
1. **Option A**: 부가 피처 미사용 — 문제 자체 소멸
2. **Option B**: `dl_name`으로 외부 DB(식약처 의약품 검색) 조회 후 수동 보완
3. **Option C**: 부가 피처 사용 시 해당 샘플만 loss masking

→ **추천**: **부가 피처 사용 여부 먼저 결정** → 사용 안 하면 A, 사용하면 B 또는 C

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

## 참고: 주요 발견 (이슈 아님)

- 232 이미지, 56 클래스, 763 어노테이션, 해상도 976×1280 단일 (train 기준)
- 클래스 불균형 51배: 일양하이트린정 2mg 153건 / 하위 10개 클래스 모두 3건
- 모든 bbox COCO large, 이미지 대비 최대 21.81% (거대 bbox 없음)
- category_id ↔ name 완전 1:1 (COCO 병합 안전)
- 물리적 bbox 겹침 0건 (IoU>0.5 중 IoU=1.0 3쌍만)
- 5-fold CV는 하위 17 클래스(30%)에 제약 → 3-fold 또는 iterative multi-label stratification 필요
- test_images 842장 내용은 미확인 (Cell 23에서 처리 예정)

## EDA 프레임워크 진행 현황 (L0~L5, 치트시트 기준)

| Layer | 상태 | 비고 |
|---|---|---|
| L0 Sanity | 진행 중 | Cell 3~12 상당 커버, Cell 19/20/24로 마무리 |
| L1 Target (Class 분포) | 완료 | Cell 14, 18 |
| L2 Input (Bbox/Image) | 진행 중 | Cell 7/16/17 완료, Cell 25/26 예정 |
| L3 Input × Target | 미착수 | **범위 외** — 팀미팅 이후 |
| L4 Split & Leakage | 부분 | Cell 18 stratified 시도만, 본격 미착수 — **범위 외** |
| L5 시각 검증 | 부분 | Cell 6/17/26 일부 샘플만 |
