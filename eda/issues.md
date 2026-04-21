# EDA 이슈 리스트 (경구약제 객체검출)

> **용도**: 가공된 데이터 손상/오류/이상치 탐지 결과. 팀미팅에서 전처리 방향 결정용.
> **최종 갱신**: 2026-04-21 (Cell 19~24 추가, 자동 스크리닝 반영)
> **근거 노트북**: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb)
> **이슈 포맷**: 심각도(🔴🟡🟠🟢) · 카테고리 · 제안 해결책 옵션 A/B/C

---

## 요약 테이블

| ID | 제목 | 심각도 | 카테고리 | 건수 | 근거 셀 | 상태 |
|---|---|---|---|---|---|---|
| A | bbox 이미지 범위 초과 | 🔴 Blocker | Annotation Quality | 1 | Cell 7 | open |
| B | area_meta ≠ w×h | 🟢 Info | Annotation Quality | 6 | Cell 7 | open |
| C | 동일 좌표 이중 라벨링 (IoU=1.0) | 🟡 Major | Annotation Quality | 3쌍 | Cell 6, 17, 24 | open |
| D | 누락 어노테이션 (3경로 교차) | 🟡 Major | Data Integrity | 8 | Cell 8 | open |
| F | 파일명 pos 2='1' 이상 | 🟠 Minor | Data Integrity | 4 | Cell 10 | open |
| G | 각도 커버리지 부족 | 🟡 Major | Metadata & Bias | 74조합 | Cell 11 | open |
| H | 가바토파정 100mg 부가 메타 전량 누락 | 🟠 Minor | Annotation Quality | 18 | Cell 15, 15-2 | open |
| I | Bbox 중심 위치 주변부 정형 배치 (중앙 공백) | 🟡 Major | Metadata & Bias | 763 (구조적) | Cell 19, 22 | open |
| J | Test 파일명 메타 정보 전무 (단순 순번) | 🔴 Blocker | Data Integrity | 842 | Cell 20 | open |
| K | Train/Test 픽셀 통계 분포 차이 | 🟡 Major* | Metadata & Bias | 1074 (전수) | Cell 23 | open (baseline 후 재평가) |

> Cell 24 자동 스크리닝(5개 플래그)으로 Train 232장 전수 검증 완료 → **신규 이슈 0건**, 기존 이슈 A/C 자동 재확인 성공.

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
- 자동 스크리닝 재확인: Cell 24 F1 플래그 1건으로 교차 확인 완료

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
- 자동 스크리닝 재확인: Cell 24 F3(3건) + F7(좌표 동일 3건) 양쪽 플래그에서 동일 3쌍 재확인

**대상 파일 상세** (Cell 24 검증):
| 파일 | cat_A | cat_B | 좌표 | area |
|---|---|---|---|---|
| K-001900-016548-019607-033009_...70_000_200.png | 16548 | 33009 | (88,255,366,209) | 76,494 |
| K-003351-020238-031863_...70_000_200.png | 3351 | 20238 | (446,844,190,191) | 36,290 |
| K-003351-029667-031863_...70_000_200.png | 3351 | 31863 | (178,249,205,203) | 41,615 |

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
- Cell 7 (파일명 기반) / Cell 8 (시각 검증) / Cell 12-2 (조합 폴더 기준) 3경로 100% 일치 확정

**누락 상세** (Cell 7 출력):
- id=3351 (일양하이트린정 2mg): 4회
- id=20014 (마도파정): 1회
- id=32310 (글리아타민연질캡슐): 1회
- id=33880 (글리틴정): 1회
- id=4543 (에어탈정): 1회

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
- Test는 파일명 메타 없음(이슈 J) → train 내부 특이사항으로만 의미

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
- 영향 범위: 74조합 (전체 114조합 중 65%)
- 재현 방법: 조합별 angle_set_counter에서 `len(angles) <= 2` 필터
- 1장만 있는 조합 36개 (각도 2개 누락) — 특정 알약 세트가 특정 각도 1개에서만 학습됨
- **Test 파일명에 각도 메타 없음(이슈 J)** → test 각도 분포 확인 불가 → 증강으로만 대응 가능

**가능한 영향**:
- 회전/시점 변화에 취약한 모델이 될 가능성
- test 이미지에서 학습 못 본 각도로 찍힌 약 → 검출 실패

**제안 해결책**:
1. **Option A**: Augmentation(회전, perspective warp)으로 보완 — 가장 현실적
2. **Option B**: 누락 각도 조합 패턴 분석 후 특정 클래스에 대해 보강 촬영 — 실행 비용 높음
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

### [I. Metadata & Bias] Bbox 중심 위치 주변부 정형 배치 (중앙 공백)

**발견자**: 승원
**심각도**: 🟡 Major
**발견일**: 2026-04-21
**카테고리**: Metadata & Bias

**내용**:
이미지 9등분 그리드에서 중앙(MC) 셀의 bbox 비율 0.79% (6/763) — 균등 기준 11.11%의 약 1/14 수준. 4코너(TL/TR/BL/BR) 평균 15.6%로 밀집. 좌우·상하 대칭이고, cy 중앙값 0.618 vs 평균 0.499 차이는 상하 쌍봉 분포(중간 공백)의 증거. Cell 3 "이미지당 3박스 151장(65%)" 패턴의 시각적 확증 — 3개=삼각 배치 / 4개=사각 배치.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 19
- 영향 범위: 763 전체 (분포 차원의 구조적 편향)
- 재현 방법: bbox 중심 좌표 정규화 후 9등분 그리드 집계
- **Cell 22 test 30장 샘플링으로 test도 동일 "주변 4코너" 배치 확인** → train/test 스타일 일관

**가능한 영향**:
- 일반적 "중앙 편향 → random crop 완화" 통념과 정반대 패턴
- Center crop / Random crop은 주변부 알약을 잘라 학습 품질 저하
- Train/test 배치 스타일 동일 확인되어 예측 가능성 확보

**제안 해결책**:
1. **Option A**: Mosaic + Rotation 증강 적용 (YOLO 기본) — 4코너 패턴 분산
2. **Option B**: 증강 제한 (center/random crop 제외, 회전만) — 안전하나 다양성 부족
3. **Option C**: Resize만 사용, 증강 최소화 — 보수적 접근

→ **추천**: A (mosaic + rotation이 실전 검증된 조합)

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

### [J. Data Integrity] Test 파일명 메타 정보 전무 (단순 순번)

**발견자**: 승원
**심각도**: 🔴 Blocker
**발견일**: 2026-04-21
**카테고리**: Data Integrity

**내용**:
Test 842장 전체가 `1.png`, `10.png`, ..., `1003.png` 형태의 단순 순번 파일명. Train의 7토큰 구조(K-code, 촬영조건, 각도, 번호)가 test에는 전혀 없음. 토큰 수 1개 842장(100%). K-code 조합 공통 0개 / Test only 0개 / Train only 5개.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 20
- 영향 범위: 842장 전체 (100%)
- 재현 방법: `[f.name for f in TEST_DIR.iterdir()][:5]` → 순번 파일만 확인됨

**가능한 영향**:
- **Test는 블라인드 평가 세트** — 촬영조건(각도/조합)을 알 수 없음 → train에서 발견한 메타 편향(이슈 G, I)을 test 기준으로 재평가 불가
- **Train의 파일명 기반 검증(Cell 7의 category_id 기대치 등) test에 재사용 불가** — test 추론 시 이미지 내용만으로 판단
- **이슈 G(각도 커버리지)가 실질적 Blocker** — train에 없는 각도가 test에 있어도 확인 불가 → 증강 필수(옵션 아님)
- Cell 22 시각 샘플링 결과 test 배치/배경 스타일은 train과 유사 확인됨

**제안 해결책**:
1. **Option A**: 이미지 내용 기반 추론만 사용 + 공격적 증강(rotation/perspective/mosaic) 적용 — 최선
2. **Option B**: 데이터 제공자에게 test 메타 제공 여부 문의 — 익명화가 평가 규칙이면 불가능
3. **Option C**: Train 내부 검증만으로 한정하고 test 성능은 "운"에 맡김 — 비권장

→ **추천**: A (이슈 G/I 증강 전략과 통합)

---
**결정**:
**담당자**:
**마감**:
**상태**: open

---

### [K. Metadata & Bias] Train/Test 픽셀 통계 분포 차이

**발견자**: 승원
**심각도**: 🟡 Major (baseline 확인 후 재평가)
**발견일**: 2026-04-21
**카테고리**: Metadata & Bias

**내용**:
Train 232장 / Test 842장 전수 픽셀 통계 비교 결과, 8개 지표(RGB 평균·표준편차, grayscale 평균, 파일 크기) 전부 KS test p=0.0000으로 분포 다름. 평균값 차이는 작으나(2.7~6.6%) **표준편차 차이가 큼(10.6~24.2%)**. 특히 g_std 24.2%, r_std 10.6%, b_std 12.1% — 일관되게 test 분산이 train보다 넓음.

**증거**:
- 코드 위치: [eda/seungwonEDA.ipynb](seungwonEDA.ipynb) Cell 23
- 영향 범위: train 232 / test 842 전수 (100%)
- 재현 방법: RGB 채널 평균/std, grayscale 평균, 파일 크기 계산 후 KS test
- 시각 자료: `figures/L5_visual_samples/train_test_pixel_stats.png`

**확인된 사실** (데이터 기반):
- 평균값은 차이 작음 (<7%) — 분포 중심은 유사
- 표준편차는 일관되게 test 쪽이 큼 (test 이미지들이 더 다양한 픽셀 분포)
- 파일 크기도 test가 +2.8% 큼 (1711KB → 1759KB)
- Cell 21/22 30장 육안 샘플에선 "유사" 판정됐으나 전수 통계는 분산 차이 확인

**가능한 영향** (baseline 없이 확정 불가):
- Test 이미지가 train보다 시각적으로 다양할 가능성
- 증강으로 커버 가능한 범위인지 미확인
- 실제 성능 저하 여부는 baseline 돌려야 판정 가능

**제안 해결책**:
1. **Option A**: Baseline 돌려 train/val/test 점수 gap 확인 → gap 크면 증강 강화
2. **Option B**: 예방적 증강 (ColorJitter, RandomBrightnessContrast) 무조건 적용
3. **Option C**: 현상 기록하고 모델링 단계에서 재판단

→ **추천**: A (baseline으로 실제 영향 측정 후 A/B 결정)

---
**결정**:
**담당자**:
**마감**:
**상태**: open (baseline 후 재평가)

---

## 참고: 주요 발견 (이슈 아님)

- 232 이미지, 56 클래스, 763 어노테이션, 해상도 976×1280 단일 (train 기준)
- 클래스 불균형 51배: 일양하이트린정 2mg 153건 / 하위 10개 클래스 모두 3건
- 모든 bbox COCO large, 이미지 대비 최대 21.81% (거대 bbox 없음)
- category_id ↔ name 완전 1:1 (COCO 병합 안전)
- 물리적 bbox 겹침 0건 (IoU>0.5 중 IoU=1.0 3쌍만)
- 5-fold CV는 하위 17 클래스(30%)에 제약 → 3-fold 또는 iterative multi-label stratification 필요
- Train 자동 스크리닝(Cell 24 5개 플래그) 결과: 의심도 0 = 203장(87.5%), 신규 이슈 0건
- Train/Test 30장 시각 샘플링(Cell 21/22): 배경/배치/각도 스타일 육안 유사

## EDA 프레임워크 진행 현황 (L0~L5, 치트시트 기준)

| Layer | 상태 | 커버 셀 |
|---|---|---|
| L0 Sanity | 완료 | Cell 6, 7, 8, 9, 10, 11, 12-1, 12-2, 13, 14, 24 |
| L1 Target (Class 분포) | 완료 | Cell 3, 4 |
| L2 Input (Bbox/Image) | 완료 | Cell 5, 16, 19 |
| L3 Input × Target | 부분 | Cell 15, 15-2 (메타), Cell 23 (분포 비교) — 본격 미착수, 팀미팅 이후 논의 |
| L4 Split & Leakage | 부분 | Cell 18 (split 가능성 논의) — 본격 미착수, 팀미팅 이후 논의 |
| L5 시각 검증 | 완료 | Cell 6, 8, 17, 21, 22, 24 |
