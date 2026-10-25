# HealthEat 평가 입력 검사와 비교 (A단계)

기존 채점기를 재사용해 평가 자료와 정답의 일치를 확인하고, 저장된 예측을 비교합니다. 모델을 로드하거나 추론하지 않습니다. 합성 MLflow 검증은 새 시험 저장소에만 기록합니다.

별도 정답 버전과 재채점 도구, 동봉할 매핑 근거는 [GT v2와 정오표](LABEL_V2.md)를 참고하세요. 기존 v1 경로의 검사 조건은 그대로입니다.

## 실행 범위

- 입력과 정답: 기존 학습 노트북과 같은 YOLO→픽셀 좌표 변환, `pill.yaml` 클래스 매핑, 기존 `gt_digest` 대조.
- 검증·채점·해시·IoU: `experiments/mlflow/backfill_runs.py`의 함수를 import해 재사용합니다.
- 비교: 과거 박스 저장 순서로 가장 높은 IoU의 미사용 새 박스를 매칭합니다. IoU≥0.5, 동점은 새 박스 저장 순서입니다. 각 방향의 미매칭 수, 대응 클래스·점수 차이를 기록합니다. 과거 예측은 정답이 아닙니다.
- 기록: `record-synthetic`은 합성 검증 묶음만 받으며 매번 새로운 로컬 SQLite 저장소를 만듭니다. 과거 기록과 실제 모델 실행으로 표시할 수 없습니다.

모델 import·설치·추론·재학습·네트워크 서비스·모델 레지스트리·기존 백필 DB 변경은 없습니다. B단계는 별도 승인 대상입니다.

## 현재 실제 자료의 확인 결과

33장·47개 박스를 재구성한 GT는 과거 평가 JSON 4개의 GT와 해시가 같습니다. 이는 과거 채점의 입력을 재구성했다는 뜻이며, 정답의 의미까지 옳다는 증명은 아닙니다.

`answer_key.json`의 `class_idx`는 현재 `pill.yaml` 인덱스와 다른 항목이32개입니다. 당시 GT 생성 코드는 해당 필드를 사용하지 않고 라벨의 클래스 ID와 클래스 이름 매핑을 사용했습니다. 새 도구도 같은 기준을 사용하되 차이를 경고로 남기며 원본을 수정하지 않습니다.

또한 `프로스카정.png`의 라벨+매핑은 두 박스를 `콜리네이트연질캡슐 400mg`으로 표시하지만 부가 정답표의 이름은 `프로스카정`입니다. 이 충돌은 어느 쪽이 정답인지 결정하지 않고 `NEEDS_LABEL_REVIEW`로 남깁니다. 현재 실제 자료에 대한 `prepare`는 `GT_REFERENCE_CONFLICT`로 중단합니다. 이미지·라벨·과거 JSON을 임의로 변경하거나 예외 옵션으로 우회하지 않습니다.

## 명령

저장소 루트에서 기존 `.venv-mlflow`를 사용합니다. 아래 `<NEW_...>` 경로는 아직 존재하지 않는 저장소 밖의 작업 폴더로 지정합니다.

```powershell
# 현재 자료를 읽기 전용으로 감사. 충돌/GT 불일치가 있으면 보고서를 남기고 exit 2.
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/evaluation.py audit-gt --project-root . --output "<NEW_AUDIT_DIR>"

# 정답 검사를 통과하는 입력만 준비. 현재 실제 자료는 위 충돌 때문에 중단됨.
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/evaluation.py prepare --project-root "<FIXTURE_PROJECT>" --predictions "<FIXTURE_JSON>" --kind synthetic --output "<NEW_BUNDLE>"

.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/evaluation.py compare --reference "<REFERENCE_BUNDLE>" --candidate "<CANDIDATE_BUNDLE>" --output "<NEW_COMPARISON>"
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/evaluation.py record-synthetic --bundle "<SYNTHETIC_BUNDLE>" --workspace "<NEW_TRACKING_DIR>"
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/evaluation.py verify-record --workspace "<TRACKING_DIR>"
```

`prepare --kind historical`은 과거 입력을 준비할 때의 구분입니다. 모델·설정 파일의 현재 해시를 남기더라도 과거 예측을 해당 모델이 만들었다는 연결은 `UNRESOLVED`로 유지합니다. `forward`는 A단계에서 허용하지 않습니다. `--kind synthetic`은 사용자가 합성 입력임을 명시하는 계약이며 자동 판별 기능이 아닙니다.

이미지 집합·GT·후처리 조건이 다르면 비교 보고서에 전체 지표 차이를 표시하지 않습니다. 파일·코드 해시가 바뀐 묶음은 재사용하지 않으며 새 출력 폴더로 다시 준비해야 합니다. 입력 경로는 내부 확인용 `local_sources.json`에만 있고 MLflow 아티팩트에는 포함하지 않습니다. 공개 전에 산출물의 데이터·파일명 공개 범위도 별도로 검토해야 합니다.

## 테스트

```powershell
.\.venv-mlflow\Scripts\python.exe -B experiments/mlflow/test_backfill.py
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/test_evaluation.py
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/test_tracking.py --workspace "<NEW_SYNTHETIC_TEST_DIR>"
```

쓰기 가능한 임시 폴더가 필요한 환경에서는 해당 테스트 프로세스의 `TEMP`·`TMP`를 전용 작업 폴더로 지정합니다. 테스트 조건을 줄이거나 건너뛰지 않습니다. 단위 테스트는 임시 합성 파일을 정리합니다. 통합 검사는 완료·실패 상태의 합성 저장소를 모두 보존합니다.

통합 검사는 정상 기록·읽기 대조, 기존 출력 거부, 아티팩트/태그 변조 탐지, 입력 변경 시 실패 기록과 재시도 거부, 모델 미수입을 확인합니다. 원본 데이터와 기존 `.mlflow-backfill`은 입력으로 열지 않습니다. 원본 파일 보존은 별도 작업 전후 해시로 확인합니다.
