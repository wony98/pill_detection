# HealthEat 실험 기록 사후 정리

과거 RTMDet 학습 로그와 알약 사진 평가 결과를 로컬 MLflow에 모읍니다. 모델을 다시 학습하거나 추론하지 않습니다. 당시부터 MLflow를 사용한 기록도 아닙니다.

## 2026-10-23 백필 결과

학습 기록 3개와 평가 기록 4개를 등록했습니다. 같은 입력·코드로 다시 실행했을 때 기존 7개 run을 재사용했고, 추가 run은 생기지 않았습니다. 전체 지표 이력과 아티팩트 해시를 읽어 대조했으며 원본 80개 파일은 바뀌지 않았습니다.

| 저장된 평가 결과 | Top-1 정답률 | Top-3 정답률 | 탐지율 | 오탐 박스 |
|---|---:|---:|---:|---:|
| baseline | 29.8% (14/47) | 40.4% (19/47) | 95.7% (45/47) | 259 |
| v2 | 31.9% (15/47) | 53.2% (25/47) | 95.7% (45/47) | 100 |
| v3 | 29.8% (14/47) | 55.3% (26/47) | 93.6% (44/47) | 99 |
| v4 | 46.8% (22/47) | 59.6% (28/47) | 95.7% (45/47) | 84 |

위 수치는 과거 JSON의 저장된 예측을 재집계한 값입니다. MLflow 도입으로 모델 성능이 올라갔다는 의미가 아닙니다. 33장의 같은 정답 내용을 사용한 비교이며, 평가와 체크포인트의 정확한 연결은 네 run 모두 미확정입니다.

확인 자료: [전체 run CSV](outputs/runs_export.csv), [최초 등록 결과](outputs/backfill_receipt.json), [재실행 결과](outputs/repeat_receipt.json), [원본 해시·기록 시점](outputs/records.json), [체크포인트 목록](outputs/checkpoint_inventory.json).

기본 검증 14건(합성 반례와 실제 평가 JSON 4개 대조), MLflow 합성 통합검사 7항목을 통과했습니다. UI 화면 검증과 모델 재학습은 수행하지 않았습니다.

## 무엇을 비교하나

- 평가 기록: baseline, v2, v3, v4의 저장된 예측 JSON 4개. 동일한 정답 박스 47개를 포함한 사진 33장의 결과입니다.
- 학습 기록: work_dir 압축파일 3개에서 설정과 학습 단계별 지표를 읽습니다. 원래 압축파일 이름을 유지해 `archive_v6`, `archive_v3`, `archive_v4`로 등록합니다.
- 체크포인트: 파일의 크기와 SHA-256만 확인합니다. 같은 바이트의 사본은 하나의 해시로 묶으며, 모델 레지스트리에는 등록하지 않습니다.

평가 JSON에는 사용한 체크포인트의 경로나 해시가 없습니다. 따라서 학습 기록과 평가 기록을 한 run으로 합치지 않습니다. 파일명이 비슷하거나 epoch가 같아도 연결이 입증된 것으로 취급하지 않습니다. 자세한 관찰은 [lineage.md](lineage.md)에 정리합니다.

## 실행

기존 학습 환경과 별도로 Python 3.12 환경을 준비합니다.

```powershell
uv venv .venv-mlflow --python 3.12
uv pip install --python .venv-mlflow/Scripts/python.exe -r experiments/mlflow/requirements.txt
```

[`sources.example.json`](sources.example.json)을 저장소 밖에 복사하고 각 입력 경로를 실제 로컬 경로로 바꿉니다. 꺾쇠로 표시한 경로는 자동으로 확장되지 않습니다. 원본 zip, 평가 JSON, 체크포인트, 현재 평가 이미지/라벨, 기존 평가 노트북이 필요합니다.

```powershell
.venv-mlflow/Scripts/python.exe -B experiments/mlflow/test_backfill.py
.venv-mlflow/Scripts/python.exe -B experiments/mlflow/backfill_runs.py --manifest "<LOCAL_MANIFEST>" --store-dir .mlflow-backfill --output-dir .mlflow-backfill/import-01
```

추적 DB와 산출물은 `.mlflow-backfill/`에만 저장합니다. 원본은 읽기 전용이며, 원본 설정 파일도 실행하지 않고 AST로 필요한 상수만 읽습니다. MLflow는 3.17.2로 고정했습니다. 실제 설치된 하위 의존성은 각 run의 `backfill_environment.json`에 기록합니다.

같은 코드·입력·규칙으로 다시 실행할 때는 새 출력 폴더를 지정합니다. 완료된 run의 설정과 전체 지표 이력이 일치하면 재사용합니다. 같은 식별자의 실패/진행 중 run이나 중복 run을 발견하면 멈춥니다. 기존 출력 폴더는 덮어쓰지 않습니다. 이 중복 방지는 한 번에 한 프로세스가 실행되는 로컬 작업을 대상으로 합니다.

백필 실행 코드가 달라지면 run 식별값도 달라집니다. 변경한 코드로 현재 7개 기록을 처음 처리할 때는 기존 run을 덮어쓰지 않고 새 run 7개를 만듭니다. README 문구만 수정하는 경우는 해당하지 않습니다.

재사용할 때는 필수 태그와 원본·계보·코드 아티팩트의 파일 목록 및 바이트 해시도 확인합니다. 새 run은 모든 기록과 입력 해시의 대조가 끝나야 완료로 전환합니다. 그 전에 오류가 나면 이번 실행에서 만든 run을 실패로 남기고 `failure.json`을 보존합니다. 기존 완료 run을 임의로 고쳐 재사용하지 않습니다.

UI가 필요할 때만 저장소 루트에서 실행하고, 사용 후 종료합니다. 아래 명령은 다른 컴퓨터에 공개하지 않는 로컬 조회용입니다.

```powershell
$env:MLFLOW_DISABLE_TELEMETRY = "true"
.venv-mlflow/Scripts/mlflow.exe ui --backend-store-uri sqlite:///.mlflow-backfill/tracking.db --host 127.0.0.1 --port 5011
```

이번 백필 스크립트는 UI 서버를 자동 실행하지 않습니다. 기본 run 목록의 시작 시간은 사후 등록 시각입니다. 원본 JSON의 `created_at`과 학습 로그 디렉터리의 날짜는 별도 근거로 보존하고, 알 수 없는 시간대는 추정하지 않습니다.

실제 실험 DB와 분리된 합성 통합검사도 제공합니다. 아직 존재하지 않는 폴더를 지정합니다. 정상 재실행, 아티팩트/태그 변조, 도중 입력 변경, 실패 run의 재등록 차단을 확인하며 실패 사례의 합성 DB도 남깁니다.

```powershell
.venv-mlflow/Scripts/python.exe -B experiments/mlflow/test_tracking_integration.py --workspace "<NEW_SYNTHETIC_TEST_FOLDER>"
```

## 지표의 뜻

평가 계산은 기존 `analysis/03_v4_evaluation_analysis.ipynb`의 `evaluate_v4_json`을 따릅니다. 정답 박스 순서대로 아직 사용하지 않은 예측 중 IoU가 가장 큰 박스를 고르고, IoU 0.5 이상이면 매칭합니다. 하나의 예측 박스는 한 번만 사용합니다.

- `top1_accuracy`, `top3_accuracy`, `detection_rate`: 사진 수가 아닌 **정답 박스 47개**가 분모입니다.
- `image_top1`, `image_top3`: 한 사진의 모든 정답을 맞춘 사진 비율입니다.
- `n_false_pos`: 어떤 정답에도 매칭되지 않은 예측 박스 수입니다.
- `reported_ms_per_image`: 과거 JSON에 적힌 추론 시간입니다. 이번 컴퓨터에서 측정한 시간이 아닙니다.
- 학습 로그의 `step`과 지표 이름은 원문대로 보존합니다. 학습 지표와 위 평가 지표는 서로 다른 데이터에서 나온 값이므로 섞어 비교하지 않습니다.

## 남기는 자료

각 run에는 원본 설정/로그 또는 예측 JSON, 원본 해시와 계보 설명, 백필 코드 사본, 설치 버전을 기록합니다. 백필 출력의 `local_provenance.json`에는 실제 로컬 경로를 남기므로 공개하지 않습니다. 이미지·라벨·가중치·전체 압축파일은 MLflow 아티팩트로 복사하지 않습니다.

저장소에는 실행 코드·테스트·이 문서와 경로를 제거한 비교 자료만 남깁니다. 추적 DB, 원본 산출물, 가상환경은 Git 제외 대상입니다.

## 확인 범위

현재 파일 해시는 입력이 사후 정리 도중 바뀌지 않았는지 확인하는 근거입니다. 과거 학습이나 평가 시점에도 같은 파일이었다는 증거는 아닙니다. 평가 결과와 특정 체크포인트의 연결, 과거 모델의 재현성, 새로운 이미지에 대한 성능은 이번 작업으로 검증하지 않습니다.

사용 API: [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), [MlflowClient](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).
