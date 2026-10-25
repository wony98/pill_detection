# HealthEat: 저장 가중치의 별도 실행·평가 근거

기존 v4 가중치로1장 시험 후33장을 새로 추론하고, 정답 라벨 v2로 채점한 기록입니다. **검토용 근거 묶음이며 독립 실행 가능한 패키지는 아닙니다.** 모델 파일·이미지 중복본·개인 경로 목록·MLflow DB는 포함하지 않았습니다.

## 결과를 읽는 기준

과거 저장 예측은 정답 정정 후 Top-1 24/47이고, 저장 가중치의 별도 실행에서는25/47을 확인했습니다. 두 기록은 같은 결과가 재현됐다는 뜻이 아닙니다. 과거 JSON을 만든 체크포인트의 연결은 여전히 미확정이고, 저장된 Top-3를 사후 필터링한 방식과 새 원시 예측을 후처리한 방식도 다릅니다.

| 별도 새 실행의 점수 기준 | 위치·약품명 동시 정답 | Top-3 내 정답 | 위치 매칭 | 미매칭 예측 박스 |
| --- | --- | --- | --- | --- |
| 0.05 | 25/47 | 29/47 | 45/47 | 105 |
| 0.25 | 25/47 | 28/47 | 45/47 | 10 |

과거 정정 기록의 Top-3 30/47과 위 새 실행의 Top-1 25/47을 한 실행의 지표로 합치지 않습니다. 33장은 모델/임계값 선택에 이미 쓴 평가셋입니다. 새 데이터 일반화·의료 사용 가능성·통제된 모델 성능 개선을 입증하지 않습니다.

## 파일 구성과 원본 연결

- `run-01/inputs/inputs.json`: 당시 모델·config·클래스 매핑·이미지·추론 코드의 해시. 원본 입력 manifest를 변경하지 않았습니다.
- `run-01/smoke/`, `run-01/full/`: 1장/33장 실행 receipt, 환경, 적용 config, raw 출력. raw는 **모델 NMS 이후 `pred_instances`, Top-3 그룹핑 이전** 값입니다.
- `run-01/evaluation/`: 두 기준의 예측·지표·과거 기록과의 설명적 비교·채점 receipt.
- `run-01/source_snapshot/`: 실행 당시 코드·설명의 고정본. 그 안의 이전 문서나 실패 상태는 작성 당시 기록이며, 현재 절차는 [FORWARD.md](../checkpoint_eval/FORWARD.md)를 참고하세요.
- `tracking-02-ubuntu/receipt.json`, `recording_environment.json`: 새 MLflow run의 기록 receipt와 기록 환경. 코드·입력은 추론 당시 버전과 기록 당시 버전을 구분합니다.
- `tracking-01/failure.json`: Windows SQLAlchemy DLL 정책 차단으로 run 생성 전에 실패한 기록. 사후에 실패를 기록한 receipt임을 명시했습니다. 원본 실패를 성공으로 고치지 않았습니다.
- `verification-completed.json`: 자동 검증이 끝난 시점의 고정 결과. 이후 교차검토나 문서 보완으로 원문을 덮어쓰지 않았습니다.
- `evidence-manifest.json`: 공개 파일과 논리적 원본 이름·원본 SHA-256·공개본 SHA-256 대조표. `original-workspace`와 `original-errata`는 논리 이름이며 실제 개인 경로가 아닙니다.
- [logs/README.md](logs/README.md): 공개 로그 사본과 치환 규칙·횟수·전후 해시. 경로만 치환하고 JSON 결과 줄은 바꾸지 않았습니다.

실제 적용 추론 경로는 `full/effective_config.py`의 `test_dataloader.dataset.pipeline`과 `full/run.json`의 `overrides`를 확인합니다. 설정 전체에 남은 원격 URL·정답 로더는 이번 추론에서 사용하지 않은 다른 설정 구간입니다.

현재 구현은 [checkpoint_eval](../checkpoint_eval/)에 있습니다. 당시 `source_snapshot/forward.py`와 현재 `forward.py`는 기록 초기화 실패 처리·명시적인 원본 코드 지정 등 후속 보완 때문에 해시가 다릅니다. MLflow receipt의 `origin_forward.py.txt`는 전자, `recording_forward.py.txt`는 후자에 연결됩니다. 이들을 포함한27개 기록 아티팩트 전체를 파일 또는 저장소 소스와 해시로 대조할 수 있습니다.

## 의도적으로 제외한 파일

각 파일의 당시 해시는 [excluded-files.json](excluded-files.json)에 있습니다. 폴더는 구성 파일별 해시로 기록했습니다.

| 제외 대상 | 이유·대응 |
| --- | --- |
| `run-01/inputs/model.pth` | 대용량 체크포인트. 원본 `demo/v4/v4.pth`와 동일 해시이며 저장소 가중치 제외 정책 유지 |
| `run-01/inputs/pill.yaml` | 기존 `demo/v4/pill.yaml`과 동일한 중복본. 과거 로컬 경로가 포함돼 새 묶음에는 복제하지 않음 |
| `run-01/inputs/images/` | 기존 `eval/test_images/` 33장과 바이트 동일한 중복본 |
| `run-01/inputs.zip` | 입력을 다시 묶은 중복 압축 파일 |
| `run-01/local_sources.json` | 원본 작업 폴더의 개인 절대경로 |
| `tracking-02-ubuntu/tracking/` | 로컬 SQLite DB·파일 URI 및 중복 아티팩트. DB는 저장소 제외 정책 유지 |
| `tracking-02-ubuntu/recorded_files/` | 같은27개 아티팩트의 중복 로컬 사본 |
| `smoke/full`의 `partial_predictions.json` | 실행 중간 산출물. 완료 raw 결과만 동봉 |
| `tmp/`, 가상환경, 캐시 | 실행 환경·임시 파일이므로 근거 묶음에서 제외 |

## 재조회와 새 실행은 별도 절차

`verify`는 원본 `local_sources.json`과 보호 입력, `verify-record`는 원본 DB와 파일 URI가 있어야 합니다. **이 공개 폴더만 다운로드해 두 명령을 바로 실행할 수 없습니다.** 원본 작업 폴더 구조·누락 파일·실행 환경이 준비된 경우에만 [FORWARD.md](../checkpoint_eval/FORWARD.md)의 `verify → verify-record` 순서로 재조회합니다. 원본 manifest·receipt의 상대 참조도 그 원래 구조를 기준으로 합니다.

새 평가를 하려면 원본 입력을 확인하고 새 작업 폴더에서 `prepare → 1장 시험 → 33장 추론 → score → record`를 수행해야 합니다. 현재 기록을 새 실행으로 재등록하거나 검증을 통과시키기 위해 해시를 바꾸지 않습니다. 범용 복원 도구나 새 OS 전체 재현 검증은 이번 묶음에 포함하지 않았습니다.

MLflow 저장·조회는 Windows DLL 정책 차단 후 별도 Ubuntu 환경에서 검증했습니다. 원래 Windows 보안 설정은 변경하지 않았습니다. 기존 추론을 반복하거나 모델을 재학습하지 않고 완료된 결과를 기록했습니다.
