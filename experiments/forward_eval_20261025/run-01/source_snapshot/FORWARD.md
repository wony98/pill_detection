# HealthEat B단계: 새 추론과 라벨 v2 평가

기존 A단계와 GT v2 도구는 변경하지 않습니다. `forward.py`는 원본을 검증해 새 작업 폴더를 만들고, `inference_runner.py`는 그 입력 사본으로만 추론합니다. 모델 재학습과 원본 MLflow 저장소 쓰기는 없습니다.

## 실행 순서

1. 기존 Windows `.venv-mlflow`에서 아래 `prepare` 실행.
2. 추론용 Python 환경에서 입력 사본의 러너로 `--smoke` 1장 실행. 실패 시 전체 실행하지 않습니다.
3. 동일 입력·환경으로 전체 33장 실행. 새 출력 폴더만 사용합니다.
4. Windows 환경에서 `score` 실행. 정답은 여기에서만 합쳐집니다.
5. `record`로 기존 `.mlflow-backfill`과 분리된 새 SQLite 저장소에 기록하고, 지표·태그·파일 해시를 읽어서 대조합니다.

```powershell
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/forward.py prepare --project-root . --workspace <NEW_WORKSPACE>
```

```bash
# 추론 환경 내부. CPU 또는 cuda:0를 명시합니다.
python -B <WORKSPACE>/inputs/inference_runner.py --inputs <WORKSPACE>/inputs --output <WORKSPACE>/smoke --device cpu --smoke
# 1장 실행 성공과 출력 확인 후에만:
python -B <WORKSPACE>/inputs/inference_runner.py --inputs <WORKSPACE>/inputs --output <WORKSPACE>/full --device cpu
```

```powershell
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/forward.py score --workspace <WORKSPACE>
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/forward.py record --workspace <WORKSPACE> --tracking <NEW_SEPARATE_TRACKING_DIRECTORY>
```

Colab을 선택하면 `inputs.zip`만 전송합니다. 라벨·정답·과거 예측·개인 로컬 경로는 포함하지 않습니다. 출력 `smoke/`와 `full/`을 원래 작업 폴더로 가져온 뒤 채점합니다. 기존 데이터와 코드를 새 버전으로 고친 경우에는 이전 묶음을 덮어쓰지 않고 `prepare`부터 새 작업 폴더를 만듭니다.

## 고정한 조건

- v4 원본 가중치와 config, 118개 클래스, Resize/Pad 1536 유지.
- 추론 시 `LoadAnnotations` 제거, annotation 경로 비우기, 사전학습 원격 초기화 해제. 원본 config는 보존하고 적용 설정을 별도 저장.
- 저장하는 raw 출력은 **MMDetection의 모델 NMS 이후 `pred_instances`**입니다. 모델 NMS 이전의 모든 후보를 저장했다고 표현하지 않습니다.
- raw float32를 반올림하지 않고 저장. `.05`와 `.25`는 raw 출력에서 각각 필터링합니다.
- `rtmdet_tiny_synth_train_v4.ipynb`에서 `iou_xyxy`, `extract_top3` 두 함수만 추출해 사용합니다. 그룹핑 IoU `.5`, Top-3, NumPy 정렬 규칙을 그대로 유지합니다. 환경과 소스 해시를 함께 남깁니다.
- 채점은 기존 `backfill_runs.score_evaluation`의 GT 순서·최대 IoU·예측 한 번 사용 규칙을 재사용합니다.
- 원본 33장은 모델/임계값 선택에 이미 사용한 평가셋입니다. 새 사진 일반화나 의료 사용 가능성의 근거가 아닙니다.
- GT v2 정정에 의한 수치 변화와 모델 성능 변화를 구분합니다. 과거 JSON의 체크포인트 연결은 미확정이며, 과거 Top-3 사후 필터와 새 raw 후처리를 동일한 통제 실험으로 표시하지 않습니다.

## 환경 후보와 현재 상태

2026-10-25 Ubuntu/Colab CLI 0.6.0의 인증과 활성 세션 없음까지 확인했습니다. 공식 Linux CPU `mmcv==2.1.0`의 Python 3.11 휠 존재도 확인했습니다. 설치·연산자 호환·모델 실행이 검증됐다는 뜻은 아닙니다.

후보 조합은 Python 3.11, PyTorch 2.1.0, torchvision 0.16.0, NumPy 1.26.4, mmengine 0.10.4, mmcv 2.1.0, mmdet 3.3.0입니다. 원래 환경을 덮어쓰지 않는 독립 환경에 설치하고 라이브러리/연산자 확인부터 진행합니다. 유료 Colab 유닛 사용은 별도 선택을 기다립니다.

## 모델 없는 검사

```powershell
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/test_forward.py
```

raw shape·유한값·클래스 범위, float32 보존, 임계값별 독립 후처리, 동일 점수와 중복 클래스, config 정답 입력 제거, 입력 경로 이탈·추가 파일·결과 변조·실패 기록·범위 불일치 거부, 합성 데이터 채점 경로를 검사합니다. 합성 검사를 실제 모델 실행이나 독립 교차검토로 보고하지 않습니다.
