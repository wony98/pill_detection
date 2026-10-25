# GT v2와 정오표

2026-10-24 사용자 확인·승인에 따라 `프로스카정.txt`의 클래스 ID 두 곳만47→16으로 바꾼 별도 정답 버전을 만듭니다. 모델 실험 이름의 v2와 정답 버전 GT v2는 서로 다릅니다.

원본 `eval/test_labels/`, 원본 ZIP과 과거 평가 JSON4개는 보존합니다. `eval/label_versions/v2/labels/`에 라벨33개를 별도로 두며, 나머지32개 파일은 원본과 바이트 동일합니다. 바뀐1개도 좌표·공백·줄바꿈은 그대로입니다. `manifest.json`에 원본·새 라벨 해시, GT v1/v2 해시, 변경 근거와 사용자 승인 범위를 기록합니다.

## 동봉된 매핑 근거

`eval/reference/class_mapping.json`은 원본 매핑 CSV에서 식별자·클래스 번호·약품명 3개 열을 발췌한118행입니다. 원본 CSV의 SHA-256과 발췌 행의 해시를 포함합니다. 원본에서 비어 있던 식별자2개는 임의로 채우지 않았습니다.

검증은 이 동봉 파일의 형식·내용 해시·정정 대상 매핑과 v2 manifest의 연결을 검사합니다. `data/final_mapping_after_category_merge.csv`는 출처 식별용으로만 기록하며 실행할 때 존재할 필요가 없습니다. 검증을 선택 사항으로 바꾸거나 없는 파일을 건너뛰는 방식이 아닙니다.

자료를 옮길 때는 v2 폴더와 manifest의 `source_files`에 열거된 파일, `experiments/checkpoint_eval/` 및 기존 `experiments/mlflow/backfill_runs.py`를 함께 둡니다. 모델 가중치·추적 DB·원본 전체 data 폴더는 라벨 검증에 필요하지 않습니다. Python과 Pillow·PyYAML 실행 환경은 별도 전제입니다. 새 OS에서 의존성 설치까지 시험했다는 뜻은 아닙니다.

## 명시적 명령

저장소 루트에서 기존 환경을 사용합니다. 새 설치나 모델 실행이 없습니다.

```powershell
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/label_errata.py create-v2 --project-root .
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/label_errata.py verify-v2 --project-root .
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/label_errata.py rescore-gt-v2 --project-root . --output "<NEW_DIRECTORY_OUTSIDE_PROJECT>"
```

`create-v2`는 고정된 원본 라벨/GT 해시와 최종 매핑을 확인한 뒤 새 폴더만 생성합니다. 이미 있는 v2나 정오표를 덮어쓰지 않습니다. 실패한 폴더는 그대로 남겨 검사합니다.

`rescore-gt-v2`는 과거 예측 JSON을 메모리로 읽어 GT 버전을 명시적으로 적용하고, 별도 파생 파일·JSON·CSV·정오표를 만듭니다. `gt_version`, `historical_json_gt_version`, 원본 SHA-256과 GT 치환 여부를 남깁니다. 기존 `evaluation.py prepare`의 v1 검사와 묵시적 GT 치환 금지는 유지됩니다. 이 명령은 기존 `prepare`의 오류를 무시하거나 우회하는 옵션이 아닙니다.

0.05와0.25는 과거 노트북의 저장된 `top3_confidence[0] >= threshold` 필터를 사용합니다. 저장된 박스 그룹·점수·순서는 다시 계산하지 않습니다. B단계 새 모델의 원시 출력에서 후처리하는 것과 동일하다고 주장하지 않습니다.

## 해석과 경계

- 점수 차이는 정답 라벨 정정의 영향이며 모델 학습·추론·성능 개선 성과가 아닙니다.
- 과거 JSON을 만든 가중치·설정의 연결은 여전히 미확정입니다.
- 정오표의 수치는 로컬 검토용입니다. 별도 승인 전 공개 보고서·이력서·포트폴리오 수치를 바꾸지 않습니다.
- MLflow 기존 run 갱신, 푸시, 배포, PDF 재렌더, 모델 실행은 이 명령에 없습니다.

테스트:

```powershell
.\.venv-mlflow\Scripts\python.exe -B experiments/checkpoint_eval/test_label_errata.py
```

검사에는 승인되지 않은 좌표/파일 변경, 기존 출력 재사용, GT 버전 허위 표시 거부와 원래 노트북 필터 규칙 대조가 포함됩니다. 테스트에서 쓰는 모델 파일은 불투명한 합성 바이트이며 역직렬화하지 않습니다.
