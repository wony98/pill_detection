# GT v2 정오표

예측이나 모델을 바꾸지 않고 정답 라벨을 정정한 재채점 기록입니다. 원본 라벨과 기존 평가 JSON은 유지했습니다.

- 변경: `프로스카정.txt`의 두 박스 클래스 ID를47에서16으로 수정. 좌표·공백·줄바꿈과 다른32개 라벨은 그대로입니다.
- 결과: 과거 v4 저장 예측의 위치·약품명 동시 정답은22/47에서24/47로, Top-3는28/47에서30/47로 정정됐습니다. 모델 성능 향상을 뜻하지 않습니다.
- 파일: `errata.json`은 매핑 근거 동봉을 보완한 뒤 현재 라벨 manifest·코드로 생성해 검토한 정오표와 바이트가 같습니다. 1차 정오표와 비교해 출처 해시만 달라졌으며16행 수치·파생 파일 목록·`errata.csv`·`정오표.md`는 동일합니다.
- GT v2와 매핑 근거: [라벨 v2](../../eval/label_versions/v2/), [매핑 발췌](../../eval/reference/class_mapping.json).

## 파생 JSON 재생성

정오표3개만 동봉하고, 같은 저장 예측을 반복하는 파생 JSON16개는 제외했습니다. 원본 평가 JSON·이미지·라벨·매핑 등 `eval/label_versions/v2/manifest.json`의 입력 파일을 갖춘 원본 작업 환경에서 다음 명령으로 새 출력 폴더에 다시 만들 수 있습니다. 원본 체크포인트를 실행하거나 다시 학습하지 않습니다.

```text
python -B experiments/checkpoint_eval/label_errata.py verify-v2 --project-root <PROJECT_ROOT>
python -B experiments/checkpoint_eval/label_errata.py rescore-gt-v2 --project-root <PROJECT_ROOT> --output <NEW_DIRECTORY_OUTSIDE_PROJECT>
```

Python·Pillow·PyYAML은 별도 필요합니다. 새 체크아웃의 줄바꿈·입력 해시가 기록과 다르면 검증은 거부되며, 해시를 임의로 다시 쓰지 않아야 합니다. 저장소 묶음만으로 새 OS에서 재현 검증을 마쳤다는 의미는 아닙니다.

0.25 결과는 이미 저장된 Top-3 그룹의 첫 점수로 사후 필터링한 값입니다. [B단계의 별도 새 추론](../../experiments/forward_eval_20261025/)에서 원시 출력에 기준을 각각 적용한 결과와 구분해 읽어야 합니다.
