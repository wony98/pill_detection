# RTMDet 알약 탐지 데모 패키지 v4

3팀 HealthEat 프로젝트.
> 당시 웹 데모는 baseline 후처리를 사용했고 v4 배포는 보류했습니다. 이번 재검증에서는 데모 코드나 배포 모델을 바꾸지 않았습니다.


## 변경 사항 (v2 → v4)

- 모델 갱신: v2 (e14) → v4 (저장 가중치 epoch 12)로 교체
- GT v2 저장 예측 비교: v2 Top-1 17/47(36.2%), v4 24/47(51.1%), 차이+14.9pp. 라벨 정정은 모델 성능 개선과 구분합니다.
- 추론 사양 변경: imgsz 1280 → 1536 (config 자동 적용), conf threshold 0.05 → 0.25 권장
- 추론 시간: 132.1ms/장 (GPU) / 약 1초 (CPU)

## 패키지 구성

| 파일 | 크기 | 설명 |
|---|---|---|
| best_coco_bbox_mAP_epoch_12.pth | 44MB | v4 가중치 (Drive 백업, 깃 미포함) |
| rtmdet_tiny_pill.py | 39KB | 학습 config (v4 정본, max_epochs=13/imgsz=1536/Albu/EMA 적용) |
| rtmdet_tiny_synth_v4_e13_seed100_results.json | 73KB | 33장 평가 결과 |
| pill.yaml | 4KB | 118 클래스 매핑 (클래스 ID 기준 출처) |
| inference_demo.py | 5.7KB | Top-3 그룹핑 추론 스크립트 |
| README.md | - | 이 문서 |

## 33장 저장 예측 재채점 (47 GT, GT v2)

정답 라벨 오기1건을 고친 뒤 과거 v4의 Top-1은22→24/47, Top-3는28→30/47로 정정됐습니다. 아래 두 모델 모두 같은 GT v2 기준입니다([정오표](../../docs/errata_label_v2_20261024/README.md)).

| 측정 | v2 (e14) | v4 (평가 JSON e13) | Δ |
|---|---|---|---|
| Top-1 | 36.2% (17/47) | 51.1% (24/47) | +14.9pp |
| Top-3 | 57.4% (27/47) | 63.8% (30/47) | +6.4pp |
| Detection | 95.7% | 95.7% | 0 |
| FP (conf 0.05) | 100 | 84 | -16 |
| FP (저장 점수 0.25 사후 필터) | 19 | 10 | -9 |

평가 JSON 파일명은 `e13`이고 보관 가중치는 epoch12입니다. 과거 평가 JSON에 체크포인트 경로나 해시가 없어 둘의 연결은 확인되지 않았습니다([계보 기록](../../experiments/mlflow/lineage.md)).

해당 저장 가중치의 별도33장 실행에서는 점수기준0.25에서 Top-1 25/47·Top-3 28/47·미매칭10개를 확인했습니다. 입력·설정·가중치·결과를 연결한 [새 실행 근거](../../experiments/forward_eval_20261025/README.md)이며, 위 과거 정정값과 동일 조건의 모델 비교로 합치지 않습니다.

### 환경별 Top-1 (당시 GT v1 분석)

| 환경 | v2 | v4 |
|---|---|---|
| palm | 30.0% | 50.0% |
| dark | 0.0% | 50.0% |
| paper | 28.6% | 42.9% |
| desk | 20.0% | 40.0% |

## 환경 세팅

```bash
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" -q
pip install "mmengine==0.10.4" "mmcv==2.1.0" "mmdet==3.3.0" -q
pip install pyyaml opencv-python pillow -q
```

mmengine 0.10.5 이상은 ASGD KeyError가 발생하므로 0.10.4를 명시 설치합니다.

## 사용법

```bash
python inference_demo.py \
    --config rtmdet_tiny_pill.py \
    --checkpoint best_coco_bbox_mAP_epoch_12.pth \
    --names pill.yaml \
    --img <이미지 경로> \
    --device cuda:0 \
    --out result.png
```

## 추론 동작 방식 (5단계)

1. mmdet API로 RTMDet-tiny 모델 로드
2. config의 test_pipeline 자동 적용 (imgsz 1536)
3. conf >= 0.25 1차 필터링 후 박스 추출 (`inference_demo.py`의 기본값)
4. IoU > 0.5 박스 그룹핑 (Top-3 클래스 추출)
5. 그룹별 대표 bbox + Top-3 클래스/confidence 출력

## 추론 권장 사양 (v4)

inference_demo.py 호출 시 또는 코드 내부에서 conf threshold 변경 권장:

```python
CONF_THRESHOLD = 0.25   # v2 default 0.05 → v4 권장 0.25
```

과거 저장 예측의 사후 필터에서는0.05와0.25 모두 GT v2 Top-1 24/47·Top-3 30/47을 유지하며 미매칭 박스가84→10으로 줄었습니다. 별도 새 원시 출력에서는 Top-1 25/47은 같지만 Top-3가29→28/47로 달라졌습니다. 이 데모는 점수를 반올림하는 자체 후처리를 사용하므로, 반올림 전 출력을 보존한 B단계 검증 도구와도 구분합니다.

## 트러블슈팅

- **mmengine ASGD KeyError**: `pip install "mmengine==0.10.4" -q` 후 런타임 재시작.
- **numpy ABI 충돌**: `pip install "numpy<2" -q` 강제 다운그레이드.
- **한글 파일명 cv2.imread 실패 (cp949)**: `cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)` 패턴 사용.
- **mmcv NMS RuntimeError (sm_90)**: H100에서 발생. A100 또는 T4 사용 권장.

## 클래스 매핑 주의

pill.yaml의 names가 기준 출처입니다. 학습 시 사용한 매핑과 동일합니다.
class_idx (0~117) ↔ dl_idx 변환은 `pill.yaml`을 통해서만 수행해야 함.

## 참고

- 당시 GT v1 앙상블 분석에서는 v4 단독을 선택했습니다. 이 결과를 GT v2로 다시 채점하지는 않았습니다.
- 학습 곡선: val mAP@.5 epoch 7부터 천장, val mAP@.5:.95 epoch 12까지 미세 상승 (저장된 best 체크포인트 epoch 12)
- 분석 노트북: `03_v4_evaluation_analysis.ipynb` (별도 공유)

문의: wony98
