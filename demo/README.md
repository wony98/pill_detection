# RTMDet 알약 탐지 데모 패키지 v4

3팀 HealthEat 프로젝트 — 지우님 웹 데모 인계용.

## 변경 사항 (v2 → v4)

- 모델 갱신: v2 (e14) → **v4 (e13)** 로 교체
- 성능 개선: Top-1 31.9% → **46.8%** (+14.9pp)
- 추론 사양 변경: imgsz 1280 → 1536 (config 자동 적용), **conf threshold 0.05 → 0.25 권장**
- 추론 시간: 132.1ms/장 (GPU) / 약 1초 (CPU)

## 패키지 구성

| 파일 | 크기 | 설명 |
|---|---|---|
| best_coco_bbox_mAP_epoch_13.pth | 24MB | v4 가중치 |
| rtmdet_tiny_pill.py | 35KB | 학습 config |
| rtmdet_tiny_synth_v4_e13_seed100_results.json | 73KB | 33장 평가 결과 |
| pill.yaml | 4KB | 118 클래스 매핑 (★ 클래스 ID의 진짜 출처) |
| inference_demo.py | 5.7KB | Top-3 그룹핑 추론 스크립트 |
| README.md | - | 이 문서 |

## 33장 평가 성능 (47 GT 단위)

| 측정 | v2 (e14) | **v4 (e13)** | Δ |
|---|---|---|---|
| Top-1 | 31.9% | **46.8%** | +14.9pp |
| Top-3 | 53.2% | **59.6%** | +6.4pp |
| Detection | 95.7% | **95.7%** | 0 |
| FP (conf 0.05) | 100 | 84 | -16 |
| FP (conf 0.25) | - | **10** | - |

### 환경별 Top-1

| 환경 | v2 | **v4** |
|---|---|---|
| palm | 30.0% | **50.0%** |
| dark | 0.0% | **50.0%** |
| paper | 28.6% | **42.9%** |
| desk | 20.0% | **40.0%** |

## 환경 세팅

```bash
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" -q
pip install "mmengine==0.10.4" "mmcv==2.1.0" "mmdet==3.3.0" -q
pip install pyyaml opencv-python pillow -q
```

★ **mmengine 0.10.5+ 사용 금지** — ASGD KeyError 발생. 반드시 0.10.4 명시 설치.

## 사용법

```bash
python inference_demo.py \
    --config rtmdet_tiny_pill.py \
    --checkpoint best_coco_bbox_mAP_epoch_13.pth \
    --names pill.yaml \
    --img <이미지 경로> \
    --device cuda:0 \
    --out result.png
```

## 추론 동작 방식 (5단계)

1. mmdet API로 RTMDet-tiny 모델 로드
2. config의 test_pipeline 자동 적용 (imgsz 1536)
3. conf >= 0.05 1차 필터링 후 박스 추출
4. IoU > 0.5 박스 그룹핑 (Top-3 클래스 추출)
5. 그룹별 대표 bbox + Top-3 클래스/confidence 출력

## 추론 권장 사양 (v4)

inference_demo.py 호출 시 또는 코드 내부에서 conf threshold 변경 권장:

```python
CONF_THRESHOLD = 0.25   # v2 default 0.05 → v4 권장 0.25
```

이유: v4는 confidence 분포가 v2보다 깔끔해 conf 0.25에서 Top-1/Top-3 손해 0, FP 84 → 10으로 88% 감소.

## 트러블슈팅

- **mmengine ASGD KeyError**: `pip install "mmengine==0.10.4" -q` 후 런타임 재시작.
- **numpy ABI 충돌**: `pip install "numpy<2" -q` 강제 다운그레이드.
- **한글 파일명 cv2.imread 실패 (cp949)**: `cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)` 패턴 사용.
- **mmcv NMS RuntimeError (sm_90)**: H100에서 발생. A100 또는 T4 사용 권장.

## 클래스 매핑 주의

★ **pill.yaml의 names가 진짜 출처**. 학습 시 사용한 매핑과 동일.
class_idx (0~117) ↔ dl_idx 변환은 `pill.yaml`을 통해서만 수행해야 함.

## 참고

- ensemble 시도: v2+v4, v3+v4, v2+v3+v4 모두 v4 단독 대비 후퇴 → 단일 모델 추론 권장
- 학습 곡선: val mAP@.5 epoch 7부터 천장, val mAP@.5:.95 epoch 12까지 미세 상승 (best_epoch=13)
- 분석 노트북: `03_v4_evaluation_analysis.ipynb` (별도 공유)

문의: 노승원