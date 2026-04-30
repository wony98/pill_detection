# RTMDet-tiny Pill Detection — 추론 데모

118종 알약 검출 모델. 박스별 Top-3 후보 출력.

---

## 이 폴더 구성

```
inference_demo.py        # 추론 데모 스크립트 (Top-3 그룹핑 포함)
rtmdet_tiny_pill.py      # mmdet config (★ 가중치 로드 필수)
pill.yaml                # class_idx 0~117 → 한글명 매핑
README.md                # 본 파일
```

> 가중치 파일(`.pth`)은 용량 문제로 레포에 포함되지 않습니다. 아래 참고.

---

## 가중치 위치 (별도)

**옵션 1 — 로컬 백업**
```
results/rtmdet_tiny_backup_seed100/best_coco_bbox_mAP_epoch_15.pth  (43MB)
```

**옵션 2 — Drive 백업** (팀 공유)
```
/MyDrive/pill_detection/results/_model_collection_seed100/  (예정)
```

추론 시 `--checkpoint` 인자로 경로 지정.

---

## 환경 세팅 (★ 함정 회피 박제값)

```bash
# 1. PyTorch 2.1 + NumPy<2
pip install torch==2.1.0 torchvision==0.16.0
pip install "numpy<2"

# 2. mmdet 스택 (mmengine 버전 박제 ★)
pip install -U openmim
mim install "mmengine==0.10.4"   # ← 0.10.5+는 PyTorch 2.1 ASGD 충돌
mim install "mmcv==2.1.0"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && pip install -e . && cd ..

# 3. 추가 의존성
pip install pyyaml
```

**CPU 추론용**이면 `torch==2.1.0+cpu` 빌드 사용:
```bash
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

---

## 사용법

```bash
# CPU 추론
python inference_demo.py \
    --img sample.png \
    --checkpoint /path/to/best_coco_bbox_mAP_epoch_15.pth

# GPU 추론
python inference_demo.py --img sample.png --device cuda:0 \
    --checkpoint /path/to/best_coco_bbox_mAP_epoch_15.pth

# 결과 JSON 저장
python inference_demo.py --img sample.png --out result.json \
    --checkpoint /path/to/best_coco_bbox_mAP_epoch_15.pth
```

> config / pill.yaml은 같은 폴더에서 실행하면 기본값으로 잡힙니다.

### 출력 예시

```
[INFO] device     : cpu
[INFO] classes    : 118

=== sample.png ===
검출 박스 2개

[박스 1] (120, 80) ~ (340, 260)
  Top-1: 마그밀정(수산화마그네슘) (idx=0, conf=0.873)
  Top-2: 인사돌플러스정 35mg/PTP (idx=56, conf=0.052)
  Top-3: 글리아타민연질캡슐 (idx=95, conf=0.031)

[박스 2] ...
```

---

## 추론 동작 방식

1. mmdet `inference_detector`로 이미지 추론 (RTMDet은 항상 300개 박스 고정 출력)
2. `conf >= 0.05` 1차 컷
3. 같은 위치(`IoU > 0.5`)끼리 greedy 그룹핑
4. 그룹 내 class별 best conf만 남기고 상위 3개 = Top-3
5. 그룹 대표 bbox = 그룹 내 최고 conf 박스

웹에 띄울 때는 `top3_class_name` (한글명) 그대로 쓰면 됩니다.

---

## 성능 참고

**평가셋 33장 / 47 bbox 채점 결과 (seed=42 / seed=100 평균):**
- Top-1 정답률: 31.9%
- Top-3 정답률: 42.6%
- Detection rate: 88.3%

**CPU 추론 속도** (Colab 2 vCPU, HF Spaces 무료 환경 근사):
- 평균 1.03초 / 장
- median 941ms

---

## 클래스 매핑 주의사항

`pill.yaml`의 `names` 리스트가 학습용 class_idx의 진짜 출처입니다.

```yaml
names:
  0: 마그밀정(수산화마그네슘)
  1: 게보린정 300mg/PTP
  ...
  117: ...
```

CSV에서 `sorted(unique) + enumerate`로 추측해서 만들면 어긋납니다. **반드시 pill.yaml 사용.**

> 참고: Kaggle 제출 시에는 학습용 class_idx (0~117)가 아니라 **원본 dl_idx**를 `category_id`로 써야 합니다. 웹 데모는 무관.

---

## 트러블슈팅

**`KeyError: 'ASGD is already registered in optimizer'`**
- 원인: mmengine 0.10.5+ + PyTorch 2.1 충돌
- 처치: `pip install "mmengine==0.10.4" --force-reinstall --no-deps`

**한글 파일명 깨짐 (Windows)**
- Python `zipfile` 모듈 권장 (`unzip` 명령은 cp949 깨짐)
