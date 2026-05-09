# 증강 기법 최종 정리 (pill_detection)

> **전제**: YOLOv8n ultralytics 기본값 적용 상태에서 추가할 기법만 정리
> **평가 조건**: 학습 = 스튜디오 스펙 / 테스트 = 실사용 스마트폰 사진

## 1. ultralytics 내장 추가 기법

| 파라미터 | 적용값 | 비고 |
|---|---|---|
| `degrees` | 10.0 | 공식 탐색 범위 0~45 내 |
| `shear` | 2.0 | 공식 탐색 범위 0~10 내 |
| `perspective` | 0.0005 | 공식 주석 "0–0.001 typical" 내 |

**출처**
- [default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)
- [tuner.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/tuner.py)

## 2. Albumentations 추가 기법

```python
import albumentations as A

custom_transforms = [
    A.OneOf([
        A.GaussianBlur(blur_limit=7, p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
        A.MedianBlur(blur_limit=7, p=1.0),
    ], p=0.3),

    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
    ], p=0.2),

    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

    A.RandomBrightnessContrast(
        brightness_limit=0.3, contrast_limit=0.3, p=0.5
    ),

    A.ImageCompression(
        quality_range=(50, 95), compression_type='jpeg', p=0.3
    ),
]
```

## 3. 기법별 값 & 출처

| 기법 | 값 | 출처 |
|---|---|---|
| GaussianBlur / MotionBlur / MedianBlur | `blur_limit=7`, OneOf p=0.3 | [ultralytics 공식 가이드](https://docs.ultralytics.com/guides/yolo-data-augmentation/) |
| GaussNoise | `var_limit=(10.0, 50.0)` | [ultralytics 공식 가이드](https://docs.ultralytics.com/guides/yolo-data-augmentation/) |
| ISONoise | `color_shift=(0.01, 0.05), intensity=(0.1, 0.5)` | [ultralytics 공식 가이드](https://docs.ultralytics.com/guides/yolo-data-augmentation/) |
| CLAHE | `clip_limit=4.0, tile_grid_size=(8, 8)` | [ultralytics 공식 가이드](https://docs.ultralytics.com/guides/yolo-data-augmentation/) |
| RandomBrightnessContrast | `brightness_limit=0.3, contrast_limit=0.3` | [ultralytics 공식 가이드](https://docs.ultralytics.com/guides/yolo-data-augmentation/) |
| ImageCompression | `quality_range=(50, 95), compression_type='jpeg', p=0.3` | [Albumentations face landmarks 튜토리얼](https://albumentations.ai/docs/examples/face-landmarks-tutorial/) |

## 4. 실사용 재현 매핑

| 실사용 조건 | 대응 기법 |
|---|---|
| 손떨림 / 초점 흐림 | GaussianBlur / MotionBlur / MedianBlur |
| 스마트폰 센서 노이즈 | GaussNoise / ISONoise |
| 실내외 조명 대비 | CLAHE / RandomBrightnessContrast |
| 블로그·SNS JPEG 아티팩트 | ImageCompression |

## 5. 검증 근거 논문

- ImageNet-C 벤치마크 (Noise/Blur/JPEG 증강의 도메인 일반화 기여): [arxiv:1903.12261](https://arxiv.org/abs/1903.12261)
- YOLOv4 (Mosaic / CutMix / HSV 조합): [arxiv:2004.10934](https://arxiv.org/abs/2004.10934)
- Copy-Paste (객체 탐지 증강): [arxiv:2012.07177](https://arxiv.org/abs/2012.07177)

