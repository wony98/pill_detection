"""
inference.py — HealthEat 알약 검출 진입점

루트에서 한 줄로 추론.

사용법 (기본 = 현재 배포 모델 baseline):
    python inference.py --img sample.png

v4 모델 사용 (성능 검증용, 웹 데모와는 별개):
    python inference.py --img sample.png --model v4

상세 옵션:
    python inference.py --help

모델 선택:
    baseline : 현재 배포 모델 (웹 데모 후처리 호환성 검증 완료)
    v4       : 분석 단계 우위 모델 (Top-1 46.8%, baseline 대비 +17.0pp)
               웹 데모 후처리 호환성 미검증 — 추후 적용 예정

실제 추론 로직은 demo/{baseline,v4}/inference_demo.py에 있고, 이 파일은 진입점 래퍼입니다.
모델 가중치 다운로드는 README.md 'Quick Start' 섹션 참조.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 1. argparse — 모델 선택 + 입력 이미지
parser = argparse.ArgumentParser(
    description="HealthEat 알약 검출 (118 클래스, RTMDet-tiny 기반)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model", choices=["baseline", "v4"], default="baseline",
    help="추론 모델 선택. baseline = 현재 배포, v4 = 분석 우위 (추후 적용)"
)
parser.add_argument("--img", required=True, help="입력 이미지 경로")
parser.add_argument("--device", default="cpu", help="cpu 또는 cuda:0")
parser.add_argument("--out", default=None, help="결과 JSON 저장 경로 (옵션)")

# 모델별 default 경로 (인자로 직접 override 가능)
parser.add_argument("--config", default=None, help="(고급) mmdetection config 경로")
parser.add_argument("--checkpoint", default=None, help="(고급) 가중치 .pth 경로")
parser.add_argument("--names", default=None, help="(고급) 클래스명 yaml 경로")

args = parser.parse_args()

# 2. 모델별 default 경로 결정
model_dir = ROOT / "demo" / args.model

if not model_dir.exists():
    print(f"[ERROR] {args.model} 모델 폴더 없음: {model_dir}")
    sys.exit(1)

config_path = Path(args.config) if args.config else model_dir / "rtmdet_tiny_pill.py"
names_path = Path(args.names) if args.names else model_dir / "pill.yaml"

# checkpoint default 파일명 — baseline/v4 각각 다를 수 있음
# 일단 폴더 내 .pth 자동 탐색 (인자로 명시 안 했을 때)
if args.checkpoint:
    ckpt_path = Path(args.checkpoint)
else:
    pth_candidates = sorted(model_dir.glob("*.pth"))
    if not pth_candidates:
        print(f"[ERROR] {model_dir}에 .pth 가중치 파일 없음")
        print(f"        README.md의 'Quick Start' 섹션에서 가중치 다운로드 안내 확인.")
        sys.exit(1)
    ckpt_path = pth_candidates[0]
    if len(pth_candidates) > 1:
        print(f"[WARN] .pth 여러 개 발견, 첫 번째 사용: {ckpt_path.name}")

# 3. 파일 존재 검증
for label, p in [("config", config_path), ("checkpoint", ckpt_path), ("names", names_path)]:
    if not p.exists():
        print(f"[ERROR] {label} 없음: {p}")
        sys.exit(1)

# 4. demo/{model}/inference_demo.py 함수 import
sys.path.insert(0, str(model_dir))
from inference_demo import load_class_names, predict
from mmdet.apis import init_detector

print(f"[INFO] model      : {args.model}")
print(f"[INFO] device     : {args.device}")
print(f"[INFO] config     : {config_path}")
print(f"[INFO] checkpoint : {ckpt_path}")

class_names = load_class_names(str(names_path))
print(f"[INFO] classes    : {len(class_names)}")

model = init_detector(str(config_path), str(ckpt_path), device=args.device)
groups = predict(model, args.img, class_names)

# 5. 결과 출력
print(f"\n=== {Path(args.img).name} ===")
print(f"검출 박스 {len(groups)}개")
for i, g in enumerate(groups, 1):
    x1, y1, x2, y2 = g["bbox"]
    print(f"\n[박스 {i}] ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f})")
    for rank, (idx, name, conf) in enumerate(
        zip(g["top3_class_idx"], g["top3_class_name"], g["top3_confidence"]), 1
    ):
        print(f"  Top-{rank}: {name} (idx={idx}, conf={conf:.3f})")

# 6. JSON 저장 (옵션)
if args.out:
    import json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "image": str(args.img),
            "model": args.model,
            "num_groups": len(groups),
            "groups": groups,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 결과 저장: {args.out}")
