"""
RTMDet-tiny Pill Detection — 추론 데모

학습된 RTMDet-tiny (118 클래스) 가중치로 단일 이미지 추론.
박스별 Top-3 그룹핑까지 포함.

환경:
    torch==2.1.0
    numpy<2
    mmengine==0.10.4   # ★ 0.10.5+ PyTorch 2.1 ASGD 충돌
    mmcv==2.1.0
    mmdet==3.3.0

사용법:
    python inference_demo.py --img sample.png
    python inference_demo.py --img sample.png --device cuda:0 --out result.json
"""
import argparse
import json
from pathlib import Path

import yaml
from mmdet.apis import init_detector, inference_detector


# Top-3 추출 파라미터 (모델 비교 노트북 동일값)
CONF_THRESH = 0.25      # 1차 conf 컷 (RTMDet은 항상 300개 박스 고정 출력)
IOU_THRESH = 0.5        # 그룹핑 IoU
TOPK = 3


def load_class_names(yaml_path):
    """pill.yaml의 names 리스트 로드.
    
    학습용 class_idx 0~117 → 한글 알약명.
    pill.yaml names가 진짜 출처 (CSV sorted 추측은 함정).
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']  # {0: '마그밀정...', 1: '게보린정...', ...}


def iou(box_a, box_b):
    """xyxy 두 박스의 IoU."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_top3(bboxes, scores, labels,
                 conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH, topk=TOPK):
    """RTMDet 출력에서 박스별 Top-3 그룹핑.
    
    1. conf >= conf_thresh 박스만 남김
    2. 같은 위치(IoU > iou_thresh)끼리 그룹핑 (greedy, conf 내림차순)
    3. 그룹 내 class별 best conf만 남기고 상위 topk = Top-3
    4. 그룹 대표 bbox = 그룹 내 최고 conf 박스
    
    Returns:
        list of {bbox, top3_class_idx, top3_confidence}
    """
    keep = scores >= conf_thresh
    bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
    
    if len(bboxes) == 0:
        return []
    
    # conf 내림차순
    order = scores.argsort()[::-1]
    bboxes, scores, labels = bboxes[order], scores[order], labels[order]
    
    n = len(bboxes)
    used = [False] * n
    groups = []
    
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        member_idxs = [i]
        for j in range(i + 1, n):
            if used[j]:
                continue
            if iou(bboxes[i], bboxes[j]) > iou_thresh:
                used[j] = True
                member_idxs.append(j)
        
        # 그룹 내 class별 best conf
        cls_best = {}
        for k in member_idxs:
            c, s = int(labels[k]), float(scores[k])
            if c not in cls_best or s > cls_best[c]:
                cls_best[c] = s
        
        ranked = sorted(cls_best.items(), key=lambda x: -x[1])[:topk]
        groups.append({
            'bbox': bboxes[i].tolist(),  # 대표 = 최고 conf 박스
            'top3_class_idx': [c for c, _ in ranked],
            'top3_confidence': [round(s, 4) for _, s in ranked],
        })
    
    return groups


def predict(model, img_path, class_names):
    """단일 이미지 추론 → 박스별 Top-3."""
    result = inference_detector(model, str(img_path))
    pred = result.pred_instances
    
    bboxes = pred.bboxes.cpu().numpy()  # (N, 4) xyxy
    scores = pred.scores.cpu().numpy()  # (N,)
    labels = pred.labels.cpu().numpy()  # (N,) class_idx 0~117
    
    groups = extract_top3(bboxes, scores, labels)
    
    # 한글명 붙이기
    for g in groups:
        g['top3_class_name'] = [class_names[c] for c in g['top3_class_idx']]
    
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='rtmdet_tiny_pill.py')
    parser.add_argument('--checkpoint', default='best_coco_bbox_mAP_epoch_15.pth')
    parser.add_argument('--names', default='pill.yaml')
    parser.add_argument('--img', required=True, help='입력 이미지 경로')
    parser.add_argument('--device', default='cpu', help='cpu 또는 cuda:0')
    parser.add_argument('--out', default=None, help='결과 JSON 저장 경로 (옵션)')
    args = parser.parse_args()
    
    print(f'[INFO] device     : {args.device}')
    print(f'[INFO] config     : {args.config}')
    print(f'[INFO] checkpoint : {args.checkpoint}')
    
    class_names = load_class_names(args.names)
    print(f'[INFO] classes    : {len(class_names)}')
    
    model = init_detector(args.config, args.checkpoint, device=args.device)
    groups = predict(model, args.img, class_names)
    
    print(f'\n=== {Path(args.img).name} ===')
    print(f'검출 박스 {len(groups)}개')
    for i, g in enumerate(groups, 1):
        x1, y1, x2, y2 = g['bbox']
        print(f'\n[박스 {i}] ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f})')
        for rank, (idx, name, conf) in enumerate(
            zip(g['top3_class_idx'], g['top3_class_name'], g['top3_confidence']), 1
        ):
            print(f'  Top-{rank}: {name} (idx={idx}, conf={conf:.3f})')
    
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({
                'image': str(args.img),
                'num_groups': len(groups),
                'groups': groups,
            }, f, ensure_ascii=False, indent=2)
        print(f'\n[INFO] 결과 저장: {args.out}')


if __name__ == '__main__':
    main()
