"""
ensemble_utils.py — RTMDet N-way ensemble 추론 + 채점 유틸.

세션 48에서 검증된 방식 A (IoU 매칭 후 conf 평균) 일반화.

사용 예시:
    >>> import json
    >>> from ensemble_utils import merge_predictions_nway, evaluate_ensemble_nway
    >>>
    >>> # 모델 결과 JSON 로드
    >>> with open('v2_results.json') as f: v2 = json.load(f)
    >>> with open('v3_results.json') as f: v3 = json.load(f)
    >>> with open('v4_results.json') as f: v4 = json.load(f)
    >>>
    >>> # 4-way ensemble 채점
    >>> metrics = evaluate_ensemble_nway([v2, v3, v4], conf_thr=0.2)
    >>> print(f"Top-1: {metrics['top1']:.1%}")
    >>>
    >>> # 단일 이미지 추론 (데모용)
    >>> v2_preds = [...]  # v2 모델의 prediction 리스트
    >>> v3_preds = [...]
    >>> merged = merge_predictions_nway([v2_preds, v3_preds], conf_thr=0.2)
"""
import unicodedata
from typing import List, Dict, Any


def nfc(s: str) -> str:
    """Unicode NFC 정규화 (한글 클래스명 비교용)."""
    return unicodedata.normalize('NFC', s) if isinstance(s, str) else s


def iou_xyxy(b1, b2) -> float:
    """xyxy bbox IoU."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def merge_predictions_nway(
    pred_lists: List[List[Dict[str, Any]]],
    conf_thr: float = 0.2,
    iou_match: float = 0.5,
) -> List[Dict[str, Any]]:
    """N-way ensemble — IoU 매칭 후 conf 평균.

    Args:
        pred_lists: 모델별 prediction 리스트의 리스트.
            각 prediction은 {bbox: [x1,y1,x2,y2], top3_class_name: [str,str,str],
            top3_confidence: [float,float,float]} 형식.
        conf_thr: top1 confidence 하한 (이하 prediction은 사전 제거).
        iou_match: 모델 간 prediction 매칭 IoU 임계값.

    Returns:
        merged predictions — {bbox, top3_class_name, top3_confidence}.

    동작:
        1) 각 모델별 conf threshold 통과 prediction만 유지.
        2) 첫 모델을 base로 잡고, 다른 모델 prediction을 IoU로 그룹화.
        3) 매칭 그룹: bbox 평균 + top3 클래스 conf 합산 후 재선정.
        4) 매칭 안 된 unique prediction은 그대로 추가.

    검증: 세션 48 v2+v3 ensemble Top-1 36.2% (v2 단독 31.9% +4.3pp).
    """
    filtered = [[p for p in preds if p['top3_confidence'][0] >= conf_thr]
                for preds in pred_lists]

    if not filtered or not filtered[0]:
        merged = []
        for preds in filtered:
            merged.extend(preds)
        return merged

    n_models = len(filtered)
    used = [set() for _ in range(n_models)]
    merged = []

    for i, base_pred in enumerate(filtered[0]):
        group = [base_pred]
        used[0].add(i)
        for m_idx in range(1, n_models):
            best_iou, best_j = 0.0, -1
            for j, p in enumerate(filtered[m_idx]):
                if j in used[m_idx]:
                    continue
                iou = iou_xyxy(base_pred['bbox'], p['bbox'])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_match and best_j >= 0:
                group.append(filtered[m_idx][best_j])
                used[m_idx].add(best_j)

        # bbox 평균 + top3 클래스 conf 합산 → 상위 3 재선정
        avg_bbox = [sum(p['bbox'][k] for p in group) / len(group) for k in range(4)]
        conf_acc = {}
        for p in group:
            for cls, c in zip(p['top3_class_name'], p['top3_confidence']):
                cls_n = nfc(cls)
                conf_acc[cls_n] = conf_acc.get(cls_n, 0) + c
        top3_sorted = sorted(conf_acc.items(), key=lambda x: -x[1])[:3]
        merged.append({
            'bbox': avg_bbox,
            'top3_class_name': [c for c, _ in top3_sorted],
            'top3_confidence': [c / len(group) for _, c in top3_sorted],
        })

    # 매칭 안 된 unique prediction 추가
    for m_idx in range(1, n_models):
        for j, p in enumerate(filtered[m_idx]):
            if j not in used[m_idx]:
                merged.append(p)

    return merged


def evaluate_with_conf_threshold(
    data: Dict[str, Any],
    conf_thr: float = 0.05,
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    """단일 모델 결과를 conf threshold 적용해서 채점.

    Args:
        data: 학습 노트북 출력 형식 ({'results': [{filename, gt_bboxes, predictions}, ...], ...}).
        conf_thr: top1 confidence 하한.
        iou_thr: GT-prediction 매칭 IoU 임계값.

    Returns:
        {top1, top3, det, fp, n_top1, n_top3, n_match, n_false_pos, total_gt}.
    """
    n_top1 = n_top3 = n_match = n_false_pos = 0
    total_gt = sum(len(r['gt_bboxes']) for r in data['results'])

    for r in data['results']:
        preds = [p for p in r['predictions'] if p['top3_confidence'][0] >= conf_thr]
        used = set()
        for gt in r['gt_bboxes']:
            best_iou, best_idx = 0.0, -1
            for j, p in enumerate(preds):
                if j in used:
                    continue
                i = iou_xyxy(gt['bbox'], p['bbox'])
                if i > best_iou:
                    best_iou, best_idx = i, j
            if best_iou >= iou_thr and best_idx >= 0:
                n_match += 1
                used.add(best_idx)
                top3 = [nfc(n) for n in preds[best_idx]['top3_class_name']]
                gt_name = nfc(gt['dl_name'])
                if top3 and top3[0] == gt_name:
                    n_top1 += 1
                if gt_name in top3:
                    n_top3 += 1
        n_false_pos += len(preds) - len(used)

    return {
        'top1': n_top1 / total_gt,
        'top3': n_top3 / total_gt,
        'det': n_match / total_gt,
        'fp': n_false_pos,
        'n_top1': n_top1, 'n_top3': n_top3, 'n_match': n_match,
        'n_false_pos': n_false_pos, 'total_gt': total_gt,
    }


def evaluate_ensemble_nway(
    data_list: List[Dict[str, Any]],
    conf_thr: float = 0.2,
    iou_thr: float = 0.5,
    iou_match: float = 0.5,
) -> Dict[str, Any]:
    """N개 모델 결과를 ensemble 후 채점 (filename 기준 매칭).

    Args:
        data_list: 학습 노트북 출력 JSON 리스트 (모든 모델이 같은 33장 평가).
        conf_thr: ensemble 시 conf threshold.
        iou_thr: GT-merged prediction 매칭 IoU.
        iou_match: 모델 간 prediction 매칭 IoU.

    Returns:
        {top1, top3, det, fp}.
    """
    assert len(data_list) >= 1, 'data_list가 비어있음'

    n_top1 = n_top3 = n_match = n_false_pos = 0
    base = data_list[0]
    total_gt = sum(len(r['gt_bboxes']) for r in base['results'])
    other_maps = [{r['filename']: r for r in d['results']} for d in data_list[1:]]

    for r in base['results']:
        pred_lists = [r['predictions']] + [m[r['filename']]['predictions'] for m in other_maps]
        if len(data_list) == 1:
            # 단일 모델 → conf threshold만 적용
            merged = [p for p in r['predictions'] if p['top3_confidence'][0] >= conf_thr]
        else:
            merged = merge_predictions_nway(pred_lists, conf_thr=conf_thr, iou_match=iou_match)

        used = set()
        for gt in r['gt_bboxes']:
            best_iou, best_idx = 0.0, -1
            for j, p in enumerate(merged):
                if j in used:
                    continue
                i = iou_xyxy(gt['bbox'], p['bbox'])
                if i > best_iou:
                    best_iou, best_idx = i, j
            if best_iou >= iou_thr and best_idx >= 0:
                n_match += 1
                used.add(best_idx)
                top3 = [nfc(n) for n in merged[best_idx]['top3_class_name']]
                gt_name = nfc(gt['dl_name'])
                if top3 and top3[0] == gt_name:
                    n_top1 += 1
                if gt_name in top3:
                    n_top3 += 1
        n_false_pos += len(merged) - len(used)

    return {
        'top1': n_top1 / total_gt,
        'top3': n_top3 / total_gt,
        'det': n_match / total_gt,
        'fp': n_false_pos,
    }


# ────── 단일 이미지 추론용 (데모/배포 시) ──────

def ensemble_inference_single_image(
    model_predictions_per_model: List[List[Dict[str, Any]]],
    conf_thr: float = 0.2,
    iou_match: float = 0.5,
) -> List[Dict[str, Any]]:
    """단일 이미지에 대해 N개 모델의 prediction을 ensemble.

    데모/배포 시 사용. 각 모델이 동일 이미지에 대해 prediction을 생성한 뒤,
    이 함수로 합치면 ensemble 결과를 얻음.

    Args:
        model_predictions_per_model: [[v2_preds], [v3_preds], [v4_preds]] 형식.
            각 prediction은 {bbox, top3_class_name, top3_confidence}.

    Returns:
        merged predictions.
    """
    return merge_predictions_nway(model_predictions_per_model, conf_thr=conf_thr, iou_match=iou_match)


if __name__ == '__main__':
    import json
    import sys

    # CLI 사용 예시: python ensemble_utils.py v2.json v3.json [v4.json ...]
    if len(sys.argv) < 2:
        print('사용법: python ensemble_utils.py <result_json_1> <result_json_2> [...]')
        sys.exit(0)

    paths = sys.argv[1:]
    data_list = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            data_list.append(json.load(f))
        print(f'로드: {p} (best_epoch={data_list[-1].get("best_epoch", "?")})')

    print(f'\n=== {len(data_list)}-way ensemble (conf 0.2) ===')
    m = evaluate_ensemble_nway(data_list, conf_thr=0.2)
    print(f"Top-1: {m['top1']:.1%}")
    print(f"Top-3: {m['top3']:.1%}")
    print(f"Det:   {m['det']:.1%}")
    print(f"FP:    {m['fp']}")
