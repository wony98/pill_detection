import numpy as np

def iou_xyxy(b1, b2):
    """xyxy 형식 IoU."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)

def extract_top3(bboxes, scores, labels, class_names,
                 conf_thr=0.05, iou_thr=0.5, topk=3):
    keep = scores >= conf_thr
    bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
    if len(bboxes) == 0:
        return []

    order = np.argsort(-scores)
    bboxes, scores, labels = bboxes[order], scores[order], labels[order]

    n = len(bboxes)
    group_id = -np.ones(n, dtype=int)
    next_gid = 0
    for i in range(n):
        if group_id[i] >= 0:
            continue
        group_id[i] = next_gid
        for j in range(i + 1, n):
            if group_id[j] >= 0:
                continue
            if iou_xyxy(bboxes[i], bboxes[j]) > iou_thr:
                group_id[j] = next_gid
        next_gid += 1

    results = []
    for gid in range(next_gid):
        mask = group_id == gid
        g_bboxes, g_scores, g_labels = bboxes[mask], scores[mask], labels[mask]

        seen_cls = {}
        for b, s, l in zip(g_bboxes, g_scores, g_labels):
            l = int(l)
            if l not in seen_cls or s > seen_cls[l][1]:
                seen_cls[l] = (b, float(s), l)

        cls_sorted = sorted(seen_cls.values(), key=lambda x: -x[1])[:topk]
        rep_bbox = cls_sorted[0][0].tolist()

        results.append({
            'bbox': rep_bbox,
            'top3_class_name': [class_names[l] for _, _, l in cls_sorted],
            'top3_confidence': [s for _, s, _ in cls_sorted],
        })
    return results
