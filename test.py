from pathlib import Path


def load_boxes(txt_path):
    """
    txt format:
        x1 y1 x2 y2
        x1 y1 x2 y2
        ...

    return:
        [(x1, y1, x2, y2), ...]
    """
    boxes = []

    txt_path = Path(txt_path)
    if not txt_path.exists():
        print(f"[Warning] File not found: {txt_path}")
        return boxes

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                continue

            x1, y1, x2, y2 = map(float, parts[:4])
            boxes.append((x1, y1, x2, y2))

    return boxes


def compute_iou(box_a, box_b):
    """
    box format:
        x1, y1, x2, y2
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    One-to-one matching.

    Returns:
        matches: [(pred_idx, gt_idx, iou), ...]
        unmatched_preds: [pred_idx, ...]
        unmatched_gts: [gt_idx, ...]
    """
    candidates = []

    # 모든 pred-GT 조합 IoU 계산
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou = compute_iou(pred, gt)

            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    # IoU 높은 것부터 매칭
    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_preds = set()
    matched_gts = set()
    matches = []

    for iou, pi, gi in candidates:
        if pi in matched_preds:
            continue
        if gi in matched_gts:
            continue

        matched_preds.add(pi)
        matched_gts.add(gi)
        matches.append((pi, gi, iou))

    unmatched_preds = [
        i for i in range(len(pred_boxes))
        if i not in matched_preds
    ]

    unmatched_gts = [
        i for i in range(len(gt_boxes))
        if i not in matched_gts
    ]

    return matches, unmatched_preds, unmatched_gts


if __name__ == "__main__":
    pred_txt = "pred_boxes.txt"
    gt_txt = "gt_boxes.txt"

    pred_boxes = load_boxes(pred_txt)
    gt_boxes = load_boxes(gt_txt)

    matches, unmatched_preds, unmatched_gts = match_boxes(
        pred_boxes,
        gt_boxes,
        iou_threshold=0.5
    )

    print("=== Pred boxes ===")
    for i, box in enumerate(pred_boxes):
        print(f"pred {i}: {box}")

    print("\n=== GT boxes ===")
    for i, box in enumerate(gt_boxes):
        print(f"gt {i}: {box}")

    print("\n=== IoU >= 0.5 Matches ===")
    for pred_idx, gt_idx, iou in matches:
        print(
            f"pred {pred_idx} matched with gt {gt_idx} | "
            f"IoU = {iou:.4f}"
        )

    print("\n=== Unmatched Pred Boxes ===")
    for idx in unmatched_preds:
        print(f"pred {idx}: {pred_boxes[idx]}")

    print("\n=== Unmatched GT Boxes ===")
    for idx in unmatched_gts:
        print(f"gt {idx}: {gt_boxes[idx]}")

    tp = len(matches)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print("\n=== Metrics ===")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
