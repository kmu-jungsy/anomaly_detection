from pathlib import Path


def load_boxes(txt_path):
    """
    txt format:
        x1 y1 x2 y2
        x1 y1 x2 y2
        ...
    """
    boxes = []
    txt_path = Path(txt_path)

    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")

    with open(txt_path, "r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                print(f"[Warning] Skip line {line_idx}: {line}")
                continue

            x1, y1, x2, y2 = map(float, parts[:4])
            boxes.append((x1, y1, x2, y2))

    return boxes


def compute_iou(box_a, box_b):
    """
    box format:
        (x1, y1, x2, y2)
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


def compute_ordered_iou(pred_txt, gt_txt, iou_threshold=0.5):
    pred_boxes = load_boxes(pred_txt)
    gt_boxes = load_boxes(gt_txt)

    n = min(len(pred_boxes), len(gt_boxes))

    print(f"Pred boxes: {len(pred_boxes)}")
    print(f"GT boxes:   {len(gt_boxes)}")
    print(f"Compare:    {n} pairs")
    print()

    tp = 0
    fp = 0
    fn = 0

    for i in range(n):
        pred = pred_boxes[i]
        gt = gt_boxes[i]
        iou = compute_iou(pred, gt)

        is_match = iou >= iou_threshold

        if is_match:
            tp += 1
        else:
            fp += 1
            fn += 1

        print(f"Pair {i}")
        print(f"  pred: {pred}")
        print(f"  gt:   {gt}")
        print(f"  IoU:  {iou:.4f}")
        print(f"  match >= {iou_threshold}: {is_match}")
        print()

    # pred가 더 많으면 남은 pred는 FP
    if len(pred_boxes) > len(gt_boxes):
        extra_fp = len(pred_boxes) - len(gt_boxes)
        fp += extra_fp
        print(f"Extra pred boxes without GT: {extra_fp} -> FP")

    # gt가 더 많으면 남은 gt는 FN
    if len(gt_boxes) > len(pred_boxes):
        extra_fn = len(gt_boxes) - len(pred_boxes)
        fn += extra_fn
        print(f"Extra GT boxes without pred: {extra_fn} -> FN")

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print("\n=== Summary ===")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


if __name__ == "__main__":
    pred_txt = "pred_boxes.txt"
    gt_txt = "gt_boxes.txt"

    compute_ordered_iou(
        pred_txt=pred_txt,
        gt_txt=gt_txt,
        iou_threshold=0.5,
    )
