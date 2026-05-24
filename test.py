pred box와 gt box의 모든 조합 IoU 계산
IoU >= iou_threshold 인 후보만 유지
IoU가 높은 순서대로 1:1 greedy matching
매칭된 pred = TP
매칭 안 된 pred = FP
매칭 안 된 gt = FN
