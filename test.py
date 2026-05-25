import cv2
import numpy as np

mask_path = "mask.jpg"
output_path = "mask_inside_white.jpg"

# read masked image
mask_img = cv2.imread(mask_path)

if mask_img is None:
    raise FileNotFoundError(f"Cannot find image: {mask_path}")

# convert to grayscale
gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

# 검은색 영역 판단 기준
# gray <= threshold : 검은색 영역
# gray > threshold  : 마스크 내부 영역
threshold = 10

black_area = gray <= threshold
inside_area = gray > threshold

# output image 생성
result = np.zeros_like(mask_img)  # 기본은 검은색

# 마스크 내부 영역을 흰색으로 변경
result[inside_area] = [255, 255, 255]

# 검은색 영역은 그대로 검은색
result[black_area] = [0, 0, 0]

# save
cv2.imwrite(output_path, result)

print(f"Saved: {output_path}")
