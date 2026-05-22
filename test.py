import cv2
import numpy as np

mask_path = "mask/02_mask.jpg"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print("shape:", mask.shape)
print("min:", mask.min())
print("max:", mask.max())
print("unique values:", np.unique(mask)[:50])
print("num unique:", len(np.unique(mask)))
