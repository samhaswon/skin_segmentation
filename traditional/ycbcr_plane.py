import cv2
import numpy as np

width, height = 256, 256

cb = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
cr = np.tile(np.arange(height, dtype=np.uint8).reshape(height, 1), (1, width))

y_plane = np.zeros((height, width), dtype=np.uint8)
mask = (cb >= 85) & (cb <= 135) & (cr >= 137) & (cr <= 180)
y_plane[mask] = 192

ycbcr = np.dstack([y_plane, cr, cb]).astype(np.uint8)
img_bgr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

cv2.imwrite("../examples/cbcr_plane_Y255_region.png", img_bgr)
