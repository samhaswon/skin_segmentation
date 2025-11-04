import cv2
import numpy as np

# Parameters for the elliptical skin model
mu_cb = 109.38
mu_cr = 152.02
a = 13.64
b = 15.38

# Build a 256x256 CbCr plane
width, height = 256, 256
cb = np.tile(np.arange(width, dtype=np.float32), (height, 1))
cr = np.tile(np.arange(height, dtype=np.float32).reshape(height, 1), (1, width))

# Apply the elliptical skin model: ((Cb - μCb)² / a²) + ((Cr - μCr)² / b²) ≤ 1
skin_mask = (((cb - mu_cb) ** 2) / (a ** 2) + ((cr - mu_cr) ** 2) / (b ** 2)) <= 1

# Create Y' channel: 255 where mask is true, else 0
y_plane = np.zeros((height, width), dtype=np.uint8)
y_plane[skin_mask] = 192

# Build a YCrCb image using OpenCV's channel order (Y, Cr, Cb)
# Cr is vertical axis, Cb is horizontal axis
ycrcb = np.dstack([y_plane, cr.astype(np.uint8), cb.astype(np.uint8)])
ycrcb = ycrcb.astype(np.uint8)

# Convert to BGR for viewing or saving
bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Save images if needed
cv2.imwrite("../examples/elliptical_cbcr_plane_Y255_region.png", bgr)
