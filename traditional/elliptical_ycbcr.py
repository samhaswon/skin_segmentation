"""
Implementation using Elliptical Skin Cluster Modeling in the YCbCr color space to segment skin pixels.
"""
import cv2
import numpy as np


def segment_skin_elliptical_ycbcr(image_bgr: np.ndarray,
                                  mu_cb: float = 109.38,
                                  mu_cr: float = 152.02,
                                  a: float = 13.64,
                                  b: float = 15.38) -> np.ndarray:
    """
    Segment likely skin pixels using Elliptical Skin Cluster Modeling in YCbCr.

    :param image_bgr: Input image in BGR order, dtype uint8, shape (H, W, 3).
    :param mu_cb: μCb
    :param mu_cr: μCr
    :param a: a
    :param b: b
    :returns: (np.ndarray) Binary mask, dtype uint8, shape (H, W), with values 0 or 255.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with shape (H, W, 3).")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_rgb must be dtype uint8.")

    # OpenCV uses YCrCb channel order, not YCbCr, because of course it does
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    # Apply (Cb - μCb)² / a² + (Cr - μCr)² / b² ≤ 1
    skin_mask = (((ycrcb[..., 2] - mu_cb) ** 2) / a ** 2 + ((ycrcb[..., 1] - mu_cr) ** 2) / b ** 2) <= 1
    skin_mask = skin_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    return skin_mask


if __name__ == '__main__':
    img = cv2.imread("/path/to/image")
    mask = segment_skin_elliptical_ycbcr(img)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
