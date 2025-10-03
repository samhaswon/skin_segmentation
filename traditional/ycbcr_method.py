"""
Implementation using the YCbCr color space to segment skin pixels.
"""
import cv2
import numpy as np


def segment_skin_ycbcr(image_rgb: np.ndarray,
                       cb_range: tuple[int, int] = (85, 135),
                       cr_range: tuple[int, int] = (135, 180)) -> np.ndarray:
    """
    Segment likely skin pixels using YCbCr thresholds, then post-process.

    Parameters
    ----------
    image_rgb
        Input image in RGB order, dtype uint8, shape (H, W, 3).
    cb_range
        Inclusive (min, max) bounds for the Cb channel used for masking.
    cr_range
        Inclusive (min, max) bounds for the Cr channel used for masking.

    Returns
    -------
    np.ndarray
        Binary mask, dtype uint8, shape (H, W), with values 0 or 255.
    """
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with shape (H, W, 3).")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be dtype uint8.")

    # OpenCV uses YCrCb channel order, not YCbCr, because of course it does
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YCrCb)

    # Build inclusive thresholds. Keep Y unconstrained.
    cr_min, cr_max = cr_range
    cb_min, cb_max = cb_range
    lower = np.array([0, cr_min, cb_min], dtype=np.uint8)
    upper = np.array([255, cr_max, cb_max], dtype=np.uint8)

    # Initial binary mask of likely skin pixels
    skin_mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    return skin_mask


if __name__ == '__main__':
    img = cv2.imread("/path/to/image")
    mask = segment_skin_ycbcr(img)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
