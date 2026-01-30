"""
Implementation using Diagonal Elliptical Skin Cluster Modeling in the YCbCr color space to
segment skin pixels.
"""
import cv2
import numpy as np


def segment_skin_diagonal_elliptical_ycbcr(
    image_bgr: np.ndarray,
    cb0: float = 109.38,
    cr0: float = 152.02,
    a: float = 8.64,
    b: float = 28.38,
    angle_deg: float = 45.0
) -> np.ndarray:
    """
    Segment skin using a rotated ellipse in Cb-Cr space:
    ((x')/a)^2 + ((y')/b)^2 <= 1, where [x', y'] is (Cb,Cr) rotated by angle_deg about (cb0, cr0).
    Note that the default parameters are reasonable defaults, but not the best parameters.
    You can find the best parameters in the usage later in this file.

    :param image_bgr: Input image in BGR order, dtype uint8, shape (H, W, 3).
    :param cb0: Ellipse center on Cb axis.
    :param cr0: Ellipse center on Cr axis.
    :param a: Semi-axis along the ellipse's local x'.
    :param b: Semi-axis along the ellipse's local y'.
    :param angle_deg: Rotation angle in degrees, positive rotates x' toward Cr.
    :return: Boolean mask where True means inside the ellipse.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with shape (H, W, 3).")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_rgb must be dtype uint8.")

    # OpenCV uses YCrCb channel order, not YCbCr, because of course it does
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    x = ycrcb[..., 2] - cb0
    y = ycrcb[..., 1] - cr0

    # Rotate coordinates into ellipse frame
    x_prime = c * x + s * y
    y_prime = -s * x + c * y

    skin_mask = (x_prime / a) ** 2 + (y_prime / b) ** 2 <= 1.0
    skin_mask = skin_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    return skin_mask


if __name__ == '__main__':
    img = cv2.imread("/path/to/image")
    # This usage utilizes parameters tuned on the dataset for the best accuracy.
    # Hence, the comparatively long numbers here.
    mask = segment_skin_diagonal_elliptical_ycbcr(
        img,
        cb0=119.31423371082059,
        cr0=155.55795465744077,
        a=37.089397631238135,
        b=14.80887169802648,
        angle_deg=148.63290799558428,
    )
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
