"""
Segment skin with a chrominance-space GMM with an MRF smoothness prior solved by graph-cuts.

https://www.academia.edu/download/88379082/A_Survey_on_Pixel-Based_Skin_Color_Detection_Techniques.pdf
"""
import pickle
import time

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


def rgb_to_ycbcr_uint8(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 to YCrCb uint8, OpenCV convention.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image, uint8.

    Returns
    -------
    np.ndarray
        YCrCb image, uint8 (Y, Cr, Cb).
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)


def posterior_skin(cbcr: np.ndarray,
                   gmm_skin: GaussianMixture,
                   gmm_bg: GaussianMixture,
                   prior_skin: float = 0.3,
                   eps: float = 1e-12) -> np.ndarray:
    """
    Compute P(skin | CbCr) per pixel using GMMs and prior.

    Parameters
    ----------
    cbcr : np.ndarray
        Array (H, W, 2) with [Cb, Cr] per pixel, float32 or float64.
    gmm_skin : GaussianMixture
        Skin GMM.
    gmm_bg : GaussianMixture
        Background GMM.
    prior_skin : float
        Prior probability of skin in [0, 1].
    eps : float
        Numerical stability term.

    Returns
    -------
    np.ndarray
        Posterior in [0, 1], shape (H, W).
    """
    h, w, _ = cbcr.shape
    x = cbcr.reshape(-1, 2)
    # score_samples returns log-likelihood under mixture
    ps = np.exp(gmm_skin.score_samples(x))
    pn = np.exp(gmm_bg.score_samples(x))
    pi_s = float(prior_skin)
    pi_n = 1.0 - pi_s
    num = pi_s * ps
    den = num + pi_n * pn + eps
    return (num / den).reshape(h, w)


def estimate_beta(image_lab: np.ndarray) -> float:
    """
    Estimate beta for contrast weights, using 4-neighborhood differences.

    Parameters
    ----------
    image_lab : np.ndarray
        Lab image (H, W, 3), float32.

    Returns
    -------
    float
        Beta value.
    """
    diffs = []
    diffs.append((image_lab[:, 1:, :] - image_lab[:, :-1, :]) ** 2)
    diffs.append((image_lab[1:, :, :] - image_lab[:-1, :, :]) ** 2)
    sse = 0.0
    count = 0
    for d in diffs:
        sse += d.sum()
        count += d.size // 3
    mean_dist2 = sse / max(count, 1)
    if mean_dist2 <= 0:
        return 1.0
    return 1.0 / (2.0 * mean_dist2)


def build_unaries(p_skin: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Build unary energies for labels [non-skin, skin].

    Parameters
    ----------
    p_skin : np.ndarray
        Posterior probabilities (H, W).
    eps : float
        Numerical stability.

    Returns
    -------
    np.ndarray
        (H, W, 2) energies.
    """
    d1 = -np.log(np.clip(p_skin, eps, 1.0))
    d0 = -np.log(np.clip(1.0 - p_skin, eps, 1.0))
    return np.stack([d0, d1], axis=-1).astype(np.float32)


def icm_segment(unaries: np.ndarray,
                image_lab: np.ndarray,
                lam: float = 40.0,
                iters: int = 5) -> np.ndarray:
    """
    ICM optimization for a contrast-sensitive Potts MRF.

    Parameters
    ----------
    unaries : np.ndarray
        (H, W, 2) energies for [non-skin, skin].
    image_lab : np.ndarray
        Lab image (H, W, 3), float32.
    lam : float
        Smoothness weight.
    iters : int
        Number of ICM sweeps.

    Returns
    -------
    np.ndarray
        Binary mask (H, W), uint8 with {0,1}.
    """
    h, w, _ = unaries.shape
    lab = image_lab.astype(np.float32, copy=False)
    U0 = unaries[:, :, 0].astype(np.float32, copy=False)
    U1 = unaries[:, :, 1].astype(np.float32, copy=False)

    beta = estimate_beta(lab)

    # Directional validity masks
    mask_n = np.ones((h, w), dtype=np.float32); mask_n[0, :] = 0.0
    mask_s = np.ones((h, w), dtype=np.float32); mask_s[-1, :] = 0.0
    mask_w = np.ones((h, w), dtype=np.float32); mask_w[:, 0] = 0.0
    mask_e = np.ones((h, w), dtype=np.float32); mask_e[:, -1] = 0.0

    def pair_weights(shifted, mask):
        diff = lab - shifted
        dist2 = np.sum(diff * diff, axis=2, dtype=np.float32)
        w = lam * np.exp(-beta * dist2, dtype=np.float32)
        w *= mask
        return w

    # Precompute contrast-sensitive weights for each direction
    wN = pair_weights(np.roll(lab, 1, axis=0), mask_n)
    wS = pair_weights(np.roll(lab, -1, axis=0), mask_s)
    wW = pair_weights(np.roll(lab, 1, axis=1), mask_w)
    wE = pair_weights(np.roll(lab, -1, axis=1), mask_e)

    sum_w = wN + wS + wW + wE  # sum of neighbor weights per pixel

    # Initialize labels from unary MAP
    labels = (U1 < U0).astype(np.uint8)

    for _ in range(iters):
        # Shifted neighbor labels
        Ln = np.roll(labels, 1, axis=0) * mask_n
        Ls = np.roll(labels, -1, axis=0) * mask_s
        Lw = np.roll(labels, 1, axis=1) * mask_w
        Le = np.roll(labels, -1, axis=1) * mask_e

        # Sum of weights where neighbor is label 1
        wl = wN * Ln + wS * Ls + wW * Lw + wE * Le

        # Binary Potts, costs: add wij if neighbor != current label
        e0 = U0 + wl
        e1 = U1 + (sum_w - wl)

        labels = (e1 < e0).astype(np.uint8)

    return labels.astype(np.uint8)


def process_image(img_bgr: np.ndarray,
                  gmm_skin: GaussianMixture,
                  gmm_bg: GaussianMixture,
                  prior_skin: float,
                  lam: float,
                  icm_iters: int) -> np.ndarray:
    """
    Segment one image, returning a binary mask.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image, BGR uint8.
    gmm_skin : GaussianMixture
        Skin model.
    gmm_bg : GaussianMixture
        Background model.
    prior_skin : float
        Prior probability for skin.
    lam : float
        Smoothness weight.
    icm_iters : int
        ICM iterations.

    Returns
    -------
    np.ndarray
        Mask uint8 with values {0, 255}.
    """
    ycrcb = rgb_to_ycbcr_uint8(img_bgr).astype(np.float32)
    # Extract [Cb, Cr] in that order (OpenCV gives [Y, Cr, Cb])
    cbcr = np.stack([ycrcb[:, :, 2], ycrcb[:, :, 1]], axis=-1)

    p = posterior_skin(cbcr, gmm_skin, gmm_bg, prior_skin=prior_skin)
    unaries = build_unaries(p)

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    skin_mask = icm_segment(unaries, img_lab, lam=lam, iters=icm_iters)

    skin_mask = (skin_mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    return skin_mask


if __name__ == "__main__":
    PRIOR_SKIN = 0.3
    LAM = 40.0
    ICM_ITERS = 5

    with open("gmm_skin.pkl", "rb") as f:
        gmm_skin = pickle.load(f)
    with open("gmm_bg.pkl", "rb") as f:
        gmm_bg = pickle.load(f)

    img = cv2.imread("/home/samuel/code_projects/Python_Code/skin/images/00752.png")
    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    start = time.perf_counter()
    mask = process_image(
        img, gmm_skin, gmm_bg,
        prior_skin=PRIOR_SKIN, lam=LAM, icm_iters=ICM_ITERS
    )
    end = time.perf_counter()
    print(f"Segmentation time: {end - start:.4f}s")
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("result.png", result)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
