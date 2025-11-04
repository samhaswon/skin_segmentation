"""
Fit a Gaussian Mixture Model in chrominance-space for use with
an MRF smoothness prior solved by graph-cuts.
"""

import os
import pickle
import random
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from numpy.random import default_rng
from sklearn.mixture import GaussianMixture


def bgr_to_ycrcb_uint8(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 image to YCrCb uint8 using OpenCV channel convention.

    :param img_bgr: Input BGR image (uint8), shape (H, W, 3).
    :type img_bgr: numpy.ndarray
    :returns: YCrCb image (uint8), shape (H, W, 3), channels ordered [Y, Cr, Cb].
    :rtype: numpy.ndarray
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)


def _sample_indices(
    idx: np.ndarray, max_take: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Randomly choose up to ``max_take`` indices from ``idx`` without replacement.

    :param idx: 1D array of candidate indices.
    :type idx: numpy.ndarray
    :param max_take: Maximum number of indices to sample.
    :type max_take: int
    :param rng: Random generator for reproducibility.
    :type rng: numpy.random.Generator
    :returns: 1D array of sampled indices, length in ``[0, min(len(idx), max_take)]``.
    :rtype: numpy.ndarray
    """
    if idx.size == 0 or max_take <= 0:
        return idx[:0]
    if idx.size <= max_take:
        return idx
    return rng.choice(idx, size=max_take, replace=False)


def collect_cbcr_samples_streaming(
    image_paths: Sequence[str],
    mask_paths: Sequence[str],
    max_skin: int = 500_000,
    max_bg: int = 500_000,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stream images and collect [Cb, Cr] samples using masks, without retaining all pixels in RAM.

    The function limits per-image sampling to keep memory bounded and builds final arrays
    of size at most ``max_skin`` and ``max_bg``.

    :param image_paths: Paths to BGR images.
    :type image_paths: Sequence[str]
    :param mask_paths: Paths to binary masks aligned with ``image_paths`` (skin=255, background=0).
    :type mask_paths: Sequence[str]
    :param max_skin: Upper bound on the number of skin samples to return.
    :type max_skin: int
    :param max_bg: Upper bound on the number of background samples to return.
    :type max_bg: int
    :param rng: Optional random generator. If ``None``, a default generator is used.
    :type rng: numpy.random.Generator | None
    :returns: Tuple of arrays ``(skin_samples, bg_samples)``, each with shape ``(N, 2)`` of [Cb, Cr].
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    :raises RuntimeError: If no samples are collected from the provided inputs.
    """
    if rng is None:
        rng = default_rng(42)

    n_imgs = max(1, min(len(image_paths), len(mask_paths)))

    # Per-image budget to avoid ever holding large intermediate arrays after sampling
    per_img_skin = max(1, max_skin // n_imgs)
    per_img_bg = max(1, max_bg // n_imgs)

    skin_parts: List[np.ndarray] = []
    bg_parts: List[np.ndarray] = []
    skin_total = 0
    bg_total = 0

    for img_p, msk_p in zip(image_paths, mask_paths):
        img = cv2.imread(img_p, cv2.IMREAD_COLOR)
        if img is None:
            continue

        m = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape[:2] != img.shape[:2]:
            continue

        ycrcb = bgr_to_ycrcb_uint8(img)
        # OpenCV YCrCb -> channels: [Y, Cr, Cb]
        cr = ycrcb[:, :, 1].reshape(-1)
        cb = ycrcb[:, :, 2].reshape(-1)

        mask = (m > 127).reshape(-1)

        idx_skin = np.flatnonzero(mask)
        idx_bg = np.flatnonzero(~mask)

        # Determine how many we can still take
        take_skin = min(per_img_skin, max(0, max_skin - skin_total))
        take_bg = min(per_img_bg, max(0, max_bg - bg_total))

        sel_skin = _sample_indices(idx_skin, take_skin, rng)
        sel_bg = _sample_indices(idx_bg, take_bg, rng)

        if sel_skin.size:
            skin_parts.append(np.stack((cb[sel_skin], cr[sel_skin]), axis=1).astype(np.float32))
            skin_total += sel_skin.size

        if sel_bg.size:
            bg_parts.append(np.stack((cb[sel_bg], cr[sel_bg]), axis=1).astype(np.float32))
            bg_total += sel_bg.size

        if skin_total >= max_skin and bg_total >= max_bg:
            break

    if not skin_parts or not bg_parts:
        raise RuntimeError(
            "No samples collected. Check paths, masks, image formats, or per-image budgets."
        )

    skin = np.concatenate(skin_parts, axis=0)
    bg = np.concatenate(bg_parts, axis=0)

    # Defensive trim in case of rounding
    if skin.shape[0] > max_skin:
        skin = skin[:max_skin]
    if bg.shape[0] > max_bg:
        bg = bg[:max_bg]

    return skin, bg


def fit_gmm(data: np.ndarray, n_components: int, reg: float = 1e-6) -> GaussianMixture:
    """
    Fit a full-covariance GMM on 2D [Cb, Cr] samples.

    :param data: Array of shape ``(N, 2)``.
    :type data: numpy.ndarray
    :param n_components: Number of mixture components.
    :type n_components: int
    :param reg: Diagonal regularization added to covariances.
    :type reg: float
    :returns: Trained GaussianMixture instance.
    :rtype: sklearn.mixture.GaussianMixture
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=reg,
        max_iter=200,
        n_init=1,
        random_state=42,
        init_params="kmeans",
        warm_start=False,
    )
    gmm.fit(data)
    return gmm


def _pair_and_shuffle(
    image_dir: str,
    mask_dir: str,
    limit: int | None,
    shuffle: bool,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Scan two directories, pair files by order, and optionally shuffle deterministically.

    :param image_dir: Directory of images.
    :type image_dir: str
    :param mask_dir: Directory of masks.
    :type mask_dir: str
    :param limit: Optional cap on the number of pairs to take.
    :type limit: int | None
    :param shuffle: Whether to shuffle the paired lists.
    :type shuffle: bool
    :param seed: Seed for deterministic shuffling.
    :type seed: int
    :returns: ``(image_list, mask_list)`` as lists of file paths.
    :rtype: Tuple[List[str], List[str]]
    """
    imgs = [x.path for x in os.scandir(image_dir) if x.is_file()]
    msks = [x.path for x in os.scandir(mask_dir) if x.is_file()]

    if limit is not None:
        imgs = imgs[:limit]
        msks = msks[:limit]

    if len(imgs) != len(msks):
        print("Warning: image and mask counts differ. Proceeding by zipping in order.")

    paired = list(zip(imgs, msks))
    if shuffle and paired:
        rnd = random.Random(seed)
        rnd.shuffle(paired)

    if paired:
        imgs, msks = map(list, zip(*paired))
    else:
        imgs, msks = [], []

    return imgs, msks


if __name__ == "__main__":
    SKIN_COMPONENTS: int = 3
    BG_COMPONENTS: int = 6
    SHUFFLE: bool = True
    MAX_SKIN: int = 500_000
    MAX_BG: int = 500_000
    LIMIT: int | None = 1134  # optional cap on pairs considered

    IMAGE_DIR = "/home/samuel/da/skindataset/images"
    MASK_DIR = "/home/samuel/da/skindataset/masks"

    image_list, mask_list = _pair_and_shuffle(
        IMAGE_DIR, MASK_DIR, limit=LIMIT, shuffle=SHUFFLE, seed=42
    )

    rng = default_rng(42)

    skin, bg = collect_cbcr_samples_streaming(
        image_list,
        mask_list,
        max_skin=MAX_SKIN,
        max_bg=MAX_BG,
        rng=rng,
    )

    print(f"Skin samples: {skin.shape[0]:,}, BG samples: {bg.shape[0]:,}")

    gmm_skin = fit_gmm(skin, n_components=SKIN_COMPONENTS)
    gmm_bg = fit_gmm(bg, n_components=BG_COMPONENTS)

    with open("gmm_skin.pkl", "wb") as f:
        pickle.dump(gmm_skin, f)
    with open("gmm_bg.pkl", "wb") as f:
        pickle.dump(gmm_bg, f)

    print("Saved gmm_skin.pkl and gmm_bg.pkl")
