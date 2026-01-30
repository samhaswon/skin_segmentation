#!/usr/bin/env python3

"""
This is for fitting the elliptical YCbCr method.
It takes quite a while to do.
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Sequence, Tuple

# pylint: disable=no-member
import cv2
import numpy as np
from tqdm import tqdm

try:
    from scipy.optimize import differential_evolution
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    # pylint: disable=unnecessary-lambda-assignment
    differential_evolution = lambda x: x


@dataclass(frozen=True)
class EllipseParams:
    """
    Dataclass for parameters for the ellipse segmentation function.
    """
    cb0: float
    cr0: float
    a: float
    b: float
    angle_deg: float


def load_gray_mask(path: str) -> np.ndarray:
    """
    Load and binarize a grayscale mask.
    :param path: The path to the image to load.
    :return: Grayscale mask.
    :raises FileNotFoundError: Path does not exist.
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8) * 255


def load_bgr(path: str) -> np.ndarray:
    """
    Loads a bgr image.
    :param path: The path to the image to load.
    :return: Color base image.
    :raises FileNotFoundError: Path does not exist.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def seg_skin_rot_ellipse(image_bgr: np.ndarray, p: EllipseParams) -> np.ndarray:
    """
    Does segmentation of the input image without post-processing using a diagonal ellipse.
    :param image_bgr: The image to segment.
    :param p: Ellipse parameters.
    :return: Segmentation mask.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]

    theta = np.deg2rad(p.angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    x = cb - p.cb0
    y = cr - p.cr0

    x_prime = c * x + s * y
    y_prime = -s * x + c * y

    inside = (x_prime / max(p.a, 1e-6)) ** 2 + (y_prime / max(p.b, 1e-6)) ** 2 <= 1.0
    mask = np.zeros_like(cb, dtype=np.uint8)
    mask[inside] = 255
    return mask


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculates the IoU between prediction and ground truth.
    :param pred: The predicted mask.
    :param gt: The ground truth mask.
    :return: IoU score.
    """
    pred_b = pred > 0
    gt_b = gt > 0

    inter = np.logical_and(pred_b, gt_b).sum(dtype=np.int64)
    union = np.logical_or(pred_b, gt_b).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mean_iou(
    pairs: Sequence[Tuple[str, str]],
    params: EllipseParams,
    stride: int = 1,
    max_images: int | None = None,
) -> float:
    """
    Calculates the average (mean) IoU score on the data.
    :param pairs: Pairs of paths to input images and masks.
    :param params: Parameters for the ellipse parameters.
    :param stride: Stride across the data.
    :param max_images: The maximum number of images to test against.
    :return: Mean IoU score on the data of the given parameters.
    """
    total = 0.0
    n = 0

    for ip, mp in tqdm(pairs[::stride], desc="Evaluating", leave=False):
        if max_images is not None and n >= max_images:
            break
        img = load_bgr(ip)
        gt = load_gray_mask(mp)
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred = seg_skin_rot_ellipse(img, params)
        total += iou_score(pred, gt)
        n += 1

    return 0.0 if n == 0 else total / float(n)


def fit_params(
    train_pairs: Sequence[Tuple[str, str]],
    budget: int = 200,
    stride: int = 1,
    max_images: int | None = None,
    seed: int = 1337,
) -> EllipseParams:
    """
    Fit the diagonal ellipse function's parameters on the dataset.
    :param train_pairs: Pairs of training data.
    :param budget: The number of iterations to run to fit parameters.
    :param stride: The stride on the train samples to test.
    :param max_images: The maximum number of images to test against.
    :param seed: The random seed to use for scipy.
    :return: The best parameters found.
    """
    bounds = [
        (60.0, 160.0),   # cb0
        (120.0, 190.0),  # cr0
        (3.0, 60.0),     # a
        (3.0, 60.0),     # b
        (0.0, 180.0),    # angle
    ]

    def objective(x: np.ndarray) -> float:
        p = EllipseParams(cb0=float(x[0]), cr0=float(x[1]),
                          a=float(x[2]), b=float(x[3]), angle_deg=float(x[4]))
        score = mean_iou(train_pairs, p, stride=stride, max_images=max_images)
        area_term = (p.a * p.b) / (60.0 * 60.0)
        return -(score - 0.01 * area_term)

    if _HAVE_SCIPY:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=max(1, budget // (15 + len(bounds))),
            popsize=15,
            tol=1e-6,
            polish=True,
            seed=seed,
        )
        x = result.x
    else:
        print("SciPy not available. Falling back to random search.")
        best_val = float("inf")
        best_x = None
        for _ in tqdm(range(budget), desc="Random Search"):
            x_try = np.array([np.random.uniform(*b) for b in bounds])
            val = objective(x_try)
            if val < best_val:
                best_val, best_x = val, x_try
        x = best_x

    return EllipseParams(cb0=float(x[0]), cr0=float(x[1]),
                         a=float(x[2]), b=float(x[3]), angle_deg=float(x[4]))


def main() -> None:
    """main"""
    image_list = [x.path for x in os.scandir("./images")]
    mask_list = [x.path for x in os.scandir("./masks")]
    image_list.sort()
    mask_list.sort()

    # Train = [:1134], Eval = [1134:]
    train_pairs = list(zip(image_list[:1134], mask_list[:1134]))
    eval_pairs = list(zip(image_list[1134:], mask_list[1134:]))

    # Fit the parameters
    best = fit_params(
        train_pairs,
        budget=400,
        stride=3,
        max_images=300,
        seed=1337,
    )

    # Calculate the mIoU on both the train and eval datasets
    train_iou = mean_iou(train_pairs, best, stride=1, max_images=None)
    eval_iou = mean_iou(eval_pairs, best, stride=1, max_images=None)

    print("\nBest parameters:")
    print(json.dumps(best.__dict__, indent=2))
    print(f"Train IoU: {train_iou:.4f}")
    print(f"Eval  IoU: {eval_iou:.4f}")

    with open("cbcr_ellipse_params.json", "w", encoding="utf-8") as f:
        json.dump(best.__dict__, f, indent=2)

    # Quick visual check
    if len(eval_pairs) > 0:
        img = load_bgr(eval_pairs[0][0])
        gt = load_gray_mask(eval_pairs[0][1])
        pred = seg_skin_rot_ellipse(img, best)
        overlay = img.copy()
        contours, _ = cv2.findContours((pred > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        out = np.hstack([
            img,
            cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR),
            overlay,
        ])
        cv2.imwrite("eval_preview.png", out)


if __name__ == "__main__":
    main()
