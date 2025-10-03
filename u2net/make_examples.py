import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm

from session import Session


def model_name_from_path(path: str) -> str:
    """Return the model name, using the filename without extension."""
    return os.path.splitext(os.path.basename(path))[0]


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert PIL image to RGB if not already."""
    return img.convert("RGB") if img.mode != "RGB" else img


def colorize_and_overlay(base: Image.Image, model_out: Image.Image) -> Image.Image:
    """
    Colorizes a cutout mask and overlays it onto the base image.

    :param base: The base PIL image.
    :param model_out: A cutout image (grayscale or binary mask) representing the region to overlay.
    :return: A new PIL image with the colorized overlay applied.
    """
    base = ensure_rgb(base)
    *_, alpha = model_out.copy().split()

    magenta = ImageOps.colorize(model_out.convert(mode="L"), black="black", white="magenta")

    result = base.copy()
    result.paste(magenta, (0, 0), alpha)

    return result


def run_and_plot(
    image_path_list: List[str],
    model_paths: List[str],
    session_factory
) -> None:
    """Run inference for each model on each input image and plot results in one figure.

    Parameters
    ----------
    image_path_list : list[str]
        List of input image paths.
    model_paths : list[str]
        Paths to model files. One Session will be created per path.
    session_factory : callable
        A callable that accepts a model path and returns a Session-like object
        with a .remove(PIL.Image) -> PIL.Image method.
    """
    rows = len(image_path_list) * len(model_paths)
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    if rows == 1:
        axes = np.array([axes])

    col_titles = ["original", "output (base with predicted alpha)", "colorized on base"]

    row_index = 0
    for img_in_path in tqdm(image_path_list):
        try:
            base_img = Image.open(img_in_path)
            base_img = ensure_rgb(base_img)
            base_name = os.path.basename(img_in_path)
            base_stem = os.path.splitext(base_name)[0]
        except Exception as exc:
            print(f"Failed to open {img_in_path}: {exc}")
            for _ in model_paths:
                for c in range(cols):
                    ax = axes[row_index, c]
                    ax.axis("off")
                    if row_index == 0 and c < len(col_titles):
                        ax.set_title(col_titles[c])
                # still add a row label so you know which pair failed
                ax0 = axes[row_index, 0]
                ax0.text(
                    -0.15, 0.5, f"{base_name} | error",
                    transform=ax0.transAxes, ha="right", va="center", clip_on=False
                )
                row_index += 1
            continue

        for model_path in model_paths:
            model_label = model_name_from_path(model_path)
            try:
                session = session_factory(model_path)
                model_out = session.remove(base_img)
                model_out = ensure_rgb(model_out)
            except Exception as exc:
                print(f"Inference failed for {model_path} on {img_in_path}: {exc}")
                for c in range(cols):
                    ax = axes[row_index, c]
                    ax.axis("off")
                    if row_index == 0 and c < len(col_titles):
                        ax.set_title(col_titles[c])
                ax0 = axes[row_index, 0]
                ax0.text(
                    -0.15, 0.5, f"{base_stem} | {model_label}",
                    transform=ax0.transAxes, ha="right", va="center", clip_on=False
                )
                row_index += 1
                continue

            overlay = colorize_and_overlay(base_img, model_out)

            triplet = [base_img, model_out, overlay]
            for c, img in enumerate(triplet):
                ax = axes[row_index, c]
                ax.imshow(img)
                ax.axis("off")
                if row_index == 0:
                    ax.set_title(col_titles[c])

            # Put a clear row label just outside the first axis
            ax0 = axes[row_index, 0]
            ax0.text(
                -0.05, 0.5, f"{model_label}",
                transform=ax0.transAxes, ha="right", va="center", clip_on=False,
            )

            row_index += 1

    # Layout: first tighten, then reserve room on the left for row labels
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)

    plt.savefig("../examples/model_plot.jpg", dpi=300)
    # plt.show()


if __name__ == "__main__":
    # Input example images
    image_path_list = [
        "../examples/S2563713746_St25_G7.5.jpeg",
        "../examples/Will-Smith-new-headshot-credit-Lorenzo-Agius.jpg",
        "../examples/zuck_original.jpg",
        "../examples/Official_portrait_of_Barack_Obama.jpg",
        "../examples/6e898e59-cee7-4d1e-97fc-5e27f075d56b.png",
        "../examples/Gemini_Generated_Image_zfkl36zfkl36zfkl.png",
        "../examples/Gemini_Generated_Image_burger.png",
    ]

    # Models to evaluate
    models = [
        "/home/samuel/ai_data/skin_models/BiRefNet/birefnet_skin.onnx",
        "/home/samuel/ai_data/skin_models/new/u2net.onnx",
        "/home/samuel/ai_data/skin_models/newp/u2netp.onnx",
        "/home/samuel/ai_data/skin_models/dlmv/dlmv.onnx",
    ]

    def session_factory(model_path: str) -> Session:
        """Create a Session for the given model path."""
        return Session(model_path)

    run_and_plot(image_path_list, models, session_factory)
