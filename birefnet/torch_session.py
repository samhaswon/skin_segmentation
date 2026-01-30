"""
Session for using BiRefNet as the initial model.
"""
from typing import Tuple, Any, Optional

import cv2
import numpy as np
from PIL.Image import Image as PILImage  # For typing
import torch
from torch import nn
import torch.nn.functional as F

from models.birefnet import BiRefNet


# Default BiRefNet size
IMAGE_SIZE = 1728


# -----------------------
# Model adapter
# -----------------------
class BiRefNetHFAdapter(nn.Module):
    """
    Wraps the base model to present logits: (N, 1, H, W).
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.base(pixel_values)
        return out[0]


@torch.compile
def resize_and_pad_square(x: torch.Tensor, target: int) -> Tuple[torch.Tensor, int, int]:
    """
    Resize CHW tensor so the longer side == target, preserve aspect,
    then pad to (target, target). target should be a multiple of 32.
    """
    c, h, w = x.shape
    if h >= w:
        new_h = target
        new_w = max(1, int(round(w * target / h)))
    else:
        new_w = target
        new_h = max(1, int(round(h * target / w)))

    x = F.interpolate(x[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]

    pad_h = target - new_h
    pad_w = target - new_w
    # pad=(left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Morphs and blurs the mask to make it a bit better (generally speaking).
    :param mask: The mask to post-process.
    :return: The post-processed mask.
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    return mask


class BiRefNetTorchSession:
    """
    Session for Torch inference of BiRefNet with post-processing.
    """

    def __init__(self, net_path: str = "models/birefnet.pth", half_precision: bool = False, device: str = None) -> None:
        checkpoint = torch.load(net_path, map_location='cpu', weights_only=False)
        net = BiRefNet(bb_pretrained=False)
        net.load_state_dict(checkpoint)
        if device is not None:
            self.device = device
        else:
            self.device = "cpu"

        self.net = BiRefNetHFAdapter(net).to(self.device)
        self.net.eval()
        self.half_precision = half_precision
        if half_precision:
            self.net.half()

    def compile(self):
        """JIT compiles the model."""
        self.net = torch.compile(self.net)

    def remove(
            self,
            img: PILImage,
            size: Optional[Tuple[int, int]] = None,
            mask_only: bool = False
    ) -> np.ndarray[Any, Tuple[np.uint8]]:
        """
        Runs inferencing with post-processing.
        :param img: The image to be processed.
        :param size: Unused, for compatibility with other onnx code.
        :param mask_only: If True, it returns only the mask.
        :return: Either the mask (L or A) or the original image with the
        alpha channel applied (RGBA).
        """
        if size is None:
            size = IMAGE_SIZE, IMAGE_SIZE
        image_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        image_tensor, pad_h, pad_w = resize_and_pad_square(image_tensor, size[0])
        image_tensor = image_tensor.to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        if self.half_precision:
            image_tensor = image_tensor.half()
        with torch.inference_mode():
            img_result = self.net(image_tensor)
        if self.half_precision:
            img_result = img_result.to(torch.float32)
        img_result = img_result[..., :IMAGE_SIZE - pad_h, :IMAGE_SIZE - pad_w]
        img_result = torch.sigmoid(img_result)
        img_result = torch.where(img_result > 0.5, img_result, torch.tensor(0.0))
        result_array = img_result.cpu().data.numpy()

        # Norm the prediction
        result_array = np.squeeze(result_array)
        alpha_channel = np.uint8(result_array * 255)
        alpha_channel = cv2.resize(alpha_channel, img.size, interpolation=cv2.INTER_LANCZOS4)
        alpha_channel = cv2.morphologyEx(alpha_channel, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        alpha_channel = cv2.GaussianBlur(alpha_channel, (3, 3), 0)
        if mask_only:
            return alpha_channel
        return np.dstack((np.array(img), alpha_channel))
