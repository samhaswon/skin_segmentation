import os
import time
from typing import Union, Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
import torch
import torch.nn.functional as F
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
from torch import nn
from tqdm import tqdm

from model.dlmv import DeepLabV3MobileNetV3

USE_TORCH = True
EXPORT = True
DEVICE = 'cuda'


def save_model_as_onnx(model, output_path, input_tensor_size=(1, 3, 512, 512)):
    """
    Saves the model in ONNX format.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    input_tensor_size : tuple, optional
        Size of the input tensor. Defaults to (1, 3, 512, 512).
    """
    x = torch.randn(*input_tensor_size, requires_grad=True)

    torch.onnx.export(
        model,
        x,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Model saved to:", output_path, "\n")


def iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    assert prediction.shape == ground_truth.shape
    intersection = np.bitwise_and(prediction, ground_truth).sum()
    union = np.bitwise_or(prediction, ground_truth).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


class TorchSession:
    def __init__(self, model: nn.Module, input_size: Optional[List[int]]):
        self.net = model.to(DEVICE)
        self.net.eval()
        if input_size is None:
            self.input_size = [512, 512]
        else:
            torch._assert(
                len(input_size) == 2,
                message="input_size must be a list of length 2"
            )
            torch._assert(
                input_size[0] == input_size[1],
                message="input_size must be a list of length 2 with identical elements"
            )
            self.input_size = input_size

    def remove(
            self,
            img: PILImage,
            size: Union[Tuple[int, int], None] = None,
            mask_only: bool = False
    ) -> np.ndarray:
        img_r = img.resize(size, Image.LANCZOS)
        image_tensor = torch.tensor(np.array(img_r), dtype=torch.float32).permute(2, 0, 1)
        image_tensor = F.interpolate(
            torch.unsqueeze(image_tensor, 0), self.input_size, mode="bilinear"
        ).type(torch.float32)
        image_tensor = torch.divide(image_tensor, torch.max(image_tensor)).type(torch.float32)
        image_tensor = image_tensor.to(DEVICE)
        img_result = self.net(image_tensor)
        img_result = img_result[0]
        result_array = img_result.cpu().data.numpy()

        re_max = np.max(result_array)
        re_min = np.min(result_array)
        result_array = (result_array - re_min) / (re_max - re_min + 1e-8)
        result_array = np.squeeze(result_array)

        alpha_channel = np.uint8(result_array * 255)
        alpha_channel = cv2.resize(alpha_channel, img.size, interpolation=cv2.INTER_LANCZOS4)
        alpha_channel = cv2.GaussianBlur(alpha_channel, (9, 9), 0)
        if mask_only:
            return alpha_channel
        return np.dstack((np.array(img), alpha_channel))


if __name__ == '__main__':
    start = time.perf_counter()

    # Quant settings
    # 'fbgemm' ('x86') for server, 'qnnpack' for mobile
    # 'x86' for x86...
    torch.backends.quantized.engine = 'qnnpack'
    filename = "/home/samuel/ai_data/skin_models/dlmv/checkpoint.pth_93.tar"
    output = "/home/samuel/ai_data/skin_models/dlmv/93_quant_qnnpack.onnx"
    torch_output = "/home/samuel/ai_data/skin_models/dlmv/93_quant_qnnpack.pth"
    size = 256

    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)

    net = DeepLabV3MobileNetV3(1)
    net.load_state_dict(checkpoint["state"]["state_dict"])
    net.eval()

    # FX Graph Mode: fuse appropriate Conv+BN(+ReLU) patterns across the MobileNetV3 backbone,
    # ASPP, and DeepLab head. No manual U2Net-style name lists.
    net_fused = fuse_fx(net)

    # Use default x86 qconfig mapping for post-training static quantization.
    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping('x86')

    dummy_tensor = torch.randn((1, 3, size, size), dtype=torch.float32)

    # Insert observers according to qconfig mapping
    net_prepared = prepare_fx(net_fused, qconfig_mapping=qconfig_mapping, example_inputs=dummy_tensor)

    # Calibrate on representative data
    session = TorchSession(net_prepared, input_size=[size, size])

    image_list = [x.path for x in os.scandir("/home/samuel/da/skindataset/images")]
    mask_list = [x.path for x in os.scandir("/home/samuel/da/skindataset/masks")]

    original_u2net_delta_list: List[float] = []

    for image_path, mask_path in tqdm(
        zip(image_list, mask_list),
        desc="Calibration",
        total=len(image_list)
    ):
        pil_image = Image.open(image_path)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert(mode="RGB")
        _ = session.remove(pil_image, mask_only=True, size=(size, size))  # run data through observers

    # Optionally, evaluate pre-convert IoU drift for reference
    # If you want that metric, uncomment the below loop and compute IoU before convert.
    # for image_path, mask_path in tqdm(...):
    #     pil_image = Image.open(image_path)
    #     if pil_image.mode == "RGBA":
    #         pil_image = pil_image.convert(mode="RGB")
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     pred = np.array(session.remove(pil_image, mask_only=True, size=(size, size)))
    #     original_u2net_delta_list.append(iou(pred, mask))

    # Convert to quantized model. Must be on CPU for quantized kernels.
    session.net = session.net.to('cpu').eval()
    session.net = convert_fx(session.net)

    if EXPORT:
        print("Exporting...")
        save_model_as_onnx(session.net, output_path=output, input_tensor_size=(1, 3, size, size))
        torch.save(session.net.state_dict(), torch_output)

    DEVICE = 'cpu'  # run int8 model on CPU
    u2net_delta_list: List[float] = []

    for image_path, mask_path in tqdm(
        zip(image_list, mask_list),
        desc="Inferencing",
        total=len(image_list)
    ):
        pil_image = Image.open(image_path)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert(mode="RGB")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = np.array(session.remove(pil_image, mask_only=True, size=(size, size)))
        u2net_delta_list.append(iou(pred, mask))

    end = time.perf_counter()

    # Was 0.86016590
    print(f"mIoU: {sum(u2net_delta_list) / len(image_list):.10f}")
    print(f"Time: {end - start:.2f}s")
