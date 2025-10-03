"""
Quantization script for U2Net and U2NetP.
"""
import os
import time
from typing import Union, Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage  # For typing
import torch
import torch.nn.functional as F
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
from torch import nn
from tqdm import tqdm

from model.u2net import U2NET, U2NETP

USE_TORCH = True
EXPORT = True
DEVICE = 'cuda'


def save_model_as_onnx(model, output_path, input_tensor_size=(1, 3, 512, 512)):
    """
    Saves the model in ONNX format.

    Parameters:
        model (nn.Module): The trained model.
        input_tensor_size (tuple, optional): The size of the input tensor. Defaults to (1, 3, 512, 512).
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
            torch.unsqueeze(image_tensor, 0), self.input_size, mode="bilinear").type(torch.float32)
        image_tensor = torch.divide(image_tensor, torch.max(image_tensor)).type(torch.float32)
        image_tensor = image_tensor.to(DEVICE)
        img_result = self.net(image_tensor)
        img_result = img_result[0][:, 0, :, :]
        result_array = img_result.cpu().data.numpy()

        # Norm the prediction
        re_max = np.max(result_array)
        re_min = np.min(result_array)
        result_array = (result_array - re_min) / (re_max - re_min)
        result_array = np.squeeze(result_array)

        alpha_channel = np.uint8(result_array * 255)
        alpha_channel = cv2.resize(alpha_channel, img.size, interpolation=cv2.INTER_LANCZOS4)
        alpha_channel = cv2.GaussianBlur(alpha_channel, (9, 9), 0)
        if mask_only:
            return alpha_channel
        return np.dstack((np.array(img), alpha_channel))


if __name__ == '__main__':
    start = time.perf_counter()
    # Quant
    filename = "./checkpoint.pth_102.tar"
    output = "./u2net_quant_qnnpack.onnx"
    output_torch = "./u2net_quant_qnnpack.pth"
    # 1024 for U2Net, 512 for U2NetP
    size = 1024

    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)

    net = U2NET(3, 1)
    net.load_state_dict(checkpoint["state"]["state_dict"])
    net.eval()

    # net.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # fbgemm or qnnpack
    qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    # qconfig_mapping = {
    #     torch.nn.Conv2d: torch.ao.quantization.get_default_qconfig('x86'),
    #     torch.nn.BatchNorm2d: torch.ao.quantization.get_default_qconfig('x86'),
    # }
    qconfig_mapping = (
        torch.ao.quantization.QConfigMapping()
        .set_global(None)  # start with nothing quantized
        .set_object_type(torch.nn.Conv2d, qconfig)
        .set_object_type(torch.nn.BatchNorm2d, qconfig)
        # catch any nested modules named “conv” or “bn”
        .set_module_name_regex(r".*conv.*", qconfig)
        .set_module_name_regex(r".*bn.*", qconfig)
    )
    reb_attrs = ["conv_s1", "bn_s1"]  # , "relu_s1"
    module_list = (
            [[f"stage1.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage1.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 8)] +
            [[f"stage1.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 7)] +
            [[f"stage2.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage2.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 7)] +
            [[f"stage2.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 6)] +
            [[f"stage3.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage3.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 6)] +
            [[f"stage3.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 5)] +
            [[f"stage4.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage4.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 5)] +
            [[f"stage4.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 4)] +
            [[f"stage5.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage5.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 5)] +
            [[f"stage5.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 4)] +
            [[f"stage6.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage6.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 5)] +
            [[f"stage6.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 4)] +
            [[f"stage1d.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage1d.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 8)] +
            [[f"stage1d.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 7)] +
            [[f"stage2d.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage2d.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 7)] +
            [[f"stage2d.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 6)] +
            [[f"stage3d.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage3d.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 6)] +
            [[f"stage3d.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 5)] +
            [[f"stage4d.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage4d.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 5)] +
            [[f"stage4d.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 4)] +
            [[f"stage5d.rebnconvin.{n}" for n in reb_attrs]] +
            [[f"stage5d.rebnconv{n}.{m}" for m in reb_attrs] for n in range(1, 5)] +
            [[f"stage5d.rebnconv{n}d.{m}" for m in reb_attrs] for n in range(1, 4)]
    )
    net_fused = torch.ao.quantization.fuse_modules(
        net, module_list, inplace=False
    )
    for group in module_list:
        for layer_name in group:
            qconfig_mapping = qconfig_mapping.set_module_name(layer_name, qconfig)
            name_split = layer_name.split('.')
            net_fused.__getattr__(name_split[0]).__getattr__(name_split[1]).__getattr__(name_split[2]).qconfig = qconfig
    for layer_name in [f"side{n}" for n in range(1, 7)]:
        net_fused.__getattr__(layer_name).qconfig = qconfig
    net_fused.outconv.qconfig = qconfig

    dummy_tensor = (torch.randn((1, 3, size, size), dtype=torch.float32))
    net_prepared = prepare_fx(net_fused, qconfig_mapping=qconfig_mapping, example_inputs=dummy_tensor)
    # net_prepared = torch.ao.quantization.prepare(net_fused, qconfig_mapping)

    session = TorchSession(net_prepared, input_size=[size, size])

    # Test
    image_list = [x.path for x in os.scandir("./images")][:1134]
    mask_list = [x.path for x in os.scandir("./masks")][:1134]

    original_u2net_delta_list: List[float] = []

    for image_path, mask_path in tqdm(
            zip(image_list, mask_list),
            desc="Calibration",
            total=len(image_list)
    ):
        pil_image = Image.open(image_path)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert(mode="RGB")
        numpy_image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Test u2net
        u2net_prediction = np.array(session.remove(pil_image, mask_only=True, size=(size, size)))
        u2net_difference = iou(u2net_prediction, mask)
        original_u2net_delta_list.append(u2net_difference)

    # session.net = torch.ao.quantization.convert(session.net.to('cpu'))  #.to(DEVICE)
    session.net = convert_fx(session.net.to('cpu'), qconfig_mapping=qconfig_mapping)

    if EXPORT:
        print("Exporting...")
        save_model_as_onnx(session.net, output_path=output, input_tensor_size=(1, 3, size, size))
        torch.save(session.net.state_dict(), output_torch)

    # session.net = session.net.to(DEVICE)
    DEVICE = 'cpu'

    u2net_delta_list: List[float] = []

    for image_path, mask_path in tqdm(
            zip(image_list, mask_list),
            desc="Inferencing",
            total=len(image_list)
    ):
        pil_image = Image.open(image_path)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert(mode="RGB")
        numpy_image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Test u2net
        u2net_prediction = np.array(session.remove(pil_image, mask_only=True, size=(size, size)))
        u2net_difference = iou(u2net_prediction, mask)
        u2net_delta_list.append(u2net_difference)

    end = time.perf_counter()

    print(f"U2NETP mIoU: {sum(u2net_delta_list) / len(image_list):.10f}\t"
          f"(Originally: {sum(original_u2net_delta_list) / len(image_list):.10f})")
    print(f"Time: {end - start:.2f}s")

