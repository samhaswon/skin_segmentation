import cv2
import numpy as np
from PIL.Image import Image as PILImage     # For typing
import torch
import torch.nn.functional as F
from typing import Union, Tuple
from .u2net import U2NET, U2NETP


class U2NetTorchSession:
    def __init__(self, half_precision: bool = False, use_small: bool = False, device: str = None):
        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
        if use_small:
            self.net = U2NETP(3, 1)
            model_path = "models/u2netp.pth"
            self.input_size = [512, 512]
        else:
            self.net = U2NET(3, 1)
            model_path = "models/u2net.pth"
            self.input_size = [1024, 1024]
        if torch.cuda.is_available() and self.device != "cpu":
            self.net.load_state_dict(
                torch.load(model_path, weights_only=False)
            )
            self.net.cuda()
        else:
            self.net.load_state_dict(
                torch.load(
                    model_path,
                    map_location=torch.device(self.device),
                    weights_only=False
                )
            )
        self.net.eval()
        self.half_precision = half_precision

    def remove(
            self,
            img: PILImage,
            size: Union[Tuple[int, int], None] = None,
            mask_only: bool = False
    ) -> np.ndarray:
        image_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)
        image_tensor = F.interpolate(
            torch.unsqueeze(image_tensor, 0), self.input_size, mode="bilinear").type(torch.float32)
        image_tensor = torch.divide(image_tensor, torch.max(image_tensor)).type(torch.float32)
        image_tensor = image_tensor.to(self.device)
        with (torch.no_grad(),
              torch.autocast(
                  device_type=self.device,
                  dtype=torch.float16,
                  enabled=self.half_precision)
              ):
            img_result = self.net(image_tensor)
        img_result = img_result[0][0, 0, :, :]
        img_result = torch.nan_to_num(img_result)

        # Norm the prediction
        re_max = torch.max(img_result)
        re_min = torch.min(img_result)
        result_array = (img_result - re_min) / (re_max - re_min) * 255

        alpha_channel = np.uint8(result_array.cpu().data.numpy())
        alpha_channel = cv2.resize(alpha_channel, img.size, interpolation=cv2.INTER_LANCZOS4)
        alpha_channel = cv2.GaussianBlur(alpha_channel, (3, 3), 0)
        if mask_only:
            return alpha_channel
        return np.dstack((np.array(img), alpha_channel))
