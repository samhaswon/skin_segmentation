from PIL import Image, ImageOps
from PIL.Image import Image as PILImage  # For typing
import onnxruntime as ort
from typing import Dict, Tuple, Union
import numpy as np
import cv2

"""
This is modified from rembg to support multiple sizes and be much more efficient for my use case.
Basically, I removed a lot of conversions and made this compatible with my models.
"""


class Session:
    def __init__(
            self,
            model_path: str,
            sess_opts: Union[ort.SessionOptions, None] = None,
            providers=None
    ):
        """Initialize an instance of the BaseSession class."""

        self.providers = []

        _providers = ort.get_available_providers()
        if providers:
            for provider in providers:
                if provider in _providers:
                    self.providers.append(provider)
        else:
            self.providers.extend(_providers)

        self.inner_session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=sess_opts,
        )

    def normalize(
            self,
            img: Image,
            size: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        im = img.convert("RGB").resize(size, Image.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / np.max(im_ary)

        tmp_img = im_ary.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmp_img, 0)
            .astype(np.float32)
        }

    def predict(self, img: PILImage, size: Tuple[int, int] = (320, 320)) -> np.ndarray:
        """
        Predicts the output mask for the input image using the loaded model.
        :param img: The image to inference on.
        :param size: The prediction size.
        :return: Prediction mask (numpy.ndarray)
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, size),
        )

        prediction = ort_outs[0][:, 0, :, :]

        ma = np.max(prediction)
        mi = np.min(prediction)

        prediction = (prediction - mi) / (ma - mi)
        prediction = np.squeeze(prediction)

        mask = cv2.resize(
            (prediction * 255).astype("uint8"),
            img.size,
            interpolation=cv2.INTER_LANCZOS4)

        return mask

    @staticmethod
    def fix_image_orientation(img: PILImage) -> PILImage:
        """
        Fix the orientation of the image based on its EXIF data.
        :param img: The image to be fixed.
        :returns: PILImage: The fixed image.
        """
        return ImageOps.exif_transpose(img)

    @staticmethod
    def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
        """
        Perform a simple cutout operation on an image using a mask.
        :param img: Base image to cut out from.
        :param mask: Mask to use to cut out from the base image. A.k.a., the alpha channel.
        """
        empty = Image.new("RGBA", img.size, 0)
        cutout = Image.composite(img, empty, mask)
        return cutout

    @staticmethod
    def post_process(mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.GaussianBlur(mask, (9, 9), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
        return mask

    def remove(
            self,
            img: PILImage,
            size: Tuple[int, int] = (320, 320),
            mask_only: bool = False
    ) -> PILImage:
        """
        Segment an input image.
        :param img: Image to segment.
        :param size: Inferencing size.
        :param mask_only: Whether to return only the mask or a cutout.
        :return: Segmented image or mask
        """

        # Fix image orientation
        img = self.fix_image_orientation(img)

        mask = self.predict(img, size)

        mask = Image.fromarray(self.post_process(mask), mode="L")
        if mask_only:
            cutout = mask
        else:
            cutout = self.naive_cutout(img, mask)

        return cutout
