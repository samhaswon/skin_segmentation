# Importing Required Modules
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage     # For typing
import onnxruntime as ort
from typing import Dict, Tuple, Union, Optional, List
import numpy as np
import cv2


class BiRefNetSession:
    def __init__(
        self,
        model_path: str,
        sess_opts: Optional[ort.SessionOptions] = None,
        providers: Optional[List[str]]=None,
        use_fp16: bool = False,
        threshold: Optional[float] = None,
    ):
        """Initialize an instance of the Session class."""

        self.providers = []
        self.use_fp16 = use_fp16
        self.threshold = threshold

        _providers = ort.get_available_providers()
        if providers:
            for provider in providers:
                if provider in _providers:
                    self.providers.append(provider)
        else:
            self.providers.extend(_providers)

        if sess_opts is None:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.inner_session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=sess_opts,
        )

    def normalize(
        self,
        img: Image,
        size: Union[Tuple[int, int], None] = None,
        convert_to: str = "RGB"
    ) -> Tuple[Dict[str, np.ndarray], int, int]:
        """
        Normalize and prepare the image for inferencing.
        :param img: The (PIL) Image to repair.
        :param size: The size for inferencing (if not index 2 and 3 of the input shape).
        :param convert_to: The color format to convert the image into (if it even matters).
        :return: The prepared image for ONNX inferencing.
        """
        if size is None:
            size = self.inner_session.get_inputs()[0].shape[2], self.inner_session.get_inputs()[0].shape[3]

        w, h = img.size
        if h >= w:
            new_h = size[0]
            new_w = max(1, int(round(w * size[0] / h)))
        else:
            new_w = size[0]
            new_h = max(1, int(round(h * size[0] / w)))

        im = img.convert(convert_to).resize((new_w, new_h), Image.LANCZOS)

        im_ary = np.array(im).astype(np.float32) / 255.0

        pad_h = size[0] - new_h
        pad_w = size[0] - new_w

        tmp_img = np.pad(im_ary, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        tmp_img = tmp_img.transpose((2, 0, 1))

        if self.use_fp16:
            tmp_img = tmp_img.astype(np.float16)

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmp_img, 0)
        }, pad_h, pad_w

    def predict(self, img: PILImage, size: Union[Tuple[int, int], None] = None) -> np.ndarray:
        """
        Predicts the output mask for the input image using the loaded model.
        :param img: The image to inference on.
        :param size: The prediction size (if not index 2 and 3 of the input shape).
        :return: Prediction mask (numpy.ndarray)
        """
        ort_inputs, pad_h, pad_w = self.normalize(img, size)
        ort_outs = self.inner_session.run(
            None,
            ort_inputs,
        )

        prediction = ort_outs[0]

        prediction = np.squeeze(prediction)
        prediction = prediction[:prediction.shape[0] - pad_h, :prediction.shape[1] - pad_w]
        # prediction = (prediction > 0.5).astype(np.float32)

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
        """
        Morphs and blurs the mask to make it a bit better (generally speaking).
        :param mask: The mask to post-process.
        :return: The post-processed mask.
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
        return mask

    def remove(
            self,
            img: PILImage,
            size: Union[Tuple[int, int], None] = None,
            mask_only: bool = False,
    ) -> PILImage:
        """
        Segment an input image.
        :param img: Image to segment.
        :param size: The inferencing size (if not index 2 and 3 of the input shape).
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
