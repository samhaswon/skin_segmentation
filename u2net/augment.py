"""
This is slow. I just didn't feel like it was worth parallelizing it. 
"""
import os
import cv2
import numpy as np
import shutil
from typing import Union

# Image number to begin with for augmentation (inclusive)
augment_num = 315


def number_img_name(num: int) -> str:
    """
    Format an image number to 5 decimal points, assuming that the image is a png.
    :param num: The image's number to be formatted.
    :return: The image filename.
    """
    return f"{num:05}.png"


def save_img(image_in: np.ndarray, filename: str) -> None:
    """
    Saves an image's augment, duplicating its mask at the same time.
    :param image_in: Image to save.
    :param filename: The name of the original file. This is used to find the mask to copy.
    :return: None
    """
    global augment_num  # For scoping

    # Make the new image's name
    image_name = number_img_name(augment_num)

    # Write the augmented image
    cv2.imwrite(f"./image/{image_name}", image_in)

    # Copy the mask for the augmented image
    shutil.copy(f"./mask/{filename}", f"./mask/{image_name}")

    # Increment the number for augmented images.
    augment_num += 1


def contrast(image_in: np.ndarray, alpha: Union[int, float], beta: int):
    """
    Adjusts the brightness and contrast of an image as follows:
    alpha 1 beta 0 --> no change
    0 < alpha < 1 --> lower contrast
    alpha > 1 --> higher contrast
    -127 < beta < +127 --> good range for brightness values
    :param image_in: Image to adjust the contrast of.
    :param alpha: The brightness scale factor.
    :param beta: The optional delta added to the scaled values, the contrast.
    :return: The adjusted image.
    """
    return cv2.convertScaleAbs(image_in, alpha=alpha, beta=beta)


def desaturation(image_in: np.ndarray, amount: float) -> np.ndarray:
    """
    Desaturates an image (or saturates) by the given amount.
    :param image_in: The image to adjust the saturation of.
    :param amount: The amount to adjust the saturation by. If amount > 1.0, saturation is increased. If amount < 1.0,
    saturation is decreased.
    :return: Image with an adjusted saturation channel.
    """
    hsv_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1] * amount
    saturation[np.where(saturation > 255)] = 255
    hsv_image[:, :, 1] = saturation
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def hue_rotation(image_in: np.ndarray, rotation: int = 30) -> np.ndarray:
    """
    Rotates the hue of a given image by rotation degrees.
    :param image_in: Image to rotate the colors of.
    :param rotation: Amount (default 30) to rotate the hue by.
    :return: Image with rotated colors.
    """
    assert 0 <= rotation <= 180  # make sure the rotation is sensible
    hsv_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    hue_channel = np.add(hue_channel, rotation)  # 0 is no change; 0 <= hue change <= 180
    hsv_new = cv2.merge([hue_channel, hsv_image[:, :, 1], hsv_image[:, :, 2]])
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


def sepia(image_in: np.ndarray, k: float) -> np.ndarray:
    """
    Applies a sepia filter to the image.
    :param image_in: The image to apply the filter to.
    :param k: The intensity of the sepia effect.
    :return: Sepia filtered image.
    """
    sepia_image = cv2.transform(image_in,
                                np.matrix(
                                    [[0.393 + 0.607 * (1 - k), 0.769 - 0.769 * (1 - k), 0.189 - 0.189 * (1 - k)],
                                     [0.349 - 0.349 * (1 - k), 0.686 + 0.314 * (1 - k), 0.168 - 0.168 * (1 - k)],
                                     [0.272 - 0.349 * (1 - k), 0.534 - 0.534 * (1 - k), 0.131 + 0.869 * (1 - k)]]
                                ))
    sepia_image[np.where(sepia_image > 255)] = 255
    sepia_image = cv2.convertScaleAbs(sepia_image, alpha=1, beta=0)
    return sepia_image


if __name__ == "__main__":
    image_list = [x for x in os.listdir("./image")]
    mask_list = [x for x in os.listdir("./mask")]

    print(f"Found {len(image_list)} images.")

    for image_path, mask_path in zip(image_list, mask_list):
        image = cv2.imread(f"./image/{image_path}")
        mask = cv2.imread(f"./mask/{mask_path}")

        # Hue augmentation
        result = image
        for _ in range(6):
            result = hue_rotation(result)
            save_img(result, filename=image_path)

        # Greyscale
        save_img(desaturation(image, 0.5), filename=image_path)
        save_img(desaturation(image, 0.3), filename=image_path)
        save_img(desaturation(image, 0.0), filename=image_path)

        # Increase saturation
        save_img(desaturation(image, 1.2), filename=image_path)
        save_img(desaturation(image, 1.5), filename=image_path)

        # Brightness and Contrast
        save_img(contrast(image, alpha=1, beta=5), filename=image_path)
        save_img(contrast(image, alpha=1.5, beta=0), filename=image_path)
        save_img(contrast(image, alpha=0.7, beta=5), filename=image_path)
        save_img(contrast(image, alpha=0.7, beta=0), filename=image_path)
        save_img(contrast(image, alpha=1.5, beta=5), filename=image_path)

        # Sepia
        for i in range(0, 8):
            save_img(sepia(image, i * 0.25), filename=image_path)
