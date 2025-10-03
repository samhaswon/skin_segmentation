########################################################################
# This script statically augments the dataset with various color based
# transforms. It's fairly slow and even when parallelized, but it
# works. If you have a significant dataset, just start this and go
# make some coffee. cv2 is fairly fast, but the 25 different augments
# done by this script are *very* slow.
########################################################################
import math
import multiprocessing
import os
import random
import shutil
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def number_img_name(num: int) -> str:
    """
    Format an image number to 5 decimal points, assuming that the image is a png.
    :param num: The image's number to be formatted.
    :return: The image filename.
    """
    return f"{num:05}.png"


def save_img(image_in: np.ndarray, filename: str, augment_num: int) -> None:
    """
    Saves an image's augment, duplicating its mask at the same time.
    :param image_in: Image to save.
    :param filename: The name of the original file. This is used to find the mask to copy.
    :return: None
    """

    # Make the new image's name
    image_name = number_img_name(augment_num)

    # Write the augmented image
    cv2.imwrite(f"./images/{image_name}", image_in, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # Copy the masks for the augmented image
    shutil.copy(f"./masks/{filename}", f"./masks/{image_name}")


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


def generate_rotated_lattice(
        width: int,
        height: int,
        spacing: float,
        randomness: float = 0.0,
        line_width: int = 1,
        rotate_degrees: int = 45,
) -> Image.Image:
    """
    Generate a 45°(rotate_degrees)-rotated lattice (white lines on black) with full coverage.

    :param width: The final image width in pixels.
    :param height: The final image height in pixels.
    :param spacing: The nominal distance between adjacent lines (perpendicular).
    :param randomness: The max jitter (±pixels) added to each line’s position.
    :param line_width: The thickness of the grid lines.
    :param rotate_degrees: The number of degrees (int) to rotate the grid by.
    :return: PIL.Image (mode 'L').
    """
    # build a base canvas large enough that rotating covers all edges
    diag = int(math.hypot(width, height))
    big = diag + int(spacing * 2)
    base = Image.new('L', (big, big), 0)
    draw = ImageDraw.Draw(base)

    # draw verticals (will become one diagonal after rotation)
    x = -big
    while x <= big * 2:
        jittered_x = x + random.uniform(-randomness, randomness)
        draw.line(((jittered_x, 0), (jittered_x, big)), fill=255,
                  width=line_width)
        x += spacing

    # draw horizontals (become the other diagonal)
    y = -big
    while y <= big * 2:
        jittered_y = y + random.uniform(-randomness, randomness)
        draw.line(((0, jittered_y), (big, jittered_y)), fill=255,
                  width=line_width)
        y += spacing

    # rotate by 45° (or rotate_degrees) and crop center to target size
    rotated = base.rotate(rotate_degrees, resample=Image.BILINEAR, expand=True)
    cx, cy = rotated.size[0] // 2, rotated.size[1] // 2

    left = cx - width // 2
    top = cy - height // 2
    right = left + width
    bottom = top + height

    return rotated.crop((left, top, right, bottom))


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


def vertical_fill(m, n, k):
    """
    Create an m x n matrix with 0s from row 0 up to (but not including) row k,
    and 255 afterward.

    :param m: Number of rows
    :param n: Number of columns
    :param k: Cutoff row index
    :return: numpy.ndarray of shape (m, n)
    """
    mat = np.zeros((m, n), dtype=np.uint8)
    mat[k:, :] = 255
    return mat


def augment_image(img_path: str, msk_path: str, augment_num: int):
    image = cv2.imread(f"./images/{img_path}")
    mask = cv2.imread(f"./masks/{msk_path}")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Hue augmentation
    result = image
    for _ in range(6):
        result = hue_rotation(result)
        save_img(result, filename=img_path, augment_num=augment_num)
        augment_num += 1

    # Greyscale
    save_img(desaturation(image, 0.5), filename=img_path, augment_num=augment_num)
    augment_num += 1
    save_img(desaturation(image, 0.3), filename=img_path, augment_num=augment_num)
    augment_num += 1
    save_img(desaturation(image, 0.0), filename=img_path, augment_num=augment_num)
    augment_num += 1

    # Increase saturation
    save_img(desaturation(image, 1.2), filename=img_path, augment_num=augment_num)
    augment_num += 1
    save_img(desaturation(image, 1.5), filename=img_path, augment_num=augment_num)
    augment_num += 1

    # Brightness and Contrast
    save_img(contrast(image, alpha=1, beta=5), filename=img_path, augment_num=augment_num)
    augment_num += 1
    save_img(contrast(image, alpha=1.5, beta=0), filename=img_path, augment_num=augment_num)
    augment_num += 1
    save_img(contrast(image, alpha=0.7, beta=5), filename=img_path, augment_num=augment_num)
    augment_num += 1
    save_img(contrast(image, alpha=0.7, beta=0), filename=img_path, augment_num=augment_num)
    augment_num += 1
    save_img(contrast(image, alpha=1.5, beta=5), filename=img_path, augment_num=augment_num)
    augment_num += 1

    # Sepia
    for i in range(0, 8):
        save_img(sepia(image, i * 0.25), filename=img_path, augment_num=augment_num)
        augment_num += 1

    lattice = generate_rotated_lattice(
        width=image.shape[1],
        height=image.shape[0],
        spacing=20,
        randomness=3,
        line_width=2,
        rotate_degrees=random.randint(0, 90)
    )
    lattice = np.array(lattice)
    lattice = np.bitwise_and(lattice, mask)
    augment_mask = cv2.subtract(mask, lattice)
    cv2.imwrite(
        f"./masks/{number_img_name(augment_num)}",
        augment_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    lattice = np.dstack((lattice, lattice, lattice))
    cv2.imwrite(
        f"./images/{number_img_name(augment_num)}",
        cv2.subtract(image, lattice),
        [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    augment_num += 1

    lattice = generate_rotated_lattice(
        width=image.shape[1],
        height=image.shape[0],
        spacing=40,
        randomness=3,
        line_width=2,
        rotate_degrees=random.randint(0, 90)
    )
    lattice_mask = vertical_fill(
        image.shape[0],
        image.shape[1],
        random.randint(image.shape[0] // 4, image.shape[0] // 2)
    )
    lattice = np.array(lattice)
    lattice = np.bitwise_and(lattice, lattice_mask)
    lattice = np.bitwise_and(lattice, mask)
    augment_mask = cv2.subtract(mask, lattice)
    cv2.imwrite(
        f"./masks/{number_img_name(augment_num)}",
        augment_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    lattice = np.dstack((lattice, lattice, lattice))
    cv2.imwrite(
        f"./images/{number_img_name(augment_num)}",
        cv2.subtract(image, lattice),
        [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )


if __name__ == "__main__":
    augment_count = 26
    image_list = [x for x in os.listdir("./images") if x.endswith("png")]  # [:4]
    mask_list = [x for x in os.listdir("./masks") if x.endswith("png")]  # [:4]

    print(f"Found {len(image_list)} images.")

    progress_bar = tqdm(total=len(image_list))
    
    augment_num = max(len(image_list) + 1, int(image_list[-1][:-4]) + 1)
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    def inc_pbar(a):
        progress_bar.update(1)

    for image_path, mask_path in zip(image_list, mask_list):
        pool.apply_async(augment_image, args=(image_path, mask_path, augment_num,), callback=inc_pbar)
        augment_num += augment_count

    # Start, do the work, and wait for results
    pool.close()
    pool.join()

    progress_bar.close()
