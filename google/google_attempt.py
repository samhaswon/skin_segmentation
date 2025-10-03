import mediapipe as mp
from mediapipe.tasks import python
import numpy as np
import cv2


BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a image segmenter instance with the image mode:
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='models/selfie_multiclass_256x256.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True)
with ImageSegmenter.create_from_options(options) as segmenter:
    # Load the input image from a numpy array.
    numpy_image = cv2.imread('images/000k9gtp50b3kj06bikpg230de34c.jpg')
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    # Segment the image
    segmented_masks = segmenter.segment(mp_image)

    # Get the prediction of each skin class and combine them.
    face_skin = segmented_masks.confidence_masks[3]
    body_skin = segmented_masks.confidence_masks[2]
    np_face_skin = np.uint8(face_skin.numpy_view().copy() * 255)
    np_body_skin = np.uint8(body_skin.numpy_view().copy() * 255)
    mask = np.bitwise_or(np_body_skin, np_face_skin)

    # Post process the mask to make this model look at least a little bit better
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered = cv2.inRange(mask, np.array([150]), np.array([255]))
    skinMask = cv2.erode(filtered, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Write the mask
    cv2.imwrite("mask.png", mask)

    # Make the cutout the simple way
    result = cv2.bitwise_and(numpy_image, numpy_image, mask=skinMask)

    # Scale the alpha channel for saving
    alpha = np.sum(result, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    result = np.dstack((result, alpha))

    # Save the cutout
    cv2.imwrite("cutout.png", result)
