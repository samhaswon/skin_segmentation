from session import Session
import numpy as np
import cv2
from PIL import Image


INFERENCE_SIZE = 1024   # Inference size of the chosen model


if __name__ == '__main__':
    session = Session(model_path="D:\\skin_models\\5796.onnx")  # Replace with the path to the checkpoint
    image_in = Image.open("images/000k9gtp50b3kj06bikpg230de34c.jpg")   # Replace with the path to your image
    result = np.array(session.remove(image_in, (INFERENCE_SIZE, INFERENCE_SIZE)))

    alpha = np.sum(result, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    result = np.dstack((result, alpha))

    cv2.imwrite("result.png", result)
