from session import Session
from PIL import Image


if __name__ == '__main__':
    session = Session(model_path="D:\\skin_models\\5796.onnx")  # Replace with the path to the checkpoint
    image_in = Image.open("images/000k9gtp50b3kj06bikpg230de34c.jpg")   # Replace with the path to your image
    session.remove(image_in).save("result.png")
