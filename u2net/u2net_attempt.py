from session import Session
from PIL import Image


if __name__ == '__main__':
    # Replace with the path to the checkpoint you want to use,
    # u2net, u2netp, or dlmv
    session = Session(model_path="/path/to/u2net.onnx")
    image_in = Image.open("images/000k9gtp50b3kj06bikpg230de34c.jpg")   # Replace with the path to your image
    session.remove(image_in).save("result.png")
