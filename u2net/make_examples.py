from session import Session
from PIL import Image


if __name__ == '__main__':
    image_path_list = [
        ("../examples/S2563713746_St25_G7.5.jpeg", "../examples/woman.png"),
        ("../examples/Will-Smith-new-headshot-credit-Lorenzo-Agius.jpg", "../examples/ws.png"),
        ("../examples/zuck_original.jpg", "../examples/zuck.png")
        ]
    
    session = Session("../skin_u2net.onnx")

    for input_image, output_image in image_path_list:
        image_in = Image.open(input_image)
        output = session.remove(image_in)
        output.save(output_image)
