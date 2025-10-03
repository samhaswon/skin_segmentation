import cv2
import numpy as np


def nothing(x):
    pass


# Load the image
image = cv2.imread('images/000k9gtp50b3kj06bikpg230de34c.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
mean, std_dev = cv2.meanStdDev(gray)

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
cv2.createTrackbar('HueMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HueMax', 'image', 20, 255, nothing)
cv2.createTrackbar('SatMin', 'image', 48, 255, nothing)
cv2.createTrackbar('SatMax', 'image', 255, 255, nothing)
cv2.createTrackbar('ValMin', 'image', 80, 255, nothing)
cv2.createTrackbar('ValMax', 'image', 255, 255, nothing)


cv2.createTrackbar('Blur', 'image', 3, 255, nothing)
cv2.createTrackbar('Morph', 'image', 1, 100, nothing)

# Create trackbars for RGB adjustments
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

while True:
    # Get current positions of the RGB trackbars
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')

    # Get current positions of the HSV trackbars
    h_min = cv2.getTrackbarPos('HueMin', 'image')
    h_max = cv2.getTrackbarPos('HueMax', 'image')
    s_min = cv2.getTrackbarPos('SatMin', 'image')
    s_max = cv2.getTrackbarPos('SatMax', 'image')
    v_min = cv2.getTrackbarPos('ValMin', 'image')
    v_max = cv2.getTrackbarPos('ValMax', 'image')

    blur = cv2.getTrackbarPos('Blur', 'image')
    morph = cv2.getTrackbarPos('Morph', 'image')

    if not blur % 2:
        blur += 1

    # Adjust RGB channels
    adjusted = cv2.merge([
        cv2.subtract(image[:, :, 0], np.uint8([b])),
        cv2.subtract(image[:, :, 1], np.uint8([g])),
        cv2.subtract(image[:, :, 2], np.uint8([r]))
    ])

    # convert it to the HSV color space
    converted_hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

    # Set the lower_hsv and upper_hsv HSV range according to the trackbar values
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    skinMask = hsv_skin_mask = cv2.inRange(converted_hsv, lower_hsv, upper_hsv)

    skinMask = cv2.inRange(converted_hsv, lower_hsv, upper_hsv)
    if not morph:
        kernel = None
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask, (blur, blur), 0)
    skin = cv2.bitwise_and(image, image, mask=skinMask)

    display_size = (1920 // 2, 1080)
    cv2.imshow('mask', np.hstack([cv2.resize(image, display_size), cv2.resize(skin, display_size)]))

    # Wait for the 'esc' key to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        alpha = np.sum(skin, axis=-1) > 0
        alpha = np.uint8(alpha * 255)
        result = np.dstack((skin, alpha))

        cv2.imwrite("tests/result.png", result)
        break

# Close all windows
cv2.destroyAllWindows()
