import cv2
import numpy as np

# Load and convert the image
imagePath = 'images/000k9gtp50b3kj06bikpg230de34c.jpg'
img = cv2.imread(imagePath)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = 5
morph = 1
within_sd = 1.2

# Load the classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Find faces
face = face_classifier.detectMultiScale(
    gray_image,  # Input image
    scaleFactor=1.1,  # Scale down the image be a certain factor
    minNeighbors=5,  # Factor to reduce false positives
    minSize=(40, 40)  # Minimum size of a face
)

# Focus on the face
rect = face[0]
inner = img[int(rect[1]):int((rect[1] + rect[3])),
            int(rect[0] + (rect[2] - rect[0])):int(rect[0] + rect[2])]

# Colorama
hsv_image = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)

# Calculate mean and standard deviation of each channel separately
h_mean, h_stddev = cv2.meanStdDev(hsv_image[:, :, 0])  # Hue channel
s_mean, s_stddev = cv2.meanStdDev(hsv_image[:, :, 1])  # Saturation channel
v_mean, v_stddev = cv2.meanStdDev(hsv_image[:, :, 2])  # Value channel

# Extract scalar values from arrays
h_mean = h_mean[0][0]
h_stddev = h_stddev[0][0]
s_mean = s_mean[0][0]
s_stddev = s_stddev[0][0]
v_mean = v_mean[0][0]
v_stddev = v_stddev[0][0]

# Define the ranges within 1 standard deviation of the mean for each channel
h_range = (int(h_mean - h_stddev * within_sd), int(h_mean + h_stddev * within_sd))
s_range = (int(s_mean - s_stddev * within_sd), int(s_mean + s_stddev * within_sd))
v_range = (int(v_mean - v_stddev * within_sd), int(v_mean + v_stddev * within_sd))

print("Hue Range:", h_range)
print("Saturation Range:", s_range)
print("Value Range:", v_range)

lower_hsv = np.array([h_range[0] if 10 > h_range[0] >= 0 else 0,
                      s_range[0] if s_range[0] >= 20 else 20,
                      v_range[0] if v_range[0] >= 60 else 60])
upper_hsv = np.array([h_range[1] if h_range[1] <= 50 else 50,
                      s_range[1] if s_range[1] <= 255 else 255,
                      v_range[1] if v_range[1] <= 255 else 255])

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Make the mask
skinMask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
if not morph:
    kernel = None
else:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
if not blur % 2:
    blur += 1
skinMask = cv2.erode(skinMask, kernel, iterations=2)
skinMask = cv2.dilate(skinMask, kernel, iterations=2)
skinMask = cv2.GaussianBlur(skinMask, (blur, blur), 0)
skin = cv2.bitwise_and(img, img, mask=skinMask)

# Remove the surrounding pixels
alpha = np.sum(skin, axis=-1) > 0
alpha = np.uint8(alpha * 255)
result = np.dstack((skin, alpha))

# Write the result
cv2.imwrite("test_auto.png", result)
