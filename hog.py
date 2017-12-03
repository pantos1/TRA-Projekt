import numpy
import cv2

#TODO: Preprocessing of image - scaling to aspect ratio of 1:2 eg. 64x128

# Read image
img = cv2.imread('bolt.png')
img = numpy.float32(img) / 255.0

# Calculate gradient
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Calculate gradient magnitude and direction ( in degrees )
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
