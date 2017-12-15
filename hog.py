import numpy
import cv2
import matplotlib.pyplot as plt
import json

# Wlasna implementacja

# Read image
img = cv2.imread('bolt.jpg')
img = numpy.float32(img) / 255.0

# Calculate gradient
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Calculate gradient magnitude and direction ( in degrees )
magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
angle = abs(angle - 180)
# plt.figure()
# plt.subplot(143)
# plt.imshow(magnitude, cmap=plt.cm.Greys_r)
# plt.subplot(144)
# plt.imshow(angle, cmap=plt.cm.Greys_r)
# plt.show()

bin_number = 8
bin = numpy.int32(bin_number*angle/180)
#Array of cells of size 8x8 pixels used to calculate histogram
bin_cells = []
magnitude_cells = []
#Size of cell for histogram
cell_x = cell_y = 8
#Picture size in pixels
y_size = img.shape[0]
x_size = img.shape[1]
n_cell_x, n_cell_y = int(x_size/cell_x), int(y_size/cell_y)
for i in range(0, n_cell_y):
    for j in range(0, n_cell_x):
        bin_cells.append(bin[i*cell_y : (i+1)*cell_y, j*cell_x : (j+1)*cell_x])
        magnitude_cells.append(magnitude[i * cell_y: (i + 1) * cell_y, j * cell_x: (j + 1) * cell_x])
# Advanced version with weigthed contribution
# histograms = [numpy.histogram(b.ravel(), bin_number+1, range = (0,8), weights = m.ravel()) for b,m in zip(bin_cells, magnitude_cells)]

#Simpler version
histograms = [numpy.bincount(b.ravel(), weights=m.ravel(), minlength=bin_number) for b, m in zip(bin_cells, magnitude_cells)]
histogram = numpy.hstack(histograms)
#Nie działa!!!
# hist_2d = numpy.array(histograms, dtype=float).reshape(n_cell_y, n_cell_x)
plt.figure()
# plt.imshow(hist_2d, cmap=plt.cm.Greys_r)
plt.show()
#Normalization
eps = 1e-7
histogram = numpy.sqrt(histogram/(histogram.sum()+eps))


cv2.imshow("Picture", img)

