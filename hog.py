import numpy
import cv2

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
bin_number = 9
bin = numpy.int32(bin_number*angle/180)
#Array of cells of size 8x8 pixels used to calculate histogram
bin_cells = []
magnitude_cells = []
#Size of cell for histogram
cell_x = cell_y = 8
#Picture size in pixels
y_size = img.shape[0]
x_size = img.shape[1]

for i in range(0, y_size/cell_y):
    for j in range(0, x_size/cell_x):
        bin_cells.append(bin[i*cell_y : (i+1)*cell_y, j*cell_x : (j+1)*cell_x])
        magnitude_cells.append(magnitude[i * cell_y: (i + 1) * cell_y, j * cell_x: (j + 1) * cell_x])
# Advanced version with weidthed contribution
# histograms = [numpy.histogram(b.ravel(), bin_number+1, range = (0,8), weights = m.ravel()) for b,m in zip(bin_cells, magnitude_cells)]

#Simpler version
histograms = [numpy.bincount(b.ravel(), weights=m.ravel(), minlength=bin_number) for b, m in zip(bin_cells, magnitude_cells)]
histogram = numpy.hstack(histograms)
#Normalization
eps = 1e-7
histogram = numpy.sqrt(histogram/(histogram.sum()+eps))

cv2.imshow("Picture", img)

