import numpy
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
import json

# Wlasna implementacja
def hog(img, bin_number = 9, cell_x = 8, cell_y = 8):
    hog_size = cell_x * cell_y * bin_number
    histogram = numpy.zeros((hog_size,1))
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

    #Array of cells of size 8x8 pixels used to calculate histogram
    bin_cells = []
    magnitude_cells = []
    #Picture size in pixels
    y_size = img.shape[0]
    x_size = img.shape[1]
    n_cell_x, n_cell_y = int(x_size/cell_x), int(y_size/cell_y)
    # for i in range(0, n_cell_y):
    #     for j in range(0, n_cell_x):
    #         bin_cells.append(bin[i*cell_y : (i+1)*cell_y, j*cell_x : (j+1)*cell_x])
    #         magnitude_cells.append(magnitude[i * cell_y: (i + 1) * cell_y, j * cell_x: (j + 1) * cell_x])
    bin_range = 360 / bin_number
    bins = (angle[...,0] % 360 / bin_range).astype(int).transpose()
    x, y = numpy.mgrid[:x_size, :y_size]
    x = x / n_cell_x
    y = y / n_cell_y
    labels = (x * cell_x + y) * bin_number + bins
    index = numpy.arange(hog_size)
    magnitude = magnitude[...,0].transpose()
    histogram = scipy.ndimage.measurements.sum(magnitude, labels, index)
    #
    # #Simpler version
    # histograms = [numpy.bincount(b.ravel(), weights=m.ravel(), minlength=bin_number) for b, m in zip(bin_cells, magnitude_cells)]
    # histogram = numpy.hstack(histograms)
    # #Nie działa!!!
    # # hist_2d = numpy.array(histograms, dtype=float).reshape(n_cell_y, n_cell_x)
    # plt.figure()
    # # plt.imshow(hist_2d, cmap=plt.cm.Greys_r)
    # plt.show()
    # #Normalization
    # eps = 1e-7
    # histogram = numpy.sqrt(histogram/(histogram.sum()+eps))
    return histogram

def main():
    # Read image
    img = cv2.imread('bolt.jpg')
    img = numpy.float32(img) / 255.0
    histogram = hog(img)
    plt.figure()
    plt.imshow(histogram, cmap=plt.cm.Greys_r)
    plt.show()

if __name__ == "__main__":
    main()
