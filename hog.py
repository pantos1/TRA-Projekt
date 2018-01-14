import numpy
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib

pos_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/Test/pos'
neg_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/Test/neg'
width = 96
height = 160

filename = 'svc.pkl'

# Wlasna implementacja
def hog(img, bin_number = 9, cell = (8,8), norm_cell=(16,16)):
    histogram =[]
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Calculate gradient magnitude and direction ( in degrees )
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    magnitude = magnitude[...,0]
    angle = angle[...,0]
    angle = abs(angle - 180)
    # plt.figure()
    # plt.subplot(143)
    # plt.imshow(magnitude, cmap=plt.cm.Greys_r)
    # plt.subplot(144)
    # plt.imshow(angle, cmap=plt.cm.Greys_r)
    # plt.show()

    #Array of cells of size 8x8 pixels used to calculate histogram
    angle_cells = []
    magnitude_cells = []
    #Picture size in pixels
    y_size = img.shape[0]
    x_size = img.shape[1]
    n_cell_x, n_cell_y = int(x_size/cell[0]), int(y_size/cell[1])

    # Wersja 1
    #Stworzenie komórek o podanych wymiarach z kątami i amplitudami
    for i in range(0, n_cell_y):
        for j in range(0, n_cell_x):
           angle_cells.append(angle[i*cell[1] : (i+1)*cell[1], j*cell[0] : (j+1)*cell[0]])
           magnitude_cells.append(magnitude[i * cell[1]: (i + 1) * cell[1], j * cell[0]: (j + 1) * cell[0]])
    del magnitude, angle

    #Liczenie histogramu dla każdej komórki
    for mag, ang in zip(magnitude_cells, angle_cells):
        mag = mag.flatten()
        ang = ang.flatten()
        hist, bins = numpy.histogram(ang, bins=bin_number, weights=mag)
        histogram.append(hist)
    del magnitude_cells, angle_cells

    #Zamiana na listę o wymiarach równych liczba komórek do histogramu razy liczba komórek do histogramu razy liczba binów
    histogram = numpy.array(histogram, dtype=float).reshape(n_cell_y, n_cell_x, bin_number)

    #Normalizacja
    norm_cells =[]
    n_norm_cell_x, n_norm_cell_y = int(x_size/norm_cell[0]), int(y_size/norm_cell[1])
    for i in range (0, n_norm_cell_y):
        for j in range(0, n_norm_cell_x):
            block_to_norm = histogram[i*int(norm_cell[1]/cell[1]) : (i+1)*int(norm_cell[1]/cell[1]), j*int(norm_cell[0]/cell[0]) : (j+1)*int(norm_cell[0]/cell[0]), :].flatten()
            norm_cells.append(block_to_norm)
    # Eps użyty, żeby nie było dzielenia przez 0 przy normowaniu
    eps = 1e-7
    #List comprehension, żeby unormować po kolei wszystkie bloki
    normalised_cells = [block/numpy.linalg.norm(block+eps) for block in norm_cells]
    del norm_cells
    #Spłaszczenie listy  w jeden wektor
    histogram = numpy.concatenate(normalised_cells).ravel().tolist()
    del normalised_cells
    return histogram

    # Wersja 2
    #  bin_range = 360 / bin_number
    # bins = (angle[...,0] % 360 / bin_range).astype(int).transpose()
    # x, y = numpy.mgrid[:x_size, :y_size]
    # x = x / n_cell_x
    # y = y / n_cell_y
    # labels = (x * cell_x + y) * bin_number + bins
    # index = numpy.arange(hog_size)
    # magnitude = magnitude[...,0].transpose()
    # histogram = scipy.ndimage.measurements.sum(magnitude, labels, index)

    # Wersja 3
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

def predict(classifier, X):
    return classifier.predict(X)

def load_classifier(filename):
    return joblib.load(filename)

def main():
    # Read image
    pos = [hog(numpy.float32(cv2.resize(cv2.imread(os.path.join(pos_path, image)), (width, height)))) for image in os.listdir(pos_path)]
    neg = [hog(numpy.float32(cv2.resize(cv2.imread(os.path.join(neg_path, image)), (width, height)))) for image in os.listdir(neg_path)]
    # img = cv2.imread('Natolin.jpg')
    # img = cv2.resize(img, (width, height))
    # img = numpy.float32(img)
    # histogram = numpy.asarray(hog(img)).reshape(1, -1)
    classifier = load_classifier(filename)
    human = predict(classifier, numpy.asarray(pos))
    no_human = predict(classifier, numpy.asarray(neg))
    print("Human")
    print(human)
    mean1 = numpy.mean(human)
    print(mean1)
    print("No human")
    print(no_human)
    mean2 = numpy.mean(no_human)
    print(mean2)

if __name__ == "__main__":
    main()
