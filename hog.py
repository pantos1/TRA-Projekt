import numpy
import cv2
import os
import matplotlib.pyplot as plt
from skimage import draw, exposure
from sklearn.externals import joblib

pos_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/Test/pos'
neg_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/Test/neg'
width = 96
height = 160

filename = 'svc.pkl'

def hog(img, bin_number = 9, cell = (8,8), norm_cell=(16,16), visualise=False):
    """
    :param img: Obraz z którego ma być policzony histogram
    :param bin_number: Liczba orientacji(kątów) w histogramie
    :param cell: Wielkość komórki (w pikselach)
    :param norm_cell: Wielkość komórki normalizacyjnej (w pikselach)
    :param visualise: Flaga stwierdzająca, czy ma być zwrócony również obraz do wizualizacji histogramu
    :return: histogram - wektor zawierający obliczony histogram, hog_image - Histogram do wizualizacji
    """

    img = img / 255.0
    histogram =[]
    # Obliczanie gradientu
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Przeliczanie gradientu na biegunowy układ współrzędnych
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # magnitude = magnitude[...,0]
    # angle = angle[...,0]
    angle = abs(angle - 180)

    #Tablica zawierająca piksele podzielone na komórki
    angle_cells = []
    magnitude_cells = []
    #Wielkość obrazu w pikselach
    y_size = img.shape[0]
    x_size = img.shape[1]
    n_cell_x, n_cell_y = int(x_size/cell[0]), int(y_size/cell[1])

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

        weight = ((ang / 20) - numpy.floor(ang / 20)).astype(numpy.float32)
        h1, b1 = numpy.histogram(20*numpy.floor(ang / 20).astype(numpy.int32), weights=mag * (1.0 - weight), bins=bin_number)
        h2, b2 = numpy.histogram(20*numpy.ceil(ang / 20).astype(numpy.int32), weights=mag * weight, bins=bin_number)
        hist = h1 + h2
        histogram.append(hist)
    del magnitude_cells, angle_cells

    #Zamiana na listę o wymiarach równych liczba komórek do histogramu razy liczba komórek do histogramu razy liczba binów
    histogram = numpy.array(histogram, dtype=float).reshape(n_cell_y, n_cell_x, bin_number)

    hog_image = None
    if visualise:
        cx, cy = cell
        sx, sy = x_size, y_size
        radius = min(cx, cy) // 2 - 1
        orientations_arr = numpy.arange(bin_number)
        dx_arr = radius * numpy.cos(orientations_arr / bin_number * 180)
        dy_arr = radius * numpy.sin(orientations_arr / bin_number * 180)
        hog_image = numpy.zeros((sy, sx), dtype=float)
        for x in range(n_cell_x):
            for y in range(n_cell_y):
                for o, dx, dy in zip(orientations_arr, dx_arr, dy_arr):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    rr, cc = draw.line(int(centre[0] - dx),
                                       int(centre[1] + dy),
                                       int(centre[0] + dx),
                                       int(centre[1] - dy))
                    hog_image[rr, cc] += histogram[y, x, o]

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

    #Spłaszczenie listy  w jeden wektor
    histogram = numpy.concatenate(normalised_cells).ravel().tolist()

    if visualise:
        return histogram, hog_image
    else:
        return histogram

def predict(classifier, X):
    return classifier.predict(X)

def load_classifier(filename):
    return joblib.load(filename)

def main():
    image_name = input("Podaj nazwę pliku ze zdjęciem:")
    img = cv2.imread(image_name, 0)
    # img = cv2.resize(img, (width, height))
    img = numpy.float32(img)
    histogram, image = numpy.asarray(hog(img, visualise = True))
    hog_image = exposure.rescale_intensity(image, in_range=(0, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.Greys_r)
    ax1.set_title('Obraz')
    ax1.set_adjustable('box-forced')
    ax2.axis('off')
    ax2.imshow(hog_image, cmap=plt.cm.Greys_r)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()

if __name__ == "__main__":
    main()
