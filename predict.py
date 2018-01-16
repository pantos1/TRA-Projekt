import hog
import cv2
import numpy

width = 96
height = 160
filename = 'svc.pkl'

def main():
    classifier = hog.load_classifier(filename)
    image_name = input("Podaj nazwę pliku ze zdjęciem:")
    img = cv2.imread(image_name, 0)
    img = cv2.resize(img, (width, height))
    img = numpy.float32(img)
    histogram = numpy.asarray(hog.hog(img)).reshape(1, -1)
    y = hog.predict(classifier, histogram)
    if y == 1:
        print("Na zdjęciu znajduje się człowiek")
    elif y == 0:
        print("Na zdjęciu nie znajduje się człowiek")

if __name__ == "__main__":
    main()