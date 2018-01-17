import hog
import os
import numpy
import cv2

filename = 'svc.pkl'
pos_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/Test/pos'
neg_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/Test/neg'
width = 96
height = 160

def main():
    classifier = hog.load_classifier(filename)

    pos = [hog.hog(numpy.float32(cv2.resize(cv2.imread(os.path.join(pos_path, image), 0), (width, height)))) for image in os.listdir(pos_path)]
    neg = [hog.hog(numpy.float32(cv2.resize(cv2.imread(os.path.join(neg_path, image), 0), (width, height)))) for image in os.listdir(neg_path)]
    x_test = pos + neg
    y_test = [1] * len(pos) + [0] * len(neg)
    accuracy = classifier.score(x_test, y_test)
    print("Dokładność")
    print(accuracy*100)
    positive = classifier.predict(pos)
    negative = classifier.predict(neg)
    print("Czułość")
    print(numpy.mean(positive)*100)
    print("Swoistość")
    print((1-numpy.mean(negative))*100)

if __name__ == "__main__":
    main()