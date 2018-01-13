import numpy
import cv2
import os
from sklearn import svm
from sklearn.externals import joblib
from hog import hog

pos_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/train_64x128_H96/pos/'
neg_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/Train/neg/'
filename = 'svc.pkl'

def train_classifier(X, y):
    classifier = svm.LinearSVC()
    classifier.fit(X, y)
    return classifier

def save_classifier(classifier, filename):
    _ = joblib.dump(classifier, filename, compress=9)

def main():
    # pos = []
    pos = [hog(numpy.float32(cv2.imread(os.path.join(pos_path, image)))) for image in os.listdir(pos_path)]
    neg = [hog(numpy.float32(cv2.imread(os.path.join(neg_path, image)))) for image in os.listdir(neg_path)]
    x = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    svc = train_classifier(x, y)
    # for image in os.listdir(pos_path):
    #     img = cv2.imread(os.path.join(pos_path, image))
    #     img = numpy.float32(img)
    #     histogram = hog(img)
    #     pos.append(histogram, 1)
    print('a')

if __name__ == '__main__':
    main()