import numpy
import cv2
import os
from sklearn import svm
from sklearn.externals import joblib
from hog import hog

pos_path = 'Training_set/train_64x128_H96/pos/'
neg_path = 'Training_set/Train/neg/'
filename = 'svc.pkl'

height = 160
width = 96

def train_classifier(X, y):
    classifier = svm.LinearSVC()
    classifier.fit(X, y)
    return classifier

def save_classifier(classifier, filename):
    _ = joblib.dump(classifier, filename, compress=9)

def get_hog(dir_path):
    histograms = [hog(numpy.float32(cv2.resize(cv2.imread(os.path.join(dir_path, image)),(width, height)))) for image in os.listdir(dir_path)]
    return histograms

def main():
    pos = [hog(numpy.float32(cv2.imread(os.path.join(pos_path, image), 0))) for image in os.listdir(pos_path)]
    neg = [hog(numpy.float32(cv2.resize(cv2.imread(os.path.join(neg_path, image), 0), (width, height)))) for image in os.listdir(neg_path)]

    x = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    svc = train_classifier(x, y)
    save_classifier(svc, filename)
    print('a')

if __name__ == '__main__':
    main()