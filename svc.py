import numpy
import cv2
import os
import threading
from sklearn import svm
from sklearn.externals import joblib
from hog import hog

pos_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/train_64x128_H96/pos/'
neg_path = 'C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt/Training_set/Train/neg/'
filename = 'svc.pkl'

height = 160
width = 96

class hog_thread(threading.Thread):
    def __init__(self, threadID, name, dir_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.dir_path = dir_path
        self.histogram = []
    def run(self):
        self.histogram = get_hog(self.dir_path)
    def join(self, timeout=None):
        threading.Thread.join(self)
        return self.histogram

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
    # # Multithreading
    # pos_thread = hog_thread(1, 'pos', pos_path)
    # neg_thread = hog_thread(2, 'neg', neg_path)
    #
    # pos_thread.start()
    # neg_thread.start()
    #
    # pos = pos_thread.join()
    # neg = neg_thread.join()

    pos = [hog(numpy.float32(cv2.imread(os.path.join(pos_path, image)))) for image in os.listdir(pos_path)]
    neg = [hog(numpy.float32(cv2.resize(cv2.imread(os.path.join(neg_path, image)), (width, height)))) for image in os.listdir(neg_path)]

    x = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    svc = train_classifier(x, y)
    save_classifier(svc, filename)
    # for image in os.listdir(pos_path):
    #     img = cv2.imread(os.path.join(pos_path, image))
    #     img = numpy.float32(img)
    #     histogram = hog(img)
    #     pos.append(histogram, 1)
    print('a')

if __name__ == '__main__':
    main()