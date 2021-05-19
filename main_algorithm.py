import pickle

import pandas as pd
import numpy as np
import cv2
import math
from sklearn import svm
from sklearn.metrics import accuracy_score

midRanges = [10, 30, 50, 70, 90, 110, 130, 150, 170]
num_of_samples = 500
constant = 144

def normalize(histogram_array):
    if (np.sum(histogram_array) == 0):
        new_histogram = [[[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                         [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                         [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                         [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    else:
        sum_ = np.sum(histogram_array)
        new_histogram = histogram_array / sum_

    return new_histogram


def magnitude(dx, dy):
    mag = []
    for i in range(len(dx)):
        for j in range(len(dx[i])):
            mag.append(math.sqrt((dx[i][j] ** 2) + (dy[i][j] ** 2)))
    mag = np.reshape(mag, (28, 28))
    return mag


def theta_calc(dx, dy):
    theta = []
    for i in range(len(dx)):
        for j in range(len(dx[i])):
            if dx[i][j] == 0:
                theta.append(0)
            else:
                theta.append(math.atan(dy[i][j] / dx[i][j]))
    thetaDeg = np.degrees(theta)
    for i in range(len(thetaDeg)):
        if thetaDeg[i] < 0:
            thetaDeg[i] = thetaDeg[i] + 180
    theta = np.reshape(thetaDeg, (28, 28))
    theta = np.pad(theta, 2, mode="constant")
    return theta


def cell_histogram(mag_cell, theta_cell):
    histogram_magnitudes = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ranges = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]

    cell_size = 8
    for i in range(cell_size):
        for j in range(cell_size):

            if theta_cell[i, j] < midRanges[0]:
                histogram_magnitudes[0] += mag_cell[i, j]
            elif theta_cell[i, j] > midRanges[8]:
                histogram_magnitudes[8] += mag_cell[i, j]
            else:
                for x in range(9):
                    if midRanges[x] <= theta_cell[i, j] < midRanges[x + 1]:
                        ratio1 = (abs(theta_cell[i, j] - midRanges[x])) / ranges[x + 1]
                        ratio2 = (abs(theta_cell[i, j] - midRanges[x + 1])) / ranges[x + 1]
                        if max(ratio1, ratio2) == ratio1:
                            histogram_magnitudes[x + 1] += (ratio1 * mag_cell[i, j])
                        else:
                            histogram_magnitudes[x] += (ratio2 * mag_cell[i, j])

    return histogram_magnitudes


########################################### Training


training_data_file = pd.read_csv('./data/mnist_Training.csv')
image_number = 0
labels = []
feature_vectors = np.array([])
for image_number in range(num_of_samples):
    print('image_number: ', image_number, '\n')

    # read one image from Training data
    image1D = pd.read_csv('./data/mnist_Training.csv', skiprows=image_number, nrows=1).astype(float)

    labels.append(image1D.iloc[0, 0])
    img = image1D.iloc[:, 1:]

    # convert image to 2D
    image = np.array(img).reshape(28, 28)

    # calculate dx and dy
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # dx=1, dy=0, kernel size = 5x5
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # calculate magnitude and theta
    mag = magnitude(dx, dy)
    theta = theta_calc(dx, dy)

    # padding
    image = np.pad(image, 2, mode="constant")
    mag = np.pad(mag, 2, mode="constant")
    theta = np.pad(theta, 2, mode="constant")

    # dimensions
    window_size = 32
    block_size = 16
    block_step = int(block_size / 2)
    cell_size = 8
    j = 0

    block_histogram = []
    feature_vector = []

    # blocks loop
    for i in range(0, window_size - block_size, block_step):
        for j in range(0, window_size - block_size, block_step):
            mag_block = mag[i:i + block_size, j:j + block_size]
            theta_block = theta[i:i + block_size, j:j + block_size]

            # cells loop
            counter = 0
            for x in range(0, block_size, cell_size):
                for y in range(0, block_size, cell_size):
                    mag_cell = mag_block[x:x + cell_size, y:y + cell_size]
                    theta_cell = theta_block[x:x + cell_size, y:y + cell_size]

                    cell_hist = cell_histogram(mag_cell, theta_cell)
                    block_histogram = np.concatenate((block_histogram,cell_hist))

                    #block_histogram.append(cell_hist)

            feature_vector = np.concatenate((feature_vector, block_histogram))
            block_histogram = []

    # now the feature vector has all image data
    # normalize
    feature_vector = normalize(feature_vector)
    feature_vectors = np.concatenate((feature_vectors,feature_vector))

feature_vectors = np.array(feature_vectors).reshape(num_of_samples, int((len(feature_vectors)/num_of_samples)))


# classification
filename = 'HoG_Model.sav'
classifier = svm.SVC()
classifier.fit(feature_vectors, labels)
predicted = classifier.predict(feature_vectors)

#save model
pickle.dump(classifier, open(filename, 'wb'))

print('\n--Training Accuracy:',accuracy_score(labels, predicted)*100,'%')


########################################### Testing


testing_data_file = pd.read_csv('./data/mnist_Testing.csv')
image_number = 0
labels_test = []
feature_vectors_test = []

for image_number in range(num_of_samples):
    print('image_number: ', image_number, '\n')

    # read one image from Training data
    image1D = pd.read_csv('./data/mnist_Testing.csv', skiprows=image_number, nrows=1).astype(float)

    labels_test.append(image1D.iloc[0, 0])
    img = image1D.iloc[:, 1:]

    # convert image to 2D
    image = np.array(img).reshape(28, 28)

    # calculate dx and dy
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # dx=1, dy=0, kernel size = 5x5
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # calculate magnitude and theta
    mag = magnitude(dx, dy)
    theta = theta_calc(dx, dy)

    # padding
    image = np.pad(image, 2, mode="constant")
    mag = np.pad(mag, 2, mode="constant")
    theta = np.pad(theta, 2, mode="constant")

    # dimensions
    window_size = 32
    block_size = 16
    block_step = int(block_size / 2)
    cell_size = 8
    j = 0

    block_histogram = []
    feature_vector = []

    # blocks loop
    for i in range(0, window_size - block_size, block_step):
        for j in range(0, window_size - block_size, block_step):
            mag_block = mag[i:i + block_size, j:j + block_size]
            theta_block = theta[i:i + block_size, j:j + block_size]

            # cells loop
            counter = 0
            for x in range(0, block_size, cell_size):
                for y in range(0, block_size, cell_size):
                    mag_cell = mag_block[x:x + cell_size, y:y + cell_size]
                    theta_cell = theta_block[x:x + cell_size, y:y + cell_size]

                    cell_hist = cell_histogram(mag_cell, theta_cell)
                    block_histogram = np.concatenate((block_histogram,cell_hist))


            feature_vector = np.concatenate((feature_vector, block_histogram))
            block_histogram = []

    # now the feature vector has all image data
    # normalize
    feature_vector = normalize(feature_vector)
    feature_vectors_test.append(feature_vector)


#load model
dim1 = int(math.sqrt(len(labels_test)))
dim2 = int(math.sqrt(len(feature_vectors_test)))
loaded_model = pickle.load(open(filename,'rb'))
feature_vectors_test= np.array(feature_vectors_test).reshape(num_of_samples,constant)


result = loaded_model.score(feature_vectors_test, labels_test)

print('\n--Testing Accuracy:',result*100,'%')


