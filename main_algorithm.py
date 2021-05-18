import pandas as pd
import numpy as np
import matplotlib as plt
import cv2
from PIL import Image
import math

midRanges = [10, 30, 50, 70, 90, 110, 130, 150, 170]

def normalize(histogram_array):
    if(np.sum(histogram_array) == 0):
        new_histogram = [[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]]
    else:
        sum_ = np.sum(histogram_array)
        new_histogram = histogram_array/sum_

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

            if theta_cell[i,j] < midRanges[0]:
                histogram_magnitudes[0] += mag_cell[i,j]
            elif theta_cell[i,j] > midRanges[8]:
                histogram_magnitudes[8] += mag_cell[i,j]
            else:
                for x in range(9):
                    if midRanges[x] <= theta_cell[i,j] < midRanges[x + 1]:
                        ratio1 = (abs(theta_cell[i,j] - midRanges[x])) / ranges[x + 1]
                        ratio2 = (abs(theta_cell[i,j] - midRanges[x + 1])) / ranges[x + 1]
                        if max(ratio1, ratio2) == ratio1:
                            histogram_magnitudes[x + 1] += (ratio1 * mag_cell[i,j])
                        else:
                            histogram_magnitudes[x] += (ratio2 * mag_cell[i,j])

    return histogram_magnitudes

########################################### Training


training_data_file = pd.read_csv('./data/mnist_Training.csv')
image_number = 0

for image_number in range(len(training_data_file)):
    print('image_number: ',image_number,'\n')

    #read one image from Training data
    image1D = pd.read_csv('./data/mnist_Training.csv', skiprows=image_number, nrows=1).astype(float)

    #convert image to 2D and pad to be of size 32x32
    image1D = image1D.iloc[:,:-1]   #skip the last element
    image = np.array(image1D).reshape(28, 28)

    #calculate dx and dy
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # dx=1, dy=0, kernel size = 5x5
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    #calculate magnitude and theta
    mag = magnitude(dx,dy)
    theta = theta_calc(dx,dy)

    #padding
    image = np.pad(image, 2, mode="constant")
    mag = np.pad(mag, 2, mode="constant")
    theta = np.pad(theta, 2, mode="constant")

    #dimensions
    window_size = 32
    block_size = 16
    block_step = int(block_size/2)
    cell_size = 8
    j=0

    block_histogram = []
    feature_vector = []

    #blocks loop
    for i in range(0, window_size-block_size, block_step):
        for j in range(0,window_size-block_size,block_step):
            mag_block = mag[i:i+block_size, j:j+block_size]
            theta_block = theta[i:i+block_size, j:j+block_size]

            #cells loop
            counter=0
            for x in range(0, block_size,cell_size):
                for y in range(0,block_size,cell_size):

                    mag_cell = mag_block[x:x+cell_size, y:y+cell_size]
                    theta_cell= theta_block[x:x+cell_size, y:y+cell_size]

                    cell_hist = cell_histogram(mag_cell,theta_cell)
                    block_histogram.append(cell_hist)

            feature_vector.append(block_histogram)
            block_histogram = []

    #now the feature vector has all image data
    #normalize
    feature_vector = normalize(feature_vector)
    print('\nfeature vector:\n',feature_vector)




########################################### Training


testing_data_file = pd.read_csv('./data/mnist_Testing.csv')
image_number = 0

for image_number_ in range(len(testing_data_file)):
    print('image_number: ',image_number_,'\n')

    #read one image from Training data
    image1D = pd.read_csv('./data/mnist_Testing.csv', skiprows=image_number_, nrows=1).astype(float)

    #convert image to 2D and pad to be of size 32x32
    image1D = image1D.iloc[:,:-1]   #skip the last element
    image = np.array(image1D).reshape(28, 28)

    #calculate dx and dy
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # dx=1, dy=0, kernel size = 5x5
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    cv2.imshow('dx',dx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #calculate magnitude and theta
    mag = magnitude(dx,dy)
    theta = theta_calc(dx,dy)

    #padding
    image = np.pad(image, 2, mode="constant")
    mag = np.pad(mag, 2, mode="constant")
    theta = np.pad(theta, 2, mode="constant")

    #dimensions
    window_size = 32
    block_size = 16
    block_step = int(block_size/2)
    cell_size = 8
    j=0

    block_histogram = []
    feature_vector = []

    #blocks loop
    for i in range(0, window_size-block_size, block_step):
        for j in range(0,window_size-block_size,block_step):
            mag_block = mag[i:i+block_size, j:j+block_size]
            theta_block = theta[i:i+block_size, j:j+block_size]

            #cells loop
            counter=0
            for x in range(0, block_size,cell_size):
                for y in range(0,block_size,cell_size):

                    mag_cell = mag_block[x:x+cell_size, y:y+cell_size]
                    theta_cell= theta_block[x:x+cell_size, y:y+cell_size]

                    cell_hist = cell_histogram(mag_cell,theta_cell)
                    block_histogram.append(cell_hist)

            feature_vector.append(block_histogram)
            block_histogram = []

    #now the feature vector has all image data
    #normalize
    feature_vector = normalize(feature_vector)
    print('\nfeature vector:\n',feature_vector)










