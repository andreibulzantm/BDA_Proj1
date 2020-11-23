import random
import time
import numpy as np
import math
from matplotlib import pyplot as plt

epsilon = 0.00001


def read_data(file_path):
    features = []
    plant_types = []

    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            line_data = line.strip('\n ').split(',')
            features.append([float(feature) for feature in line_data[:-1]])
            plant_types.append(line_data[-1])
            line = f.readline()

    return np.asarray(features), plant_types


def initialize_centroids(np_data, clusters=3):  # Step UF2
    list_data = list(np_data)
    return np.asarray(random.choices(list_data, k=clusters))


def euclid_dist(point, cluster):
    dist = 0
    for i in range(0, 4):
        dist += math.pow(float(point[i]) - float(cluster[i]), 2)
    return math.sqrt(dist)


def calculate_membership_mat(np_data, initial_centroids, clusters=3, alg=0, fuzzification=1):  # Steps UF3 & UF4
    init_matrix = np.zeros((len(np_data), clusters))

    if alg == 0:
        for i in range(0, len(np_data)):
            min_idx = 0
            min = 999
            for j in range(0, clusters):
                for k in range(0, clusters):
                    if j != k:
                        d1 = euclid_dist(np_data[i], initial_centroids[j])
                        d2 = euclid_dist(np_data[i], initial_centroids[k])
                        if d1 <= d2 and d1 < min:
                            min = d1
                            min_idx = j
                        else:
                            if d2 < min:
                                min = d2
                                min_idx = k

            init_matrix[i][min_idx] = 1

    else:
        for i in range(0, len(np_data)):
            for j in range(0, clusters):
                leftSum = euclid_dist(np_data[i], initial_centroids[j])
                if leftSum == 0:
                    leftSum = epsilon
                leftSum = 1 / leftSum
                power = 2 / (fuzzification - 1)
                leftSum = math.pow(leftSum, power)
                rightSum = 0
                for k in range(0, clusters):
                    value = euclid_dist(np_data[i], initial_centroids[k])
                    if value == 0:
                        value = epsilon
                    value = 1 / value
                    value = math.pow(value, power)
                    rightSum += value
                init_matrix[i][j] = leftSum / rightSum

    return init_matrix


def compute_new_centroids(np_data, membership_matrix, clusters=3, fuzzification=1):  # Step UF5
    new_centroids = np.zeros((clusters, 4))
    for i in range(0, clusters):
        lsum = 0
        rsum = 0
        for j in range(len(np_data)):
            lsum += np.multiply(math.pow(membership_matrix[j][i], fuzzification), np_data[j])
            rsum += math.pow(membership_matrix[j][i], fuzzification)
        new_centroids[i] = np.divide(lsum, rsum)
    return new_centroids


def unified(k, np_data, alg, max_iter, fuzzification=1):
    init_centroids = initialize_centroids(np_data, k)
    for iter in range(max_iter):
        membership_mat = calculate_membership_mat(np_data, init_centroids, clusters=k, alg=alg,
                                                  fuzzification=fuzzification)

        new_centroids = compute_new_centroids(np_data, membership_mat, k, fuzzification=fuzzification)
        init_centroids = new_centroids
    # init_centroids o să aibă valoarea finală după for, practic e step UF6 (doar că ar mai trebui pus și al doilea stopping criterion cu un epsilon de diferență centroizi la iterații diferite)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(np_data[:, 0], np_data[:, 1], np_data[:, 2], s=5, cmap=plt.hot())
    img = ax.scatter(init_centroids[:, 0], init_centroids[:, 1], init_centroids[:, 2], marker='*', c='r', s=250)
    plt.show()

    plt.scatter(np_data[:, 0], np_data[:, 1], s=7)
    plt.scatter(init_centroids[:, 0], init_centroids[:, 1], marker='*', c='g', s=150)
    plt.show()

    plt.scatter(np_data[:, 2], np_data[:, 3], s=7)
    plt.scatter(init_centroids[:, 2], init_centroids[:, 3], marker='*', c='g', s=150)
    plt.show()


start = time.time()
data, plant_types = read_data('./input/Iris-150.txt')

unified(3, data, 0, 100)  # max_iter = <some_val> și k (clusters) = some_num sunt step UF1

# print(init_centroids)
end = time.time()
# print(end-start)
