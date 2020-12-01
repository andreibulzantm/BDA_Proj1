import random
import time
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

epsilon = 0.00001
stop_eps = 0.00001


def read_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    for idx in range(len(data)):
        data[idx] = data[idx].split(',')
        data[idx][-1] = data[idx][-1].replace('\n', '')

    for i in range(len(data)):
        for j in range(len(data[i]) -1):
            data[i][j] = float(data[i][j])
        #print(data[i])

    return data


def initialize_centroids(np_data, clusters=3): #Step UF2
    list_data = list(np_data)
    return np.asarray(random.choices(list_data, k=clusters))


def euclid_dist(point, cluster):
    dist = 0
    for i in range(0, 4):
        dist += math.pow(float(point[i]) - float(cluster[i]), 2)
    return math.sqrt(dist)


def calculate_membership_mat(np_data, initial_centroids, clusters=3, alg=0, fuzzification=1): #Steps UF3 & UF4
    #broadcasting instead of fors

    init_matrix = np.zeros((len(np_data), clusters))

    if alg == 0:
        mtrx1 = np_data - initial_centroids[0]
        mtrx1 = np.power(mtrx1, 2)
        mtrx1_dists = np.sum(mtrx1, axis=1)
        mtrx1_dists = np.sqrt(mtrx1_dists)

        mtrx2 = np_data - initial_centroids[1]
        mtrx2 = np.power(mtrx2, 2)
        mtrx2_dists = np.sum(mtrx2, axis=1)
        mtrx2_dists = np.sqrt(mtrx2_dists)

        mtrx3 = np_data - initial_centroids[2]
        mtrx3 = np.power(mtrx3, 2)
        mtrx3_dists = np.sum(mtrx3, axis=1)
        mtrx3_dists = np.sqrt(mtrx3_dists)

        final_mtrx_dists = np.stack((mtrx1_dists, mtrx2_dists, mtrx3_dists), axis=1)
        b = np.zeros_like(final_mtrx_dists)
        b[np.arange(len(final_mtrx_dists)), final_mtrx_dists.argmin(1)] = 1
        #print(final_mtrx_dists)
        return b

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
            if i == len(np_data)-1:
                print(min, min_idx)
                print(init_matrix)
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


def compute_new_centroids(np_data, membership_matrix, clusters=3, fuzzification=1): #Step UF5
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
    start = time.time()
    init_centroids = initialize_centroids(np_data, k)
    for iter in range(max_iter):
        membership_mat = calculate_membership_mat(np_data, init_centroids, clusters=k, alg=alg,
                                                  fuzzification=fuzzification)

        new_centroids = compute_new_centroids(np_data, membership_mat, k, fuzzification=fuzzification)
        print(np.sum(np.abs(new_centroids - init_centroids)) / (new_centroids.shape[0] * new_centroids.shape[1]))
        if np.sum(np.abs(new_centroids - init_centroids)) / (new_centroids.shape[0] * new_centroids.shape[1]) < stop_eps:
            break
        init_centroids = new_centroids


    if k == 3:
        membership_mat_max = membership_mat.argmax(1)
        print(membership_mat_max)
        good_0_0 = 0
        good_0_1 = 0
        good_0_2 = 0
        good_1_0 = 0
        good_1_1 = 0
        good_1_2 = 0
        good_2_0 = 0
        good_2_1 = 0
        good_2_2 = 0
        cnt0, cnt1, cnt2 = 0, 0, 0
        for i in range(len(all_labels)):
            if all_labels[i] == '0':
                cnt0+=1
                if membership_mat_max[i] == 0:
                    good_0_0 += 1
                elif membership_mat_max[i] == 1:
                    good_0_1 += 1
                else:
                    good_0_2 += 1
            elif all_labels[i] == '1':
                if membership_mat_max[i] == 0:
                    good_1_0 += 1
                elif membership_mat_max[i] == 1:
                    good_1_1 += 1
                else:
                    good_1_2 += 1
            elif all_labels[i] == '2':
                if membership_mat_max[i] == 0:
                    good_2_0 += 1
                elif membership_mat_max[i] == 1:
                    good_2_1 += 1
                else:
                    good_2_2 += 1
        print("cnt0", cnt0)
        print(good_0_0, good_0_1, good_0_2)
        good_0 = max(good_0_0, good_0_1, good_0_2)
        good_1 = max(good_1_0, good_1_1, good_1_2)
        good_2 = max(good_2_0, good_2_1, good_2_2)
        print("identified 0, 1, 2", good_0, good_1, good_2)
        acc0 = good_0 / (good_0_0 + good_0_1 + good_0_2)
        acc1 = good_1 / (good_1_0 + good_1_1 + good_1_2)
        acc2 = good_2 / (good_2_0 + good_2_1 + good_2_2)
        print("acc0, acc1, acc2", acc0, acc1, acc2)
        total_acc = (good_0 + good_1 + good_2) / np_data.shape[0]

        print("total_acc:", total_acc)
        #acc = good / (good_0 + good_1 + good_2)
        #print("acc", acc)
        #print(good)
        #print(good_0, good_1, good_2, "good0, 1, 2")
    end = time.time()
    print(end - start)
    #init_centroids o să aibă valoarea finală după for, practic e step UF6 (doar că ar mai trebui pus și al doilea stopping criterion cu un epsilon de diferență centroizi la iterații diferite)
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



data = read_data('D:\BDA\KaggleBDA\input\Iris-150.txt')
all_labels = np.asarray(data)[:, -1]
all_labels = np.where(all_labels == 'Iris-setosa', int(0), all_labels)
all_labels = np.where(all_labels == 'Iris-versicolor', 1, all_labels)
all_labels = np.where(all_labels == 'Iris-virginica', 2, all_labels)

print(all_labels)
plant_types = []
for elem in data:
    #plant_types.append(elem[-1])
    del elem[-1]


np_data = np.asarray(data)
print(np_data.shape)
print(all_labels.shape)
unified(3, np_data, 0, 100) #max_iter = <some_val> și k (clusters) = some_num sunt step UF1

#print(init_centroids)
