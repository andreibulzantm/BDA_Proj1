import math
import random
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from utils import split_in_partitions


def initialize_centroids(np_data, clusters=3):  # Step UF2
    list_data = list(np_data)
    return np.asarray(random.choices(list_data, k=clusters))


def euclid_dist(point, cluster):
    dist = 0
    for i in range(0, 4):
        dist += math.pow(float(point[i]) - float(cluster[i]), 2)
    return math.sqrt(dist)


def calculate_membership_mat(np_data, initial_centroids, clusters=3, alg=0, fuzzification=1,
                             epsilon=0.00001):  # Steps UF3 & UF4
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


def step(data: np.array, centroids: np.array, k, alg, fuzzification):
    membership_mat = calculate_membership_mat(
        data, centroids, clusters=k, alg=alg,
        fuzzification=fuzzification
    )

    centroids = compute_new_centroids(data, membership_mat, k, fuzzification=fuzzification)

    return centroids


def fit(k, data: np.array, alg, max_iter, fuzzification=1):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        centroids = step(data, centroids, k, alg, fuzzification)

    return centroids
    # centroids o să aibă valoarea finală după for, practic e step UF6 (doar că ar mai trebui pus și al doilea stopping criterion cu un epsilon de diferență centroizi la iterații diferite)


def compute_new_centroids_multi_process(np_data, membership_matrix, clusters=3, fuzzification=1):  # Step UF5
    new_centroids = np.zeros((clusters, 4))
    new_mem_degree = np.zeros(clusters)

    for i in range(0, clusters):
        left_sum = np.zeros(4)
        right_sum = 0
        for j in range(len(np_data)):
            left_sum += np.multiply(math.pow(membership_matrix[j][i], fuzzification), np_data[j])
            right_sum += math.pow(membership_matrix[j][i], fuzzification)
        new_centroids[i] = left_sum
        new_mem_degree[i] = right_sum
    return new_centroids, new_mem_degree


def step_multi_process(data: np.array, centroids: np.array, k, alg, fuzzification):
    membership_mat = calculate_membership_mat(
        data, centroids, clusters=k, alg=alg,
        fuzzification=fuzzification
    )

    centroids, mem_degree = compute_new_centroids_multi_process(data, membership_mat, k, fuzzification=fuzzification)
    return centroids, mem_degree


def fit_multiprocess(k, partitions, data: np.array, alg, max_iter, fuzzification=1):
    data_partitions = split_in_partitions(data, partitions)
    centroids = initialize_centroids(data, k)

    for i in tqdm(range(max_iter), desc="Iterations"):
        process_arguments = list(zip(
            data_partitions,
            [centroids] * partitions,
            [k] * partitions,
            [alg] * partitions,
            [fuzzification] * partitions
        ))

        with Pool(partitions) as pool:
            partitions_result = pool.map(
                _step_wrapper,
                process_arguments
            )
            centroids = compute_global_centroids(k, partitions_result)

    return centroids


def compute_global_centroids(clusters, partitions_result):
    new_centroids = np.zeros((clusters, 4))
    for i in range(0, clusters):
        centroid = np.zeros(4)
        mem_degree = 0
        for alfa in range(0, len(partitions_result)):
            centroid += partitions_result[alfa][0][i]
            mem_degree += partitions_result[alfa][1][i]

        if mem_degree > 0.0:
            new_centroids[i] = np.divide(centroid, mem_degree)
        else:
            new_centroids[i] = centroid
    return new_centroids


def _step_wrapper(params):
    return step_multi_process(
        data=params[0],
        centroids=params[1],
        k=params[2],
        alg=params[3],
        fuzzification=params[4]
    )
