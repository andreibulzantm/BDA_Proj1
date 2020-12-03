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


def calculate_membership_mat(np_data, initial_centroids, clusters=3, alg=0, fuzzification=1.1,
                             epsilon=0.00001):  # Steps UF3 & UF4
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

    else:
        mtrx1 = np_data - initial_centroids[0]
        mtrx1 = np.power(mtrx1, 2)
        mtrx1_dists = np.sum(mtrx1, axis=1)
        mtrx1_dists = np.sqrt(mtrx1_dists)
        mtrx1_dists = np.where(mtrx1_dists == 0, epsilon, mtrx1_dists)
        mtrx1_inversed = np.divide(1, mtrx1_dists)
        mtrx1_inversed = np.power(mtrx1_inversed, (2 / (fuzzification - 1)))

        mtrx2 = np_data - initial_centroids[1]
        mtrx2 = np.power(mtrx2, 2)
        mtrx2_dists = np.sum(mtrx2, axis=1)
        mtrx2_dists = np.sqrt(mtrx2_dists)
        mtrx2_dists = np.where(mtrx2_dists == 0, epsilon, mtrx2_dists)
        mtrx2_inversed = np.divide(1, mtrx2_dists)
        mtrx2_inversed = np.power(mtrx2_inversed, (2 / (fuzzification - 1)))

        mtrx3 = np_data - initial_centroids[2]
        mtrx3 = np.power(mtrx3, 2)
        mtrx3_dists = np.sum(mtrx3, axis=1)
        mtrx3_dists = np.sqrt(mtrx3_dists)
        mtrx3_dists = np.where(mtrx3_dists == 0, epsilon, mtrx3_dists)
        mtrx3_inversed = np.divide(1, mtrx3_dists)
        mtrx3_inversed = np.power(mtrx3_inversed, (2 / fuzzification - 1))

        sum_mtrx_inversed = mtrx1_inversed + mtrx2_inversed + mtrx3_inversed

        left = np.stack((mtrx1_inversed, mtrx2_inversed, mtrx3_inversed), axis=1)
        right = sum_mtrx_inversed
        final = np.divide(left, right[:, None])
        return final

    return init_matrix


def compute_new_centroids(np_data, membership_matrix, clusters=3, fuzzification=1.1):  # Step UF5
    new_centroids = np.zeros((clusters, 4))
    for i in range(0, clusters):
        lsum = 0
        rsum = 0
        for j in range(len(np_data)):
            lsum += np.multiply(math.pow(membership_matrix[j][i], fuzzification), np_data[j])
            rsum += math.pow(membership_matrix[j][i], fuzzification)
        new_centroids[i] = np.divide(lsum, rsum)
    return new_centroids


def step(data: np.array, centroids: np.array, k, alg, fuzzification=1.1):
    membership_mat = calculate_membership_mat(
        data, centroids, clusters=k, alg=alg,
        fuzzification=fuzzification
    )

    centroids = compute_new_centroids(data, membership_mat, k, fuzzification=fuzzification)

    return centroids


def fit(k, data: np.array, alg, max_iter, fuzzification=1.1, epsilon_stop=0.0001):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        prev_centroids = centroids
        centroids = step(data, centroids, k, alg, fuzzification)
        if np.sum(np.abs(centroids - prev_centroids)) / (centroids.shape[0] * prev_centroids.shape[1]) < epsilon_stop:
            break




    return centroids
    # centroids o să aibă valoarea finală după for, practic e step UF6 (doar că ar mai trebui pus și al doilea stopping criterion cu un epsilon de diferență centroizi la iterații diferite)


def compute_new_centroids_multi_process(np_data, membership_matrix, clusters=3, fuzzification=1.1):  # Step UF5
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


def step_multi_process(data: np.array, centroids: np.array, k, alg, fuzzification=1.1):
    membership_mat = calculate_membership_mat(
        data, centroids, clusters=k, alg=alg,
        fuzzification=fuzzification
    )

    centroids, mem_degree = compute_new_centroids_multi_process(data, membership_mat, k, fuzzification=fuzzification)
    return centroids, mem_degree


def fit_multiprocess(k, partitions, data: np.array, alg, max_iter, fuzzification=1.1, epsilon_stop=0.0001):
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
            prev_centroids = centroids
            centroids = compute_global_centroids(k, partitions_result)
            if np.sum(np.abs(centroids - prev_centroids)) / (
                    centroids.shape[0] * prev_centroids.shape[1]) < epsilon_stop:
                break
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
