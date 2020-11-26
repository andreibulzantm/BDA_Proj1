import math
import time

import numpy as np
from matplotlib import pyplot as plt


def read_data(file_path: str):
    features = []
    plant_types = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            line_data = line.strip('\n ').split(',')
            features.append([float(feature) for feature in line_data[:-1]])
            plant_types.append(line_data[-1])
            line = f.readline()

    return np.asarray(features), np.asarray(plant_types)


def split_in_partitions(data: np.array, num_partitions: int) -> np.array:
    total_data_num = data.shape[0]
    data_per_partition = math.floor(total_data_num / num_partitions)
    extra_last_partition = total_data_num % num_partitions

    partitions = []
    data_idx = 0
    for current_partition_idx in range(num_partitions):
        if current_partition_idx == num_partitions - 1:
            data_per_partition += extra_last_partition

        partitions.append(
            data[data_idx:data_idx+data_per_partition]
        )

    return partitions


def plot_centroids(data: np.array, centroids: np.array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, cmap=plt.hot())
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='r', s=250)
    plt.show()

    plt.scatter(data[:, 0], data[:, 1], s=7)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='g', s=150)
    plt.show()

    plt.scatter(data[:, 2], data[:, 3], s=7)
    plt.scatter(centroids[:, 2], centroids[:, 3], marker='*', c='g', s=150)
    plt.show()


def plot_time_results(alg_references: list, reference_label: str, times: list):
    plt.plot(alg_references, times, 'ro')
    plt.xlabel(reference_label)
    plt.ylabel('Seconds')
    plt.axis([0, alg_references[-1]*1.2, 0, times[-1]*1.2])
    plt.show()


def timeit(func, *args):
    """
        Returns the runtime of the `func` in seconds
    """
    start = time.time()
    func(*args)
    end = time.time()

    return end - start
