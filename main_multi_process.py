import argparse
import time

from unified import fit_multiprocess
from utils import read_data, plot_centroids, split_in_partitions

parser = argparse.ArgumentParser()
parser.add_argument('--num_centroids', type=int, default=3)
parser.add_argument('--alg', type=str, choices=('KM', 'FCM'), default='KM')
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--input_path', type=str, default='./input/Iris-150.txt')
parser.add_argument('--num_partitions', type=int, default=7)

if __name__ == '__main__':
    args = parser.parse_args()

    alg = {
        'KM': 0,
        'FCM': 1
    }[args.alg]

    start = time.time()

    data, plant_types = read_data(args.input_path)
    plant_partitions = split_in_partitions(plant_types, args.num_partitions)

    centroids = fit_multiprocess(args.num_centroids, args.num_partitions, data, alg, args.max_iter)

    end = time.time()
    print(f'Algorithm took: {end - start} seconds')

    plot_centroids(data, centroids)
