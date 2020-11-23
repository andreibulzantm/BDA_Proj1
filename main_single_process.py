import argparse
import time

from unified import fit
from utils import read_data, plot_data

parser = argparse.ArgumentParser()
parser.add_argument('--num_centroids', type=int, default=3)
parser.add_argument('--alg', type=str, choices=('KM', 'FCM'), default='KM')
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--input_path', type=str, default='./input/Iris-150.txt')


if __name__ == '__main__':
    args = parser.parse_args()

    alg = {
        'KM': 0,
        'FCM': 1
    }[args.alg]

    start = time.time()

    data, plant_types = read_data(args.input_path)
    centroids = fit(
        args.num_centroids,
        data,
        alg,
        args.max_iter
    )  # max_iter = <some_val> și k (clusters) = some_num sunt step UF1

    end = time.time()
    print(f'Algorithm took: {end - start} seconds')

    plot_data(data, centroids)


