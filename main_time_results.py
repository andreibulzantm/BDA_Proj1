import argparse
import os

import tqdm

import utils
from unified import fit, fit_multiprocess
from utils import timeit

parser = argparse.ArgumentParser()
parser.add_argument('--num_centroids', type=int, default=3)
parser.add_argument('--alg', type=str, choices=('KM', 'FCM'), default='FCM')
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--data_dir_path', type=str, default='./input/')
parser.add_argument('--workers', type=int, default=2,)
parser.add_argument('--fuzzification', type=int, default=1.1)


FILES = ['Iris-150.txt', 'Iris-1500.txt', 'Iris-15000.txt', 'Iris-150000.txt', 'Iris-1500000.txt', 'Iris-15000000.txt']


if __name__ == '__main__':
    args = parser.parse_args()

    alg = {
        'KM': 0,
        'FCM': 1
    }[args.alg]

    data_lengths = []
    alg_times = []
    for file in tqdm.tqdm(FILES[:-1]):
        data, _ = utils.read_data(os.path.join(args.data_dir_path, file))
        data_lengths.append(data.shape[0])

        if args.workers == 1:
            alg_args = (
                args.num_centroids,
                data,
                alg,
                args.max_iter,
                args.fuzzification,
            )
            alg_times.append(timeit(fit, *alg_args))
        else:
            alg_args = (
                args.num_centroids,
                args.workers,
                data,
                alg,
                args.max_iter,
                args.fuzzification,
            )
            alg_times.append(timeit(fit_multiprocess, *alg_args))

    utils.plot_time_results(data_lengths, 'Data length', alg_times)
    print(data_lengths)
    print(alg_times)
