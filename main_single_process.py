import argparse
import time

from unified import fit, calculate_membership_mat
from utils import read_data, plot_centroids
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--num_centroids', type=int, default=3)
parser.add_argument('--alg', type=str, choices=('KM', 'FCM'), default='FCM')
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--input_path', type=str, default='./input/Iris-150.txt')


if __name__ == '__main__':
    args = parser.parse_args()
    show_accuracy = True

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
    membership_mat = calculate_membership_mat(
        data, centroids, clusters=3, alg=alg,
    )
    if show_accuracy is True:
        np_data = data
        all_labels = plant_types
        all_labels = np.where(all_labels == 'Iris-setosa', int(0), all_labels)
        all_labels = np.where(all_labels == 'Iris-versicolor', 1, all_labels)
        all_labels = np.where(all_labels == 'Iris-virginica', 2, all_labels)
        membership_mat_max = membership_mat.argmax(1)
        print(membership_mat_max)
        dictCount0 = dict()
        dictCount1 = dict()
        dictCount2 = dict()
        dictCount0[0] = 0
        dictCount0[1] = 0
        dictCount0[2] = 0
        dictCount1[0] = 0
        dictCount1[1] = 0
        dictCount1[2] = 0
        dictCount2[0] = 0
        dictCount2[1] = 0
        dictCount2[2] = 0
        cnt0, cnt1, cnt2 = 0, 0, 0
        for i in range(len(all_labels)):
            if membership_mat_max[i] == 0:
                cnt0 += 1
            elif membership_mat_max[i] == 1:
                cnt1 += 1
            elif membership_mat_max[i] == 2:
                cnt2 += 1
            if all_labels[i] == '0':
                dictCount0[membership_mat_max[i]] += 1
            elif all_labels[i] == '1':
                dictCount1[membership_mat_max[i]] += 1
            elif all_labels[i] == '2':
                dictCount2[membership_mat_max[i]] += 1

        print(dictCount0, dictCount1, dictCount2)
        print(dictCount0.values(), dictCount1.values(), dictCount2.values())
        maxDict0 = (list(dictCount0.values()).index(max(dictCount0.values())), max(dictCount0.values()))
        maxDict1 = (list(dictCount1.values()).index(max(dictCount1.values())), max(dictCount1.values()))
        maxDict2 = (list(dictCount2.values()).index(max(dictCount2.values())), max(dictCount2.values()))
        print(maxDict0)
        print(maxDict1)
        print(maxDict2)
        lastDict = dict()
        lastDict[maxDict0[0]] = maxDict0[1]
        if maxDict1[0] in lastDict:
            lastDict[maxDict2[0]] = maxDict2[1]
            if 0 not in lastDict:
                lastDict[0] = list(dictCount1.values())[0]
            elif 1 not in lastDict:
                lastDict[1] = list(dictCount1.values())[1]
            elif 2 not in lastDict:
                lastDict[2] = list(dictCount1.values())[2]
        else:
            lastDict[maxDict1[0]] = maxDict1[1]
            if 0 not in lastDict:
                lastDict[0] = list(dictCount2.values())[0]
            elif 1 not in lastDict:
                lastDict[1] = list(dictCount2.values())[1]
            elif 2 not in lastDict:
                lastDict[2] = list(dictCount2.values())[2]
        print(lastDict)
        precision_0 = lastDict[0] / eval('cnt' + str(maxDict0[0]))
        precision_1 = lastDict[1] / eval('cnt' + str(maxDict1[0]))
        precision_2 = lastDict[2] / eval('cnt' + str(maxDict2[0]))
        print("precisions:", str(lastDict[0]) + '/' + str(eval('cnt' + str(maxDict0[0]))),
              str(lastDict[1]) + '/' + str(eval('cnt' + str(maxDict1[0]))),
              str(lastDict[2]) + '/' + str(eval('cnt' + str(maxDict2[0]))))
        precision_total = (lastDict[0] + lastDict[1] + lastDict[2]) / (
                    eval('cnt' + str(maxDict0[0])) + eval('cnt' + str(maxDict1[0])) + eval('cnt' + str(maxDict2[0])))

        print("precisions:", precision_0, precision_1, precision_2)
        print("precision total:", precision_total)

        recall_0 = lastDict[0] / (np_data.shape[0] / 3)
        recall_1 = lastDict[1] / (np_data.shape[0] / 3)
        recall_2 = lastDict[2] / (np_data.shape[0] / 3)
        recall_total = (lastDict[0] + lastDict[1] + lastDict[2]) / np_data.shape[0]
        print("recalls:", str(lastDict[0]) + '/' + str(np_data.shape[0] / 3),
              str(lastDict[1]) + '/' + str(np_data.shape[0] / 3),
              str(lastDict[2]) + '/' + str(np_data.shape[0] / 3))
        print("recalls:", recall_0, recall_1, recall_2)
        print("recall total:", recall_total)



    end = time.time()
    print(f'Algorithm took: {end - start} seconds')

    plot_centroids(data, centroids)


