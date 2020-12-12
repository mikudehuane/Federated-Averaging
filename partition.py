import random
import os.path as osp
import os
import pickle as pkl
from math import floor

from common import *
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from common import *


def partition_class(train_set, shuffle=True):
    """
    :param shuffle: whether to shuffle data in one class
    :return: inds inds[i] are data indices with class i
    """
    inds = [[] for _ in range(NUM_CIFAR10_CLASSES)]

    for i, (fig, label) in enumerate(train_set):
        inds[label].append(i)

    if shuffle:
        for inds_i in inds:
            random.shuffle(inds_i)

    return inds


def partition(raw_data_dir=data_dir, output_data_dir=osp.join('data', 'debug'), num_clients=100, num_samples_base=5, num_samples_clamp_thres=2):
    """
    Each class with num_clients // NUM_CIFAR10_CLASSES clients
    :param num_clients: should be multiple of 10 (CIFAR10 class number)
    :param num_samples_base: control the degree of data imbalance, larger means imbalance lower
    :param num_samples_clamp_thres: control the degree of data imbalance, trivial
    """
    # parameter check
    if num_samples_base <= num_samples_clamp_thres:
        print("num_samples_bas should be larger than num_samples_clamp_thres")
        raise AssertionError()
    if num_clients % NUM_CIFAR10_CLASSES != 0:
        print('num_clients should be a multiple of NUM_CIFAR10_CLASSES')
        raise AssertionError() 
    if osp.exists(output_data_dir):
        print('Output data directory exists!')
        raise AssertionError()
    else:
        os.makedirs(output_data_dir)

    NUM_SAMPLES1CLASS = 5000    
    num_clients1class = num_clients // NUM_CIFAR10_CLASSES

    train_set = torchvision.datasets.CIFAR10(root=raw_data_dir, train=True, download=True, transform=transform)

    inds = partition_class(train_set)

    for class_ in range(NUM_CIFAR10_CLASSES):
        # unbalance partition
        while True:
            num_samples_list = num_samples_base + np.random.randn(num_clients1class)
            num_samples_list[num_samples_list < num_samples_base - num_samples_clamp_thres] = \
                num_samples_base - num_samples_clamp_thres
            num_samples_list[num_samples_list > num_samples_base + num_samples_clamp_thres] = \
                num_samples_base + num_samples_clamp_thres
            sum_weight = num_samples_list.sum()
            num_samples_list /= sum_weight  # normalize

            pre_weight = 0.0
            accumulated = [0.0]
            for end in range(num_clients1class):
                pre_weight = pre_weight + num_samples_list[end]
                accumulated.append(pre_weight)
            accumulated[-1] = 1.0
            accumulated = np.round(np.array(accumulated) * NUM_SAMPLES1CLASS).astype(int)

            num_samples_list = [accumulated[i + 1] - accumulated[i] for i in range(num_clients1class)]
            num_samples_list = np.array(num_samples_list)
            if (num_samples_list <= 0).sum() == 0:
                break
            else:
                print(f'In class {class_}, client with no data exists, retrying...')

        start_train = 0
        for worker_ind in range(num_clients1class):
            worker_identifier = worker_ind + num_clients1class * class_
            worker_dir = osp.join(output_data_dir, str(worker_identifier))
            os.makedirs(worker_dir)

            num_train_samples = num_samples_list[worker_ind]

            for ind in inds[class_][start_train: start_train + num_train_samples]:
                data = train_set[ind]
                with open(osp.join(worker_dir, str(ind)), 'wb') as f:
                    pkl.dump(data, f)

            start_train += num_train_samples


def _test():
    partition(num_clients=100, output_data_dir=osp.join('data', 'debug100-1'))


if __name__ == '__main__':
    _test()
