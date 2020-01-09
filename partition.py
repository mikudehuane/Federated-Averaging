import random
import os.path as osp
import pickle as pkl
from math import floor

from common import *
import numpy as np


def partition_data(train_set, test_set, path=osp.join('cache', 'inds.pkl'), overwrite=False, shuffle=True):
    """
    :param train_set:
    :param test_set:
    :param path:
    :param overwrite:
    :param shuffle:
    :return: inds: dict inds['train'][i] data indices in train_set with class i
    """
    if osp.exists(path) and not overwrite:
        with open(path, 'rb') as f:
            inds = pkl.load(f)
            print(f"inds loaded from {path}")
    else:
        inds = {'train': [[] for _ in range(NUM_CIFAR10_CLASSES)],
                'test': [[] for _ in range(NUM_CIFAR10_CLASSES)]}
        # inds['train'][i]: data indices with class i (train)
        for i, (fig, label) in enumerate(train_set):
            inds['train'][label].append(i)
        for i, (fig, label) in enumerate(test_set):
            inds['test'][label].append(i)
        with open(path, 'wb') as f:
            pkl.dump(inds, f)
            print(f"inds dumped into {path}")

    if shuffle:
        for inds_i in inds['train']:
            random.shuffle(inds_i)
        for inds_i in inds['test']:
            random.shuffle(inds_i)

    return inds


def partition_group(inds, num_classes1group, verbose=False):
    """
    partition the inds returned by partition_data further into 10**? groups
    :param verbose:
    :param num_classes1group:
    :param inds:
    :return: groups contrast shape as inds
    """
    num_types = NUM_CIFAR10_CLASSES ** num_classes1group
    num_samples1group1class = {
        'train': (NUM_CIFAR10_TRAIN // NUM_CIFAR10_CLASSES) //
                 (num_types * num_classes1group // NUM_CIFAR10_CLASSES),
        'test': (NUM_CIFAR10_TEST // NUM_CIFAR10_CLASSES) //
                (num_types * num_classes1group // NUM_CIFAR10_CLASSES)
    }

    if verbose:
        debug_train_inds = np.arange(NUM_CIFAR10_TRAIN)  # check how many inds covered
        debug_test_inds = np.arange(NUM_CIFAR10_TEST)

    current_inds = {  # log pointer to inds
        'train': [0 for _ in range(NUM_CIFAR10_CLASSES)],
        'test': [0 for _ in range(NUM_CIFAR10_CLASSES)]
    }

    groups = []
    for it_sum in range(num_types):
        inds_this = {'train': [], 'test': []}
        for i in range(num_classes1group):
            # we process class ci in this iteration
            # c0 = 0 c1 = 0 c2 = 0 -> c0 = 1 c1 = 0 c2 = 0 -> c0 = 2 c1 = 0 c2 = 0
            ci = floor((it_sum % (NUM_CIFAR10_CLASSES ** (i + 1))) / (NUM_CIFAR10_CLASSES ** i))

            inds_this['train'].extend(inds['train'][ci][current_inds['train'][ci]:
                                                        current_inds['train'][ci] +
                                                        num_samples1group1class['train']])
            inds_this['test'].extend(inds['test'][ci][current_inds['test'][ci]:
                                                      current_inds['test'][ci] +
                                                      num_samples1group1class['test']])
            # update pointer
            current_inds['train'][ci] += num_samples1group1class['train']
            current_inds['test'][ci] += num_samples1group1class['test']
        inds_this['train'] = np.array(inds_this['train'])
        inds_this['test'] = np.array(inds_this['test'])
        if verbose:
            debug_train_inds[inds_this['train']] = -1
            debug_test_inds[inds_this['test']] = -1
            print(f'debug_train_inds: {(debug_train_inds == -1).sum()}')
            print(f'debug_test_inds: {(debug_test_inds == -1).sum()}')
        groups.append(inds_this)
    return groups