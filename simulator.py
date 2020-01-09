import math
import os
import random
import traceback
from math import floor

import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
import os.path as osp
import pickle as pkl
import numpy as np
from copy import deepcopy

from torch import nn
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from common import *
from copy import deepcopy

from logger import Logger
from models import CNNCifar
from worker import Worker
from checkpoint import save_checkpoint, load_checkpoint
from partition import partition_group, partition_data
from widgets import init_pars, get_spare_dir, aggregate_model, zero_model, clamp


class Simulator:

    def __init__(self, checkpoint_dir, **pars):
        """
        :param pars:
            *** I=20: aggregate every I iterations
            *** N=200: N workers
            *** num_classes1group=3:
                we partition dataset into NUM_CIFAR10_CLASSES ** num_classes1group groups,
                each group with this number classes
            train_batch_size: for each worker
            verbose: False print verbose message
            num_samples_base, num_samples_clamp_thres: for unbalanced partitioning
        """
        self.pars = dict(I=20, N=200,
                         num_classes1group=3,
                         train_batch_size=50, test_batch_size=100,
                         verbose=False,
                         num_samples_base=5.0,
                         num_samples_clamp_thres=2.0)
        init_pars(self.pars, pars)

        num_classes1group = self.pars['num_classes1group']
        N = self.pars['N']
        verbose = self.pars['verbose']
        train_batch_size = self.pars['train_batch_size']
        num_samples_base = self.pars['num_samples_base']
        num_samples_clamp_thres = self.pars['num_samples_clamp_thres']
        test_batch_size = self.pars['test_batch_size']

        if self.pars['num_samples_base'] < self.pars['num_samples_clamp_thres'] + 1:
            print("num_samples_bas should be larger than num_samples_clamp_thres")
            raise AssertionError()

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                      download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                     download=True, transform=transform)
        self.test_loader = DataLoader(self.test_set, batch_size=test_batch_size, shuffle=False)
        if self.pars['verbose']:
            print(f"Length of cifar10 train set: {len(self.train_set)}")
            print(f"Length of cifar10 test set: {len(self.test_set)}")

        # prepare workers
        # device train_set and test_set with respect to class
        inds = partition_data(self.train_set, self.test_set, path=osp.join(cache_dir, 'inds.pkl'))

        # partition data into groups
        groups = partition_group(inds, num_classes1group, verbose=verbose)
        train_group_cat = np.array([group['train'] for group in groups]).flatten()
        self.num_total_samples = len(train_group_cat)

        # partition groups across phase
        num_groups = len(groups)
        assert num_groups >= 1

        if osp.exists(osp.join(checkpoint_dir, 'workers.pkl')):  # breakpoint detected:
            with open(osp.join(checkpoint_dir, 'workers.pkl'), 'rb') as f:
                self.workers = pkl.load(f)
            with open(osp.join(checkpoint_dir, 'test_loader.pkl'), 'rb') as f:
                self.test_loader = pkl.load(f)
            print('*** checkpoint detected!')
            print('self.workers loaded from {}'.format(osp.join(checkpoint_dir, 'inds.pkl')))
            print('self.test_loader loaded from {}'.format(osp.join(checkpoint_dir, 'test_loaders.pkl')))
        else:
            # further partition groups across workers
            workers = [Worker() for _ in range(N)]

            # judge num_samples for each worker
            while True:
                num_samples_list = num_samples_base + np.random.randn(N)
                num_samples_list[num_samples_list < num_samples_base - num_samples_clamp_thres] = \
                    num_samples_base - num_samples_clamp_thres
                num_samples_list[num_samples_list > num_samples_base + num_samples_clamp_thres] = \
                    num_samples_base + num_samples_clamp_thres
                sum_weight = num_samples_list.sum()
                num_samples_list /= sum_weight  # normalize

                pre_weight = 0.0
                accumulated = [0.0]
                for end in range(N):
                    pre_weight = pre_weight + num_samples_list[end]
                    accumulated.append(pre_weight)
                accumulated[-1] = 1.0
                accumulated = np.round(np.array(accumulated) * len(train_group_cat)).astype(int)

                num_samples_list = [accumulated[i + 1] - accumulated[i] for i in range(N)]
                num_samples_list = np.array(num_samples_list)
                if (num_samples_list <= 0).sum() == 0:
                    break

            start_train = 0
            for ind, worker in enumerate(workers):
                num_train_samples = num_samples_list[ind]
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    train_group_cat[start_train: start_train + num_train_samples]
                )
                train_loader_this = \
                    DataLoader(self.train_set, batch_size=min(train_batch_size, num_train_samples),
                               sampler=train_sampler)

                worker.train_loader = train_loader_this
                worker.num_train = int(num_train_samples)

                start_train += num_train_samples

            self.workers = np.array(workers)
            with open(osp.join(checkpoint_dir, 'workers.pkl'), 'wb') as f:
                pkl.dump(self.workers, f)
                print('self.workers saved to {}'.format(osp.join(checkpoint_dir, 'inds.pkl')))
            with open(osp.join(checkpoint_dir, 'test_loader.pkl'), 'wb') as f:
                pkl.dump(self.test_loader, f)
                print('self.test_loader saved to {}'.format(osp.join(checkpoint_dir, 'test_loader.pkl')))

    def train_(self, criterion, logger, model, **pars):
        """
        grad_cache will be updated in-place
        *** only learning_rate and current_loader_ind need to be loaded at checkpoint
        K: the number of active clients
        """
        default_pars = dict(learning_rate=1e-2, K=10,
                            num_its=5000,
                            lr_decay=0.5, decay_step_size=1000,
                            print_every=50, checkpoint_interval=1000)
        init_pars(default_pars, pars)
        pars = default_pars

        K = pars['K']
        learning_rate = pars['learning_rate']
        num_its = pars['num_its']
        lr_decay = pars['lr_decay']
        decay_step_size = pars['decay_step_size']
        print_every = pars['print_every']
        checkpoint_interval = pars['checkpoint_interval']

        I = self.pars['I']
        N = self.pars['N']

        checkpoint_dir = self.checkpoint_dir

        logger.add_meta_data(pars, 'training')
        logger.add_meta_data(self.pars, 'simulation')

        if use_cuda:
            model = model.to(torch.device('cuda'))
        else:
            model = model.to(torch.device('cpu'))

        if osp.exists(osp.join(checkpoint_dir, 'meta.pkl')):
            current_it = load_checkpoint(checkpoint_dir, model, logger)
        else:
            current_it = 0

        while True:
            current_lr = learning_rate * (lr_decay ** (current_it // decay_step_size))
            print(f"current_it={current_it}, current_lr={current_lr}", end='\r')

            global_model = deepcopy(model)
            zero_model(global_model)

            # set the number of  active clients
            idxs_users = np.random.choice(range(N), K, replace=False)
            for idx in idxs_users:
                worker = self.workers[idx]
                local_model = deepcopy(model)
                worker.train_(local_model, criterion, current_lr=current_lr, num_its=I)
                aggregate_model(global_model, local_model, 1,
                                N / K * (worker.num_train / self.num_total_samples))
            model = global_model
            logger.add_train_loss(list(model.parameters())[0][0][0][0][0], current_it, 'model-par')

            if current_it % print_every == 0:
                # fedavg
                fed_acc_array = self.test_model(model)
                fed_acc = np.array(fed_acc_array).mean()
                print('%d fedavg test acc: %.3f%%' % (current_it, fed_acc * 100.0))
                logger.add_test_acc(fed_acc, current_it, 'fedavg')

            if current_it % checkpoint_interval == 0:
                save_checkpoint(current_it, model, logger, checkpoint_dir)

            if current_it == num_its:
                print('Finished Training')
                return

            current_it += 1

    def test_model(self, model):
        num_right = 0
        num_samples = 0
        for inputs, labels in self.test_loader:
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            preds = outputs.argmax(axis=1)
            num_right += (preds == labels).sum().item()
            num_samples += len(labels)
        return num_right / num_samples


def _test():
    from matplotlib import pyplot as plt

    I = 10
    N = 100
    train_batch_size = 2

    tb_fdir = osp.join('runs', 'tensorboard', 'isolation')
    other_fdir = osp.join('runs', 'my-logger')
    checkpoint_fdir = osp.join('runs', 'checkpoint')

    cdir = 'train'
    tb_dir = osp.join(tb_fdir, cdir)
    other_dir = osp.join(other_fdir, cdir)
    checkpoint_dir = osp.join(checkpoint_fdir, cdir)

    model = CNNCifar()

    simulator = Simulator(checkpoint_dir=checkpoint_dir, I=I, N=N, train_batch_size=train_batch_size)

    logger = Logger(tb_dir=tb_dir, other_dir=other_dir)

    # count_train, count_test = 0, 0
    # for worker_ind, worker in enumerate(simulator.workers):
    #     count_train += simulator.workers[worker_ind].num_train
    #     plt.scatter(simulator.workers[worker_ind].num_train, 0)
    # plt.show()
    # print(count_train, count_test)

    simulator.train_(criterion=torch.nn.CrossEntropyLoss(),
                     model=model,
                     logger=logger,
                     learning_rate=1e-1,
                     K=10,
                     num_its=8000,
                     lr_decay=0.9,
                     decay_step_size=1000,
                     print_every=10,
                     checkpoint_interval=5)


if __name__ == '__main__':
    _test()
