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
from model import Net
from worker import Worker
from checkpoint import save_checkpoint, load_checkpoint
from partition import partition_group, partition_data
from widgets import init_pars, get_spare_dir, aggregate_model, zero_model, clamp


class Simulator:

    def __init__(self, checkpoint_dir, **pars):
        """
        :param pars:
            *** I=20: aggregate every I iterations
            *** E=3: data vary every E aggregations
            *** N=200: N workers
            *** num_phases=5: each period has <> distributions
            *** num_classes1group=3:
                we partition dataset into NUM_CIFAR10_CLASSES ** num_classes1group groups,
                each group with this number classes
            num_samples_std=0.3: standard deviation of samples(proportional with mean)
                the number of samples each client in each phase possesses is larger than 1
                normal distribution with std = ...
                the same worker in different phase can have different number of samples
                dist >= this or 1 will be cutted
            train_batch_size=50: for each worker
            test_batch_size=100: for each worker (in test, all data will be traversed)
            use_cuda=torch.cuda.is_available()
            cache_dir='cache' save intermediate files
            verbose=False print verbose message
        running:
            workers
        """
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.__init_pars(pars)
        self.__init_dataset()
        self.__prepare_workers()

    def __init_pars(self, pars):
        self.pars = dict(I=20, E=3, N=200,
                         num_classes1group=3,
                         num_phases=5,
                         train_batch_size=50, test_batch_size=100,
                         use_cuda=torch.cuda.is_available(),
                         cache_dir='cache',
                         verbose=False,
                         num_samples_base=5.0,
                         num_samples_clamp_thres=2.0)
        init_pars(self.pars, pars)

        if self.pars['num_samples_base'] < self.pars['num_samples_clamp_thres'] + 1:
            print("num_samples_bas should be larger than num_samples_clamp_thres")
            raise AssertionError()

        if not osp.exists(self.pars['cache_dir']):
            os.makedirs(self.pars['cache_dir'])

    def __init_dataset(self):
        verbose = self.pars['verbose']

        self.train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                      download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                     download=True, transform=transform)
        if verbose:
            print(f"Length of cifar10 train set: {len(self.train_set)}")
            print(f"Length of cifar10 test set: {len(self.test_set)}")

    def __prepare_workers(self):
        num_classes1group = self.pars['num_classes1group']
        cache_dir = self.pars['cache_dir']
        N = self.pars['N']
        num_phases = self.pars['num_phases']
        verbose = self.pars['verbose']
        use_cuda = self.pars['use_cuda']
        train_batch_size = self.pars['train_batch_size']
        test_batch_size = self.pars['test_batch_size']
        num_samples_base = self.pars['num_samples_base']
        num_samples_clamp_thres = self.pars['num_samples_clamp_thres']

        checkpoint_dir = self.checkpoint_dir

        # device train_set and test_set with respect to class
        inds = partition_data(self.train_set, self.test_set, path=osp.join(cache_dir, 'inds.pkl'))

        # partition data into groups
        groups = partition_group(inds, num_classes1group, verbose=verbose)

        # partition groups across phase
        num_groups1phase = len(groups) // num_phases
        assert num_groups1phase >= 1

        self.num_train1phase = num_groups1phase * len(groups[0]['train'])
        self.num_test1phase = num_groups1phase * len(groups[0]['test'])
        self.test_loaders = []

        if osp.exists(osp.join(checkpoint_dir, 'workers.pkl')):  # breakpoint detected:
            with open(osp.join(checkpoint_dir, 'workers.pkl'), 'rb') as f:
                self.workers = pkl.load(f)
            with open(osp.join(checkpoint_dir, 'test_loaders.pkl'), 'rb') as f:
                self.test_loaders = pkl.load(f)
            print('*** checkpoint detected!')
            print('self.workers loaded from {}'.format(osp.join(checkpoint_dir, 'inds.pkl')))
            print('self.test_loaders loaded from {}'.format(osp.join(checkpoint_dir, 'test_loaders.pkl')))
        else:
            # further partition groups across workers
            num_samples_mean_train = num_groups1phase * len(groups[0]['train']) // N

            workers = [Worker(use_cuda=use_cuda) for _ in range(N)]
            for phase_ind in range(num_phases):
                group_ind_s = phase_ind * num_groups1phase
                group_ind_e = (phase_ind + 1) * num_groups1phase
                groups_this = groups[group_ind_s: group_ind_e]

                group_cat = deepcopy(groups_this[0])
                for group_this in groups_this[1:]:
                    group_cat['train'] = np.append(group_cat['train'], group_this['train'])
                    group_cat['test'] = np.append(group_cat['test'], group_this['test'])

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
                    accumulated = np.round(np.array(accumulated) * len(group_cat['train'])).astype(int)

                    num_samples_list = [accumulated[i + 1] - accumulated[i] for i in range(N)]
                    num_samples_list = np.array(num_samples_list)
                    if (num_samples_list <= 0).sum() == 0:
                        break

                test_loader_this = DataLoader(self.test_set, batch_size=test_batch_size,
                                              sampler=SubsetRandomSampler(group_cat['test']))
                # print(len(test_loader_this))
                self.test_loaders.append(test_loader_this)

                start_train = 0
                for ind, worker in enumerate(workers):
                    num_train_samples = num_samples_list[ind]
                    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                        group_cat['train'][start_train: start_train + num_train_samples]
                    )
                    train_loader_this = \
                        DataLoader(self.train_set, batch_size=min(train_batch_size, num_train_samples),
                                   sampler=train_sampler)

                    worker.train_loader_list.append(train_loader_this)
                    worker.num_train_list.append(int(num_train_samples))

                    start_train += num_train_samples

            self.workers = np.array(workers)
            with open(osp.join(checkpoint_dir, 'workers.pkl'), 'wb') as f:
                pkl.dump(self.workers, f)
                print('self.workers saved to {}'.format(osp.join(checkpoint_dir, 'inds.pkl')))
            with open(osp.join(checkpoint_dir, 'test_loaders.pkl'), 'wb') as f:
                pkl.dump(self.test_loaders, f)
                print('self.test_loaders saved to {}'.format(osp.join(checkpoint_dir, 'test_loaders.pkl')))

    def train_(self, criterion, logger, checkpoint_dir, **pars):
        """
        grad_cache will be updated in-place
        *** only learning_rate and current_loader_ind need to be loaded at checkpoint
        :param checkpoint_dir:
        :param logger:
        :param criterion:
        :param pars:
            *** num_its: number of aggregations (not iterations)
        """
        default_pars = dict(learning_rate=1e-2,
                            num_its=5000,
                            lr_decay=0.5, decay_step_size=1000,
                            model_ori_path=osp.join('cache', 'model_ori.pth'),
                            print_every=50, checkpoint_interval=1000,
                            is2chain=False)
        init_pars(default_pars, pars)
        pars = default_pars

        learning_rate = pars['learning_rate']
        num_its = pars['num_its']
        lr_decay = pars['lr_decay']
        decay_step_size = pars['decay_step_size']
        model_ori_path = pars['model_ori_path']
        print_every = pars['print_every']
        checkpoint_interval = pars['checkpoint_interval']
        is2chain = pars['is2chain']

        I = self.pars['I']
        E = self.pars['E']
        N = self.pars['N']
        num_phases = self.pars['num_phases']
        use_cuda = self.pars['use_cuda']

        logger.add_meta_data(pars, 'training')
        logger.add_meta_data(self.pars, 'simulation')

        model = torch.load(model_ori_path)
        if use_cuda:
            model = model.to(torch.device('cuda'))
        else:
            model = model.to(torch.device('cpu'))

        ave_models = [zero_model(deepcopy(model)) for _ in range(num_phases)]
        chain2models = None
        if is2chain:
            chain2models = [deepcopy(model) for _ in range(num_phases)]

        if osp.exists(osp.join(checkpoint_dir, 'meta.pkl')):
            current_it = load_checkpoint(checkpoint_dir, model, chain2models, ave_models, logger)
        else:
            current_it = 0

        while True:
            current_loader_ind = (current_it // E) % num_phases
            current_lr = learning_rate * (lr_decay ** (current_it // decay_step_size))
            print(f"current_it={current_it}, current_loader_ind={current_loader_ind}, current_lr={current_lr}, ", end='')

            global_model = deepcopy(model)
            zero_model(global_model)
            if is2chain:
                global_chain2_model = deepcopy(chain2models[current_loader_ind])
                zero_model(global_chain2_model)

            k = 100
            idxs_users = np.random.choice(range(N), k, replace=False)
            tmp_num = 0
            for idx in idxs_users:
                worker = self.workers[idx]
                tmp_num = tmp_num + worker.num_train_list[current_loader_ind]
            for idx in idxs_users:
                worker = self.workers[idx]
                local_model = deepcopy(model)
                worker.train_(local_model, criterion, loader_ind=current_loader_ind, current_lr=current_lr, num_its=I)
                aggregate_model(global_model, local_model, 1,
                                (worker.num_train_list[current_loader_ind] / tmp_num))
                if is2chain:
                    local_chain2_model = deepcopy(chain2models[current_loader_ind])
                    worker.train_(local_chain2_model, criterion, loader_ind=current_loader_ind, current_lr=current_lr,
                                  num_its=I)
                    aggregate_model(global_chain2_model, local_chain2_model, 1,
                                    (worker.num_train_list[current_loader_ind] / tmp_num))
            model = global_model
            if is2chain:
                chain2models[current_loader_ind] = global_chain2_model

            if current_it >= 0:
                if is2chain:
                    chain1_loss = self.get_train_loss(model, criterion, current_loader_ind)
                    chain2_loss = self.get_train_loss(chain2models[current_loader_ind], criterion, current_loader_ind)
                    print(f'1: {chain1_loss}, 2: {chain2_loss}, ', end='')
                    if chain1_loss < chain2_loss:
                        print('chain1')
                        aggregate_model(ave_models[current_loader_ind], model, 1 / 2, 1 / 2)
                    else:
                        print('chain2')
                        aggregate_model(ave_models[current_loader_ind], chain2models[current_loader_ind], 1 / 2, 1 / 2)
                else:
                    aggregate_model(ave_models[current_loader_ind], model, 1 / 2, 1 / 2)
                    print('one chain')

            logger.add_train_loss(list(ave_models[0].parameters())[0][0][0][0][0].item(), current_it, 'ave_model')
            current_it += 1

            if current_it % print_every == 0:
                # fedavg
                fed_acc_array = []
                for i in range(num_phases):
                    fed_acc_array.append(self.test_model(model, i))
                fed_acc = np.array(fed_acc_array).mean()
                print('%d fedavg test acc: %.3f%%' % (current_it, fed_acc * 100.0))
                logger.add_test_acc(fed_acc, current_it, 'fedavg')

                our_acc_array = []
                for i in range(num_phases):
                    our_acc_array.append(self.test_model(ave_models[i], i))
                our_acc = np.array(our_acc_array).mean()
                print('%d our test acc: %.3f%%' % (current_it, our_acc * 100.0))
                logger.add_test_acc(our_acc, current_it, 'our')

            if current_it % checkpoint_interval == 0:
                save_checkpoint(current_it, model, chain2models, ave_models, logger, checkpoint_dir)

            if current_it == num_its:
                print('Finished Training')
                return

    def get_train_loss(self, model, criterion, current_loader_ind):
        loss = 0.0
        for worker in self.workers:
            loss += worker.get_train_loss(model, criterion, current_loader_ind)
        return loss

    def test_model(self, model, current_loader_ind):
        num_right = 0
        num_samples = 0
        for inputs, labels in self.test_loaders[current_loader_ind]:
            if self.pars['use_cuda']:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            preds = outputs.argmax(axis=1)
            num_right += (preds == labels).sum().item()
            num_samples += len(labels)
        return num_right / num_samples


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    I = 15
    E = 200
    N = 100
    num_phases = 5
    train_batch_size = 2

    use_checkpoint = False

    tb_fdir = osp.join('runs', 'tensorboard', 'isolation')
    other_fdir = osp.join('runs', 'my-logger')
    checkpoint_fdir = osp.join('runs', 'checkpoint')

    cdir = 'one chain'

    tb_dir = osp.join(tb_fdir, cdir)
    # tb_dir = get_spare_dir(tb_fdir, cdir, new=not use_checkpoint)
    # cdir_real = osp.split(tb_dir)[-1]
    other_dir = osp.join(other_fdir, cdir)
    checkpoint_dir = osp.join(checkpoint_fdir, cdir)

    cache_dir = osp.join('cache')

    model_ori_path = osp.join(cache_dir, 'model_ori.pth')
    if not osp.exists(model_ori_path):
        os.makedirs(osp.split(model_ori_path)[0], exist_ok=True)
        model = Net()
        torch.save(model, model_ori_path)

    simulator = Simulator(checkpoint_dir=checkpoint_dir,
                          num_phases=num_phases,
                          I=I, E=E, N=N,
                          train_batch_size=train_batch_size)

    logger = Logger(tb_dir=tb_dir, other_dir=other_dir)

    # for i in range(num_phases):
    #     count_train, count_test = 0, 0
    #     for worker_ind, worker in enumerate(simulator.workers):
    #         count_train += simulator.workers[worker_ind].num_train_list[i]
    #         count_test += simulator.workers[worker_ind].num_test_list[i]
    #         logger.log_num_samples(i, worker_ind,
    #                                simulator.workers[worker_ind].num_train_list[i],
    #                                simulator.workers[worker_ind].num_test_list[i])
    #         plt.scatter(simulator.workers[worker_ind].num_train_list[i],
    #                     simulator.workers[worker_ind].num_test_list[i])
    #     plt.show()
    #     print(count_train, count_test)

    simulator.train_(criterion=torch.nn.CrossEntropyLoss(),
                     logger=logger,
                     checkpoint_dir=checkpoint_dir,
                     learning_rate=1e-2,
                     num_its=10000,
                     lr_decay=0.9,
                     decay_step_size=1000,
                     model_ori_path=model_ori_path,
                     print_every=10, # 10
                     checkpoint_interval=100, # 500
                     is2chain=False)
