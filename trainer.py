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
from model import CNNCifar
from worker import Worker
from checkpoint import save_checkpoint, load_checkpoint
from partition import *
from widgets import init_pars, get_spare_dir, aggregate_model, zero_model, clamp
from worker import Worker


class Trainer:

    def __init__(self, workers_dir, test_set, checkpoint_dir, **pars):
        """
        :param workers_dir: the directory your partitioned data resides
        :param pars:
            *** num_epoches: aggregate every ... epoches
            *** learning_rate
            *** num_clients1round 
            lr_decay, decay_step_size: decay lr_decay every decay_step_size
            num_rounds
            train_batch_size: for each worker
            print_every: test every ...
            verbose: print verbose message?
        """
        self.workers_dir = workers_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.pars = dict(
            num_epoches=20,
            learning_rate=1e-2, 
            num_clients1round=10,
            num_rounds=10000,
            lr_decay=1, decay_step_size=1,
            train_batch_size=50, test_batch_size=100,
            print_every=1, checkpoint_interval=1000,
            verbose=False
        )
        init_pars(self.pars, pars)

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.pars['test_batch_size'],
                                                  shuffle=False)

        self.workers = []
        self.num_total_samples = 0
        self.num_clients = 0
        for root, dirs, files in os.walk(self.workers_dir):
            dirs = sorted(dirs, key=lambda x: int(x))
            for dire in dirs:
                this_worker = Worker(dire, self.workers_dir)
                self.workers.append(this_worker)
                self.num_total_samples += this_worker.num_samples
                self.num_clients += 1
            break

    def train_(self, criterion, logger, model):
        """
        *** only learning_rate and current_loader_ind need to be loaded at checkpoint
        K: the number of active clients
        """
        pars =  self.pars
        num_epoches = pars['num_epoches']
        learning_rate = pars['learning_rate'] 
        num_clients1round = pars['num_clients1round']
        num_rounds = pars['num_rounds']
        lr_decay = pars['lr_decay']
        decay_step_size = pars['decay_step_size']
        train_batch_size = pars['train_batch_size']
        print_every = pars['print_every']
        checkpoint_interval = pars['checkpoint_interval']
        verbose = pars['verbose']
        checkpoint_dir = self.checkpoint_dir
        num_total_samples = self.num_total_samples
        num_clients = self.num_clients

        logger.add_meta_data(pars, 'training')

        if use_cuda:
            model = model.to(torch.device('cuda'))
        else:
            model = model.to(torch.device('cpu'))

        if osp.exists(osp.join(checkpoint_dir, 'meta.pkl')):
            current_round = load_checkpoint(checkpoint_dir, model, logger)
        else:
            current_round = 0

        while True:
            current_lr = learning_rate * (lr_decay ** (current_round // decay_step_size))
            print(f"current_round={current_round}, current_lr={current_lr}", end='\r')

            global_model = deepcopy(model)

            # set the number of  active clients
            workers_this = np.random.choice(self.workers, num_clients1round, replace=False)
            for worker in workers_this:
                local_update = worker.train(model, criterion, current_lr=current_lr, num_epoches=num_epoches, batch_size=train_batch_size)
                aggregate_model(global_model, local_update, 1,
                                1 / num_clients1round)
            model = global_model
            logger.add_scalar('model par for debug', list(model.parameters())[0][0][0][0][0], current_round)

            if current_round % print_every == 0:
                fed_acc = self.test_model(model)
                print('%d fedavg test acc: %.3f%%' % (current_round, fed_acc * 100.0) + ' ' * 10)
                logger.add_scalar('test accuracy', fed_acc, current_round)

            if current_round % checkpoint_interval == 0:
                save_checkpoint(current_round, model, logger, checkpoint_dir)

            if current_round == num_rounds:
                print('Finished Training')
                return

            current_round += 1

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
    tb_fdir = osp.join('runs', 'tensorboard', 'isolation')
    other_fdir = osp.join('runs', 'my-logger')
    checkpoint_fdir = osp.join('runs', 'checkpoint')

    cdir = 'debug3'

    tb_dir = osp.join(tb_fdir, cdir)
    other_dir = osp.join(other_fdir, cdir)
    checkpoint_dir = osp.join(checkpoint_fdir, cdir)

    model = CNNCifar()

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform)

    trainer = Trainer(
        workers_dir=osp.join('data', 'debug'),
        test_set=test_set,
        checkpoint_dir=checkpoint_dir,
        num_epoches=1,
        learning_rate=0.1,
        num_clients1round=10,
        num_rounds=10000,
        lr_decay=1,
        decay_step_size=1,
        train_batch_size=1000,
        test_batch_size=100,
        print_every=1,
        checkpoint_interval=1,
        verbose=False
    )

    logger = Logger(tb_dir=tb_dir, other_dir=other_dir)

    trainer.train_(nn.CrossEntropyLoss(), logger, model)

if __name__ == '__main__':
    _test()
