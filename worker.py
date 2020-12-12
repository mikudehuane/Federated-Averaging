import os

import torch
from torch import optim
from torch.utils.data import sampler
from torch.utils.tensorboard import SummaryWriter

from common import use_cuda
from widgets import init_pars
import os.path as osp
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import random
from copy import deepcopy


class LocalDataset(Dataset):

    def __init__(self, root_dir):
        """
        :param root_dir:<str> the root directory files saved in
        """
        super()
        self.root_dir = root_dir

        for i, (root, dirs, files) in enumerate(os.walk(self.root_dir)):
            if i == 0:
                self.index_entries = [osp.join(root, filen) for filen in files] 
            else:
                print('Directory structure not expected!')
                raise AssertionError()
        random.shuffle(self.index_entries)

    def __len__(self):
        return len(self.index_entries)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: (resized img, class)
        """
        with open(self.index_entries[idx], 'rb') as f:
            data = pkl.load(f)

        return data


class Worker:
    def __init__(self, identifier, dir=osp.join('data', 'debug')):
        self.data_dir = osp.join(dir, str(identifier))
        if not osp.exists(self.data_dir):
            print(f"Given worker {self.data_dir} doesn't have its data partitioned")
            raise AssertionError()
        self.dataset = LocalDataset(self.data_dir)

    @property
    def num_samples(self):
        return len(self.dataset)

    def train(self, model, criterion, current_lr, num_epoches, batch_size=5, verbose=False):
        """
        new grads are logged in model itself
        :param num_its: the number of iterations to train locally
        :param current_lr: current learning rate
        :param model: model to be trained in place
        :param criterion:
        :return: model update
        """
        running_model = deepcopy(model)

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        if use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        running_model = running_model.to(device)
        optimizer = optim.SGD(running_model.parameters(), lr=current_lr)

        for epoch in range(num_epoches):
            for it, data in enumerate(loader):  # _ start from 0
                inputs, labels = data
                if use_cuda:
                    inputs = inputs.to(device=torch.device('cuda'))
                    labels = labels.to(device=torch.device('cuda'))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = running_model(inputs)
                loss = criterion(outputs, labels)

                if verbose:
                    print(f'Epoch: {epoch}, it: {it}, loss: {loss.item()}')

                loss.backward()
                optimizer.step()
        
        return [new_par.data - old_par.data for new_par, old_par in zip(running_model.parameters(), model.parameters())]


def _test():
    worker = Worker(0)
    from model import CNNCifar
    from torch import nn

    net = CNNCifar()
    worker.train_(net, nn.CrossEntropyLoss(), 0.01, 10, 5, True)


if __name__ == '__main__':
    _test()
