import os

import torch
from torch import optim
from torch.utils.data import sampler
from torch.utils.tensorboard import SummaryWriter

from common import use_cuda
from widgets import init_pars
import os.path as osp


class Worker:
    def __init__(self):
        self.train_loader = None
        self.num_train = None

    def train_(self, model, criterion, current_lr, num_its):
        """
        new grads are logged in model itself
        without zero_grad
        :param num_its: the number of iterations to train locally
        :param current_lr: current learning rate
        :param model: model to be trained in place
        :param criterion:
        """
        loader = self.train_loader

        if use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=current_lr)

        count = 0
        while True:
            for _, data in enumerate(loader, 0):  # _ start from 0
                inputs, labels = data
                if use_cuda:
                    inputs = inputs.to(device=torch.device('cuda'))
                    labels = labels.to(device=torch.device('cuda'))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                count += 1
                if count == num_its:
                    return
