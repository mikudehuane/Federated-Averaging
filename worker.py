import os

import torch
from torch import optim
from torch.utils.data import sampler
from torch.utils.tensorboard import SummaryWriter

from widgets import init_pars
import os.path as osp


class Worker:
    def __init__(self, **pars):
        """
        :param train_loader_list: list of loaders in each epoch
        :param pars:
            use_cuda(init to torch.cuda.is_available())
        """
        self.train_loader_list = []
        self.num_train_list = []

        self.__init_pars(pars)

    def __init_pars(self, pars):
        self.pars = dict(use_cuda=torch.cuda.is_available())
        init_pars(dst=self.pars, src=pars)

    def get_train_loss(self, model, criterion, loader_ind):
        use_cuda = self.pars['use_cuda']
        if use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        train_iter = iter(self.train_loader_list[loader_ind])
        model.to(device)
        inputs, labels = train_iter.next()
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        return loss

    def train_(self, model, criterion, loader_ind, current_lr, num_its):
        """
        new grads are logged in model itself
        without zero_grad
        :param loader_ind: use train_loader_list[loader_ind]
        :param num_its: the number of iterations to train locally
        :param current_lr: current learning rate
        :param model: model to be trained in place
        :param criterion:
        """
        use_cuda = self.pars['use_cuda']
        if use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        loader = self.train_loader_list[loader_ind]

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
