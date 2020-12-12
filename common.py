import os.path as osp
import os

import torch
from torchvision import transforms

cifar10_classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_dir = osp.join(osp.join('/root/last-avg/data'))
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

NUM_CIFAR10_CLASSES = 10
NUM_CIFAR10_TRAIN = 50000
NUM_CIFAR10_TEST = 10000
