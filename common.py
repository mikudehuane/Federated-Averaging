import os.path as osp
from torchvision import transforms

cifar10_classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# data_dir = osp.join('data')
data_dir = osp.join('D:\\Projects\\data')

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

NUM_CIFAR10_CLASSES = 10
NUM_CIFAR10_TRAIN = 50000
NUM_CIFAR10_TEST = 10000
