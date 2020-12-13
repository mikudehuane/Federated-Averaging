import os
import os.path as osp

from torch.utils.tensorboard import SummaryWriter

from widgets import get_spare_dir
import pickle as pkl


class Logger:
    def __init__(self, tb_dir, other_dir):
        """
        :param tb_dir: father tb_dir
        :param other_dir: child tb_dir will be indexed by 0, 1, 2, ...
        """
        self.tb_dir = tb_dir
        self.other_dir = other_dir

        if not osp.exists(self.tb_dir):
            os.makedirs(self.tb_dir)
        if not osp.exists(self.other_dir):
            os.makedirs(self.other_dir)

        self.summaryWriter = SummaryWriter(log_dir=self.tb_dir)

    def log_num_samples(self, phase_ind, worker_ind, num_train_samples, num_test_samples):
        with open(osp.join(self.other_dir, 'num-samples_log.txt'), 'a') as f:
            f.write(f"phase:{phase_ind}, worker:{worker_ind}, train:{num_train_samples}, test:{num_test_samples}\n")

    def add_train_loss(self, loss, it_num, tag):
        self.summaryWriter.add_scalar(tag + "-training loss", loss, it_num)

    def add_test_acc(self, acc, it_num, tag):
        self.summaryWriter.add_scalar(tag + "-test Accuracy", acc, it_num)

    def add_meta_data(self, pars, tag):
        with open(osp.join(self.other_dir, 'parameters.txt'), 'a') as f:
            f.write(tag + '\n')
            for k, v in pars.items():
                f.write(str(k) + ':' + str(v) + '\n')
            f.write('-' * 80 + '\n')

    def close(self):
        self.summaryWriter.close()

    def flush(self):
        self.summaryWriter.flush()
