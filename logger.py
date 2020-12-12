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

    def add_scalar(self, tag, value, it_num):
        self.summaryWriter.add_scalar(tag, value, it_num)

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