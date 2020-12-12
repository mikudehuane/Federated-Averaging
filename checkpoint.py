import os
import pickle as pkl
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(current_round, model, logger, dire):
    meta_dict = dict(
        current_round=current_round
    )
    with open(osp.join(dire, 'meta.pkl'), 'wb') as f:
        pkl.dump(meta_dict, f)

    logger.flush()

    torch.save(model.state_dict(), osp.join(dire, 'model.pth'))

    print('*** checkpoint saved to directory: {}'.format(dire))


def load_checkpoint(dire, model, logger):
    with open(osp.join(dire, 'meta.pkl'), 'rb') as f:
        meta_dict = pkl.load(f)
    current_round = meta_dict['current_round']

    logger.close()
    logger.summaryWriter = SummaryWriter(log_dir=logger.tb_dir, purge_step=current_round)

    model_state_dict = torch.load(osp.join(dire, 'model.pth'))
    model.load_state_dict(model_state_dict)

    print('*** checkpoint loaded from directory: {}'.format(dire))
    print('*** training restart from iteration: {}'.format(current_round))
    return current_round
