import os
import pickle as pkl
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(current_it, model, chain2models, ave_models, logger, dire):
    meta_dict = dict(
        current_it=current_it
    )
    with open(osp.join(dire, 'meta.pkl'), 'wb') as f:
        pkl.dump(meta_dict, f)

    logger.flush()

    torch.save(model.state_dict(), osp.join(dire, 'model.pth'))

    if chain2models:
        for i, chain2model in enumerate(chain2models):
            torch.save(chain2model.state_dict(), osp.join(dire, 'chain2model' + str(i) + '.pth'))

    for i, ave_model in enumerate(ave_models):
        torch.save(ave_model.state_dict(), osp.join(dire, 'ave_model' + str(i) + '.pth'))

    print('*** checkpoint saved to directory: {}'.format(dire))


def load_checkpoint(dire, model, chain2models, ave_models, logger):
    with open(osp.join(dire, 'meta.pkl'), 'rb') as f:
        meta_dict = pkl.load(f)
    current_it = meta_dict['current_it']

    logger.close()
    logger.summaryWriter = SummaryWriter(log_dir=logger.tb_dir, purge_step=current_it)

    model_state_dict = torch.load(osp.join(dire, 'model.pth'))
    model.load_state_dict(model_state_dict)

    for i, ave_model in enumerate(ave_models):
        model_state_dict = torch.load(osp.join(dire, 'ave_model' + str(i) + '.pth'))
        ave_model.load_state_dict(model_state_dict)

    if chain2models:
        for i, chain2model in enumerate(chain2models):
            model_state_dict = torch.load(osp.join(dire, 'chain2model' + str(i) + '.pth'))
            chain2model.load_state_dict(model_state_dict)

    print('*** checkpoint loaded from directory: {}'.format(dire))
    print('*** training restart from iteration: {}'.format(current_it))
    return current_it
