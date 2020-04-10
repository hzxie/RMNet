# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:30:26
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-10 14:50:58
# @Email:  cshzxie@gmail.com

import logging
import torch

import utils.data_loaders
import utils.helpers

from models.stm import STM


def inference_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                   batch_size=1,
                                                   num_workers=cfg.CONST.N_WORKERS,
                                                   collate_fn=utils.data_loaders.collate_fn,
                                                   pin_memory=True,
                                                   shuffle=False)

    # Setup networks and initialize networks
    stm = STM(cfg)

    if torch.cuda.is_available():
        stm = torch.nn.DataParallel(stm).cuda()

    # Load the pretrained model from a checkpoint
    logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    stm.load_state_dict(checkpoint['stm'])

    # Switch models to evaluation mode
    stm.eval()

    # The inference loop
    for idx, (video_name, n_objects, frames, _) in enumerate(test_data_loader):
        video_name = video_name[0]
        n_objects = n_objects[0]
        frames = utils.helpers.var_or_cuda(frames[0])

        print(frames.shape)