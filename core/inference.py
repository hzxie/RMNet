# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:30:26
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-06 09:08:11
# @Email:  cshzxie@gmail.com

import logging
import os
import torch

import utils.data_loaders
import utils.helpers

from tqdm import tqdm

from models.stm import STM


def inference_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    test_data_loader = torch.utils.data.DataLoader(
        dataset=utils.data_loaders.DatasetCollector.get_dataset(
            cfg, cfg.DATASET.TEST_DATASET, utils.data_loaders.DatasetSubset.TEST),
        batch_size=1,
        num_workers=cfg.CONST.N_WORKERS,
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
    for idx, (video_name, n_objects, frames, masks, target_objects) in enumerate(test_data_loader):
        with torch.no_grad():
            est_probs = utils.helpers.multi_scale_inference(cfg, stm, frames, masks,
                                                            target_objects, n_objects)

            video_name = video_name[0]
            output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark', cfg.CONST.EXP_NAME,
                                         video_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            frames = frames[0]
            est_masks = torch.argmax(est_probs[0], dim=1)
            n_frames = est_masks.size(0)
            for i in tqdm(range(n_frames), leave=False, desc=video_name):
                frame = frames[i]
                est_mask = est_masks[i]
                segmentation = utils.helpers.get_segmentation(frame, est_mask, {
                    'mean': cfg.CONST.DATASET_MEAN,
                    'std': cfg.CONST.DATASET_STD,
                })
                segmentation.save(os.path.join(output_folder, '%05d.png' % i))
