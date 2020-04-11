# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:30:11
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-11 22:20:26
# @Email:  cshzxie@gmail.com

import logging
import torch

import utils.data_loaders
import utils.helpers

from tqdm import tqdm

from models.stm import STM
from utils.average_meter import AverageMeter
from utils.metrics import Metrics


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, stm=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.VAL),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.N_WORKERS,
                                                       collate_fn=utils.data_loaders.collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    if stm is None:
        stm = STM(cfg)

        if torch.cuda.is_available():
            stm = torch.nn.DataParallel(stm).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        stm.load_state_dict(checkpoint['stm'])

    # Switch models to evaluation mode
    stm.eval()

    # Set up loss functions
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=cfg.CONST.INGORE_IDX)

    # The testing loop
    n_videos = len(test_data_loader)
    test_losses = AverageMeter()
    test_metrics = AverageMeter(Metrics.names())

    for idx, (video_name, n_objects, frames, masks) in enumerate(test_data_loader):
        with torch.no_grad():
            est_probs = stm(frames, masks, n_objects)

            video_name = video_name[0]
            frames = frames[0]
            masks = torch.argmax(masks[0], dim=1)
            est_probs = est_probs[0]
            est_masks = torch.argmax(est_probs, dim=1)
            n_frames = est_masks.size(0)

            _loss = ce_loss(est_probs, masks).item()
            test_losses.update(_loss)
            _metrics = Metrics.get(est_probs, masks)
            test_metrics.update(_metrics)

            if test_writer is not None and idx < 3:
                for i in tqdm(range(0, n_frames, cfg.TEST.VISUALIZE_EVERY),
                              leave=False,
                              desc=video_name):
                    frame = frames[i]
                    est_mask = est_masks[i]
                    segmentation = utils.helpers.get_segmentation(frame, est_mask, {
                        'mean': cfg.CONST.DATASET_MEAN,
                        'std': cfg.CONST.DATASET_STD,
                    })
                    test_writer.add_image('%s/Frame%03d' % (video_name, i), segmentation,
                                          epoch_idx)

            logging.info('Test[%d/%d] VideoName = %s CE = %.4f Metrics = %s' %
                         (idx + 1, n_videos, video_name, _loss, ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
