# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:30:11
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-11-05 15:03:16
# @Email:  cshzxie@gmail.com

import logging
import numpy as np
import torch

import utils.data_loaders
import utils.helpers

from tqdm import tqdm

from models.rmnet import RMNet
from models.tiny_flownet import TinyFlowNet
from models.lovasz_loss import LovaszLoss
from utils.average_meter import AverageMeter
from utils.metrics import Metrics


def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             tflownet=None,
             rmnet=None):
    # Set up data loader
    if test_data_loader is None:
        # Set up data loader
        test_data_loader = torch.utils.data.DataLoader(
            dataset=utils.data_loaders.DatasetCollector.get_dataset(
                cfg, cfg.DATASET.TEST_DATASET, utils.data_loaders.DatasetSubset.VAL),
            batch_size=1,
            num_workers=cfg.CONST.N_WORKERS,
            pin_memory=True,
            shuffle=False)

    # Setup networks and initialize networks
    if rmnet is None:
        tflownet = TinyFlowNet(cfg)
        rmnet = RMNet(cfg)

        if torch.cuda.is_available():
            tflownet = torch.nn.DataParallel(tflownet).cuda()
            rmnet = torch.nn.DataParallel(rmnet).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        tflownet.load_state_dict(checkpoint['tflownet'])
        rmnet.load_state_dict(checkpoint['rmnet'])

    # Switch models to evaluation mode
    tflownet.eval()
    rmnet.eval()

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()
    nll_loss = torch.nn.NLLLoss(ignore_index=cfg.CONST.IGNORE_IDX)
    lovasz_loss = LovaszLoss(ignore_index=cfg.CONST.IGNORE_IDX)

    # The testing loop
    n_videos = len(test_data_loader)
    test_losses = AverageMeter()
    test_metrics = AverageMeter(Metrics.names())

    for idx, (video_name, n_objects, frames, masks, optical_flows) in enumerate(test_data_loader):
        # Test only selected videos to accelerate the testing process
        if not epoch_idx == -1 and idx not in cfg.TEST.TESTING_VIDEOS_INDEXES:
            continue

        with torch.no_grad():
            # Fix Assertion Error:  all(map(lambda i: i.is_cuda, inputs))
            if torch.cuda.device_count() > 1:
                frames = utils.helpers.var_or_cuda(frames)
                masks = utils.helpers.var_or_cuda(masks)
                optical_flows = utils.helpers.var_or_cuda(optical_flows)

            # Fix bugs: OOM error for large videos
            try:
                if epoch_idx == -1:
                    est_flows, est_probs = utils.helpers.multi_scale_inference(cfg, tflownet, rmnet, frames,
                                                                    masks, n_objects)
                else:
                    est_flows = tflownet(frames)
                    est_probs = rmnet(frames, masks, est_flows, n_objects, cfg.TEST.MEMORIZE_EVERY)

                est_probs = est_probs.permute(0, 2, 1, 3, 4)
                masks = torch.argmax(masks, dim=2)
                est_masks = torch.argmax(est_probs, dim=1)

                if cfg.TRAIN.NETWORK == 'TinyFlowNet':
                    loss = l1_loss(est_flows, optical_flows)
                else:    # RMNet
                    loss = lovasz_loss(est_probs, masks) + nll_loss(torch.log(est_probs), masks)

            except Exception as ex:
                logging.exception(ex)
                continue

            test_losses.update(loss.item())
            metrics = Metrics.get(est_masks[0], masks[0])
            test_metrics.update(metrics, torch.max(n_objects[0]).item())

            video_name = video_name[0]
            if test_writer is not None and idx < 3 and cfg.TEST.VISUALIZE_EVERY > 0:
                frames = frames[0]
                n_frames = est_masks.size(1)

                for i in tqdm(range(0, n_frames, cfg.TEST.VISUALIZE_EVERY),
                              leave=False,
                              desc=video_name):
                    est_segmentation = utils.helpers.get_segmentation(
                        frames[i], est_masks[0][i], {
                            'mean': cfg.CONST.DATASET_MEAN,
                            'std': cfg.CONST.DATASET_STD,
                        }, cfg.CONST.IGNORE_IDX)
                    gt_segmentation = utils.helpers.get_segmentation(frames[i], masks[0][i], {
                        'mean': cfg.CONST.DATASET_MEAN,
                        'std': cfg.CONST.DATASET_STD,
                    }, cfg.CONST.IGNORE_IDX)
                    test_writer.add_image(
                        '%s/Frame%03d' % (video_name, i),
                        np.concatenate((est_segmentation, gt_segmentation), axis=0), epoch_idx)

            logging.info('Test[%d/%d] VideoName = %s Loss = %.4f Metrics = %s' %
                         (idx + 1, n_videos, video_name, loss, ['%.4f' % m for m in metrics]))

    # Print testing results
    logging.info('[Test Summary] Loss = %.4f Metrics = %s' %
                 (test_losses.avg(), ['%.4f' % tm for tm in test_metrics.avg()]))

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch', test_losses.avg(), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return Metrics(cfg.TEST.MAIN_METRIC_NAME, test_metrics.avg())
