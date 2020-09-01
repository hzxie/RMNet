# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:30:03
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-09-01 15:25:07
# @Email:  cshzxie@gmail.com

import logging
import os
import random
import torch
import uuid
import zipfile

import utils.data_loaders
import utils.helpers

from time import time

from core.test import test_net
from models.stm import STM
from models.lovasz_loss import LovaszLoss
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.summary_writer import SummaryWriter


def train_net(cfg):
    # Set up data loader
    train_data_loader = torch.utils.data.DataLoader(
        dataset=utils.data_loaders.DatasetCollector.get_dataset(
            cfg, cfg.DATASET.TRAIN_DATASET, utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.N_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=utils.data_loaders.DatasetCollector.get_dataset(
            cfg, cfg.DATASET.TEST_DATASET, utils.data_loaders.DatasetSubset.VAL),
        batch_size=1,
        num_workers=cfg.CONST.N_WORKERS,
        pin_memory=True,
        shuffle=False)

    # Set up networks
    stm = STM(cfg)
    stm.kv_memory.apply(utils.helpers.init_weights)
    stm.kv_query.apply(utils.helpers.init_weights)
    stm.decoder.apply(utils.helpers.init_weights)
    logging.info('Parameters in STM: %d.' % (utils.helpers.count_parameters(stm)))

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        if torch.__version__ >= '1.2.0' and cfg.TRAIN.USE_BATCH_NORM:
            torch.distributed.init_process_group('nccl',
                                                 init_method='file:///tmp/stm-%s' %
                                                 uuid.uuid4().hex,
                                                 world_size=1,
                                                 rank=0)
            stm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stm)

        stm = torch.nn.DataParallel(stm).cuda()

    # Create the optimizers
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, stm.parameters()),
                                  lr=cfg.TRAIN.LEARNING_RATE,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                  betas=cfg.TRAIN.BETAS)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=cfg.TRAIN.LR_MILESTONES,
                                                        gamma=cfg.TRAIN.GAMMA)

    # Set up loss functions
    nll_loss = torch.nn.NLLLoss(ignore_index=cfg.CONST.IGNORE_IDX)
    lovasz_loss = LovaszLoss(ignore_index=cfg.CONST.IGNORE_IDX)

    # Load the pretrained model if exists
    init_epoch = 0
    best_metrics = None
    METRICS_THRESHOLD = Metrics(
        cfg.TEST.MAIN_METRIC_NAME,
        [cfg.TRAIN.CKPT_SAVE_THRESHOLD for i in range(len(Metrics.names()))])

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = Metrics(cfg.TEST.MAIN_METRIC_NAME, checkpoint['best_metrics'])
        stm.load_state_dict(checkpoint['stm'])
        logging.info('Recover completed. Current epoch = #%d; best metrics = %s.' %
                     (init_epoch, best_metrics))

    # Set up folders for logs, snapshot and checkpoints
    output_dir = os.path.join(cfg.DIR.OUTPUT_DIR, '%s', cfg.CONST.EXP_NAME)
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(cfg, 'train')
    val_writer = SummaryWriter(cfg, 'test')

    # Backup current code snapshot
    cfg.DIR.SNAPSHOTS = os.path.join(cfg.DIR.OUTPUT_DIR, 'snapshots')
    if not os.path.exists(cfg.DIR.SNAPSHOTS):
        os.makedirs(cfg.DIR.SNAPSHOTS)

    with zipfile.ZipFile(os.path.join(cfg.DIR.SNAPSHOTS, '%s.zip' % cfg.CONST.EXP_NAME),
                         'w') as zf:
        root_dir = os.getcwd()
        for dirname, subdirs, files in os.walk(root_dir):
            if os.path.normpath(dirname).find(os.path.normpath(cfg.DIR.OUTPUT_DIR)) != -1:
                continue

            _dirname = os.path.relpath(dirname, root_dir)
            zf.write(_dirname)
            for filename in files:
                zf.write(os.path.join(_dirname, filename))

    # Training/Testing the network
    losses = AverageMeter()
    n_batches = len(train_data_loader)
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        if cfg.TRAIN.USE_BATCH_NORM:
            stm.train()
        else:
            stm.eval()

        # Update frame step
        if cfg.TRAIN.USE_RANDOM_FRAME_STEPS:
            if epoch_idx + cfg.TRAIN.LAST_N_EPOCHES_FIXING_FRAME_STEPS >= cfg.TRAIN.N_EPOCHS:
                # Keep the frame step == 1 for the last several epochs
                max_frame_steps = 1
            else:
                max_frame_steps = random.randint(
                    1, min(cfg.TRAIN.MAX_FRAME_STEPS, epoch_idx // 5 + 2))

            train_data_loader.dataset.set_frame_step(random.randint(1, max_frame_steps))
            logging.info('[Epoch %d/%d] Set frame step to %d' %
                         (epoch_idx, cfg.TRAIN.N_EPOCHS, train_data_loader.dataset.frame_step))

        batch_end_time = time()
        for batch_idx, (video_name, n_objects, frames, masks,
                        optical_flows) in enumerate(train_data_loader):
            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            data_time.update(time() - batch_end_time)

            try:
                frames = utils.helpers.var_or_cuda(frames)
                masks = utils.helpers.var_or_cuda(masks)
                optical_flows = utils.helpers.var_or_cuda(optical_flows)

                est_probs = stm(frames, masks, optical_flows, n_objects, cfg.TRAIN.MEMORIZE_EVERY)
                est_probs = utils.helpers.var_or_cuda(est_probs[:, 1:]).permute(0, 2, 1, 3, 4)
                masks = torch.argmax(masks[:, 1:], dim=2)
                loss = lovasz_loss(est_probs, masks) + nll_loss(torch.log(est_probs), masks)
                losses.update(loss.item())
                stm.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as ex:
                logging.warn(ex)
                continue

            train_writer.add_scalar('Loss/Batch', loss.item(), n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info(
                '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.4f' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(),
                 data_time.val(), losses.val()))

        lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch', losses.avg(), epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Loss = %.4f' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, losses.avg()))

        # Evaluate the current model
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, stm)

        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.CKPT_SAVE_FREQ == 0 and metrics.better_than(METRICS_THRESHOLD):
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-epoch-%03d.pth' % epoch_idx)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'stm': stm.state_dict()
            }, output_path)  # yapf: disable
            logging.info('Saved checkpoint to %s ...' % output_path)

        if metrics.better_than(best_metrics):
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-best.pth')
            best_metrics = metrics
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'stm': stm.state_dict()
            }, output_path)  # yapf: disable
            logging.info('Saved checkpoint to %s ...' % output_path)

    train_writer.close()
    val_writer.close()
