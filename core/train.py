# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:30:03
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-13 11:11:25
# @Email:  cshzxie@gmail.com

import logging
import os
import torch

import utils.data_loaders
import utils.helpers

from time import time
from tensorboardX import SummaryWriter

from core.test import test_net
from models.stm import STM
from utils.average_meter import AverageMeter
from utils.metrics import Metrics


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
        cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.N_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.VAL),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.N_WORKERS,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', cfg.CONST.EXP_NAME)
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Set up networks
    stm = STM(cfg)
    stm.kv_memory.apply(utils.helpers.init_weights)
    stm.kv_query.apply(utils.helpers.init_weights)
    stm.decoder.apply(utils.helpers.init_weights)
    logging.info('Parameters in STM: %d.' % (utils.helpers.count_parameters(stm)))

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        stm = torch.nn.DataParallel(stm).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, stm.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=cfg.TRAIN.LR_MILESTONES,
                                                        gamma=cfg.TRAIN.GAMMA)

    # Set up loss functions
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=cfg.CONST.INGORE_IDX)

    # Load the pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = Metrics(cfg.TEST.MAIN_METRIC_NAME, checkpoint['best_metrics'])
        stm.load_state_dict(checkpoint['stm'])
        logging.info('Recover completed. Current epoch = #%d; best metrics = %s.' %
                     (init_epoch, best_metrics))

    # Training/Testing the network
    losses = AverageMeter()
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        stm.train()
        torch.autograd.set_detect_anomaly(True)

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (video_name, n_objects, frames, masks) in enumerate(train_data_loader):
            data_time.update(time() - batch_end_time)

            est_probs = stm(frames, masks, n_objects)
            loss = None
            for i in range(cfg.TRAIN.BATCH_SIZE):
                _loss = ce_loss(est_probs[i], torch.argmax(masks[i], dim=1))

                if loss is None:
                    loss = _loss
                else:
                    loss += _loss

            loss /= cfg.TRAIN.BATCH_SIZE
            losses.update(loss.item())
            stm.zero_grad()
            loss.backward()
            optimizer.step()

            n_itr = (epoch_idx - 1) * n_batches + batch_idx
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
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, losses.avg()))

        # Evaluate the current model
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, stm)

        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            file_name = 'ckpt-best.pth' if metrics.better_than(
                best_metrics) else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'stm': stm.state_dict()
            }, output_path)  # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            if metrics.better_than(best_metrics):
                best_metrics = metrics

    train_writer.close()
    val_writer.close()
