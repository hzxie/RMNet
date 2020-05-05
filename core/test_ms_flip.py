# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:30:11
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-30 17:41:38
# @Email:  cshzxie@gmail.com

import logging
import numpy as np
import torch

import utils.data_loaders
import utils.helpers

from tqdm import tqdm

from models.stm import STM
from models.lovasz_loss import LovaszLoss
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
import os

def test_ms_flip_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, stm=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True


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
    test_scales=cfg.TEST.TEST_SCALES
    test_fliplr=cfg.TEST.FLIPLR
    if(test_fliplr):
        test_fliplr=[False,True]
    else:
        test_fliplr=[False]

    nll_loss = torch.nn.NLLLoss(ignore_index=cfg.CONST.IGNORE_IDX)
    lovasz_loss = LovaszLoss(ignore_index=cfg.CONST.IGNORE_IDX)

    # The testing loop
    n_videos = len(test_data_loader)
    # n_videos=2
    all_est_probs=list()
    all_masks=list()
    all_objects=list()
    all_shapes=list()
    for id_s,test_scale in enumerate(test_scales):
        for id_f,fliplr in enumerate(test_fliplr):
            test_losses = AverageMeter()
            test_metrics = AverageMeter(Metrics.names())
            for idx, (video_name, n_objects, frames, masks, target_objects) in enumerate(test_data_loader):
                # Test only selected videos to accelerate the testing process
                if(idx>=n_videos):
                    break
                if not epoch_idx == -1 and idx not in cfg.TEST.TESTING_VIDEOS_INDEXES:
                    continue
                # print('Before frame resize')
                # print(frames.shape)
                B,N,C,H,W=frames.shape
                # frames=torch.nn.functional.interpolate(frames,size=[C,int(H*test_scale),int(W*test_scale)],mode='bilinear')

                inter_frames=torch.zeros([B,N,C,int(H*test_scale),int(W*test_scale)])
                for i in range(N):
                    inter_frames[:,i,:,:,:]=torch.nn.functional.interpolate(frames[:,i,:,:,:],size=[int(H*test_scale),int(W*test_scale)],mode='bilinear')
                frames=inter_frames

                original_masks = masks.clone()

                # print(masks.type())
                masks=masks.float()
                # print(masks.type())
                B,N,O,H,W=masks.shape
                inter_masks=torch.zeros([B,N,O,int(H*test_scale),int(W*test_scale)])
                for i in range(N):
                    inter_masks[:,i,:,:,:]=torch.nn.functional.interpolate(masks[:,i,:,:,:],size=[int(H*test_scale),int(W*test_scale)],mode='nearest')
                # masks=masks.int()
                masks=inter_masks.int()


                # print('After frame resize')
                # print(frames.shape)

                # No need to resize the mask in inference
                # masks = torch.nn.functional.interpolate(masks, size=[C,int(H*test_scale),int(W*test_scale)], mode='nearest')
                if(fliplr):
                    frames=torch.flip(frames,[4])
                    masks=torch.flip(masks,[4])
                with torch.no_grad():
                    # Fix Assertion Error:  all(map(lambda i: i.is_cuda, inputs))
                    if torch.cuda.device_count() > 1:
                        frames = utils.helpers.var_or_cuda(frames)
                        masks = utils.helpers.var_or_cuda(masks)
                        target_objects = utils.helpers.var_or_cuda(target_objects)

                    # Fix bugs: OOM error for large videos
                    try:
                        est_probs = stm(frames, masks, target_objects, n_objects, cfg.TEST.MEMORIZE_EVERY)
                    except Exception as ex:
                        logging.warn(ex)
                        print(ex)
                        continue

                    if(fliplr):
                        est_probs=torch.flip(est_probs,[4])

                    #Resize est_probs
                    # print('Before est resize')
                    # print(est_probs.shape)

                    # est_probs = torch.nn.functional.interpolate(est_probs,size=[C,H,W],mode='bilinear')

                    inter_probs = torch.zeros([B, N, O, H,W])
                    for i in range(N):
                        inter_probs[:, i, :, :, :] = torch.nn.functional.interpolate(est_probs[:, i, :, :, :],size=[H,W],mode='nearest')
                    est_probs=inter_probs

                    # print('After est resize')
                    # print(est_probs.shape)

                    est_probs = est_probs.permute(0, 2, 1, 3, 4)
                    all_est_probs.append(est_probs.cpu())

                    # output_folder=os.path.join(cfg.DIR.OUT_PATH, 'probilities', cfg.CONST.EXP_NAME)
                    # np.save('%s%s.npy'%(output_folder,video_name),est_probs)

                    masks=original_masks
                    masks = torch.argmax(masks, dim=2)
                    est_masks = torch.argmax(est_probs, dim=1)

                    if(id_s==0 and id_f==0):
                        all_masks.append(masks)
                        all_objects.append(n_objects[0].item())
                        all_shapes.append(masks.shape)

                    loss = nll_loss(torch.log(est_probs), masks).item() + lovasz_loss(est_probs,
                                                                                      masks).item()
                    test_losses.update(loss)
                    metrics = Metrics.get(est_masks[0], masks[0])
                    test_metrics.update(metrics, n_objects[0].item())

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

                # logging.info('Test[%d/%d]\tVideoName\t%s\tNumObject\t%d\tLoss\t%.4f Metrics\t%.4f\t%.4f\t%.4f' %
                #              (idx + 1, n_videos, video_name,n_objects,loss, metrics[0],metrics[1],metrics[2]))

            # Print testing results
            logging.info(f'Test scale:{test_scale} flip lr:{fliplr} ')
            logging.info('[Test Summary] Loss = %.4f Metrics = %s' %
                         (test_losses.avg(), ['%.4f' % tm for tm in test_metrics.avg()]))

            # Add testing results to TensorBoard
            if test_writer is not None:
                test_writer.add_scalar('Loss/Epoch', test_losses.avg(), epoch_idx)
                for i, metric in enumerate(test_metrics.items):
                    test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)


    test_metrics.reset()
    # n_videos=2
    for id_v in range(n_videos):
        # current_shape=all_shapes[id_v]
        # B, N, O, H, W=current_shape
        # sum_est_probs = torch.zeros([B, O, N, H, W])
        # print(sum_est_probs.shape)
        sum_est_probs=0
        for id_s,test_scale in enumerate(test_scales):
            for id_f,fliplr in enumerate(test_fliplr):
                sum_est_probs+=all_est_probs[id_s*n_videos*len(test_fliplr)+id_f*n_videos+id_v]
        sum_est_probs=sum_est_probs/(len(test_scales)*len(test_fliplr))
        sum_est_masks = torch.argmax(sum_est_probs, dim=1)
        masks=all_masks[id_v]
        metrics = Metrics.get(sum_est_masks[0], masks[0])
        test_metrics.update(metrics, all_objects[id_v])
        logging.info('Test[%d/%d]\t NumoBject\t%d\tMetrics\t%.4f\t%.4f\t%.4f' %
                     (id_v + 1, n_videos,all_objects[id_v], metrics[0], metrics[1], metrics[2]))
    logging.info('[MS Flip Test Summary] Metrics = %s' %
                 (['%.4f' % tm for tm in test_metrics.avg()]))
        # sum_est_probs=es
    # for():

    return Metrics(cfg.TEST.MAIN_METRIC_NAME, test_metrics.avg())
