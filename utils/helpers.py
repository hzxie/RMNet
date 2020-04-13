# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:17:25
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-12 19:59:12
# @Email:  cshzxie@gmail.com

import numpy as np
import scipy.ndimage.morphology
import torch


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(network):
    return sum(p.numel() for p in network.parameters())


def to_onehot(mask, k):
    h, w = mask.shape
    one_hot_masks = np.zeros((k, h, w), dtype=np.uint8)
    for k_idx in range(k):
        one_hot_masks[k_idx] = (mask == k_idx)

    return one_hot_masks


def get_segmentation(frame, mask, normalization_parameters):
    mask = mask.cpu().numpy()
    if frame is None:
        return mask

    mean = normalization_parameters['mean']
    std = normalization_parameters['std']
    frame = frame.permute(1, 2, 0).cpu().numpy()
    frame = (frame * std + mean) * 255
    frame = frame.astype(np.uint8)

    ALPHA = 0.4
    PALETTE = np.reshape([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [191, 0, 0],
        [64, 128, 0],
        [191, 128, 0],
        [64, 0, 128],
        [191, 0, 128],
        [64, 128, 128],
        [191, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 191, 0],
        [128, 191, 0],
        [0, 64, 128],
        [128, 64, 128],
    ], (-1, 3))

    objects = np.unique(mask)
    for o_id in objects[1:]:
        foreground = frame * ALPHA + np.ones(frame.shape) * (1 - ALPHA) * np.array(PALETTE[o_id])
        binary_mask = mask == o_id

        frame[binary_mask] = foreground[binary_mask]
        countours = scipy.ndimage.morphology.binary_dilation(binary_mask) ^ binary_mask
        frame[countours, :] = 0

    return frame
