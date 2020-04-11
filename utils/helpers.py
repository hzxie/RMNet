# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:17:25
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-11 16:01:11
# @Email:  cshzxie@gmail.com

import numpy as np
import scipy.ndimage.morphology
import torch


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def get_segmentation(frame, mask, normalization_parameters):
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
