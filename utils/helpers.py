# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:17:25
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-13 20:11:50
# @Email:  cshzxie@gmail.com

import numpy as np
import scipy.ndimage.morphology
import torch
import torch.nn.functional as F


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


def get_mask_probabilities(stm, frames, masks, n_objects, memorize_every):
    batch_size = len(frames)
    est_probs = []
    for i in range(batch_size):
        n_frames, k, h, w = masks[i].size()
        to_memorize = [j for j in range(0, n_frames, memorize_every)]

        _est_masks = torch.zeros(n_frames, k, h, w).float()
        _est_masks[0] = masks[i][0]

        keys = None
        values = None
        for t in range(1, n_frames):
            # Memorize
            prev_mask = var_or_cuda(_est_masks[t - 1])
            prev_key, prev_value = stm(frames[i][t - 1].unsqueeze(dim=0),
                                       prev_mask.unsqueeze(dim=0), n_objects[i])
            if t - 1 == 0:
                this_keys, this_values = prev_key, prev_value
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)

            if t - 1 in to_memorize:
                keys, values = this_keys, this_values

            # Segment
            logit = stm(frames[i][t].unsqueeze(dim=0), this_keys, this_values,
                        n_objects[i]).squeeze(dim=0)
            _est_masks[t] = F.softmax(logit, dim=0)

        est_probs.append(_est_masks)

    return est_probs


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
