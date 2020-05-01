# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:17:25
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-01 11:01:14
# @Email:  cshzxie@gmail.com

import numpy as np
import scipy.ndimage.morphology
import torch

from PIL import Image


def var_or_cuda(x):
    x = x.contiguous()
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

    if type(mask) == torch.Tensor:
        one_hot_masks = torch.from_numpy(one_hot_masks)

    return one_hot_masks


def get_bounding_boxes(mask):
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(cols) == 0 or len(rows) == 0:
        return None, None, None, None

    x_min, x_max = cols[[0, -1]]
    y_min, y_max = rows[[0, -1]]

    return x_min, x_max, y_min, y_max


def img_denormalize(image, mean, std):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * std + mean) * 255
    return image.astype(np.uint8)


def img_normalize(image, mean, std, order='HWC'):
    image = (image.astype(np.float32) / 255. - mean) / std
    return image.transpose((2, 0, 1)) if order == 'CHW' else image


def get_segmentation(frame, mask, normalization_params, ignore_idx=255, alpha=0.4):
    PALETTE = np.array([[i, i, i] for i in range(256)])
    PALETTE[:16] = np.array([
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
    ])

    mask = mask.cpu().numpy()
    if frame is None:
        mask = Image.fromarray(mask.astype(np.uint8))
        mask.putpalette(PALETTE.reshape(-1).tolist())
        return mask

    frame = img_denormalize(frame, normalization_params['mean'], normalization_params['std'])
    objects = np.unique(mask)
    for o_id in objects[1:]:
        if o_id == ignore_idx:
            continue

        foreground = frame * alpha + np.ones(frame.shape) * (1 - alpha) * np.array(PALETTE[o_id])
        binary_mask = mask == o_id

        frame[binary_mask] = foreground[binary_mask]
        countours = scipy.ndimage.morphology.binary_dilation(binary_mask) ^ binary_mask
        frame[countours, :] = 0

    return Image.fromarray(frame)
