# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 17:01:04
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-10 14:49:59
# @Email:  cshzxie@gmail.com

import cv2
import math
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
            })  # yapf: disable

    def __call__(self, frames, masks):
        for tr in self.transformers:
            transform = tr['callback']
            frames, masks = transform(frames, masks)

        return frames, masks


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, frames, masks):
        return torch.from_numpy(np.array(frames)).float(), torch.from_numpy(np.array(masks))


class Normalize(object):
    def __init__(self, parameters):
        self.mean = parameters['mean']
        self.std = parameters['std']

    def __call__(self, frames, masks):

        for idx, f in enumerate(frames):
            f /= 255.
            f -= self.mean
            f /= self.std
            frames[idx] = f

        return frames, masks
