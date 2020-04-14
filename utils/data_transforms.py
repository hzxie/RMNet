# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 17:01:04
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-14 14:55:37
# @Email:  cshzxie@gmail.com

import numpy as np
import torch
import random
import sys


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
            })  # yapf: disable

    def __call__(self, frames, masks, n_objects):
        for tr in self.transformers:
            transform = tr['callback']
            frames, masks = transform(frames, masks, n_objects)

        return frames, masks


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, frames, masks, n_objects):
        frames = torch.from_numpy(np.array(frames)).float().permute(0, 3, 1, 2)
        masks = torch.from_numpy(np.array(masks))

        return frames, masks


class Normalize(object):
    def __init__(self, parameters):
        self.mean = parameters['mean']
        self.std = parameters['std']

    def __call__(self, frames, masks, n_objects):
        for idx, f in enumerate(frames):
            f /= 255.
            f -= self.mean
            f /= self.std
            frames[idx] = f

        return frames, masks


class RandomCrop(object):
    def __init__(self, parameters):
        self.height = parameters['height']
        self.width = parameters['width']

    def __call__(self, frames, masks, n_objects):
        n_frames = len(frames)
        for i in range(n_frames):
            x_min = sys.maxsize
            y_min = sys.maxsize
            x_max = 0
            y_max = 0
            # Detect bounding boxes
            for j in range(1, n_objects + 1):
                _x_min, _x_max, _y_min, _y_max = self._get_bounding_boxes(masks[i] == j)
                # Bug Fix: the object is out of current frame
                if _x_min is None or _x_max is None or _y_min is None or _y_max is None:
                    continue

                x_min = min(x_min, _x_min)
                x_max = max(x_max, _x_max)
                y_min = min(y_min, _y_min)
                y_max = max(y_max, _y_max)

            # Crop the frame and mask with the bounding box
            bbox_height = y_max - y_min
            bbox_width = x_max - x_min

            # Determine the top left coordinates of the crop box
            img_h, img_w = masks[i].shape
            height_diff = abs(bbox_height - self.height)
            width_diff = abs(bbox_width - self.width)

            if bbox_height <= self.height:
                y_min = random.randint(max(y_min - height_diff, 0),
                                       min(img_h - self.height, y_min))
            else:
                y_min = random.randint(y_min, y_min + height_diff)

            if bbox_width <= self.width:
                x_min = random.randint(max(x_min - width_diff, 0), min(img_w - self.height, x_min))
            else:
                x_min = random.randint(x_min, x_min + width_diff)

            # Crop the frame and mask
            frames[i] = frames[i][y_min:y_min + self.height, x_min:x_min + self.width, :]
            masks[i] = masks[i][y_min:y_min + self.height, x_min:x_min + self.width]

        return frames, masks

    def _get_bounding_boxes(self, mask):
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if len(cols) == 0 or len(rows) == 0:
            return None, None, None, None

        x_min, x_max = cols[[0, -1]]
        y_min, y_max = rows[[0, -1]]

        return x_min, x_max, y_min, y_max
