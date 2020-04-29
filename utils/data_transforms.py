# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 17:01:04
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-29 12:38:49
# @Email:  cshzxie@gmail.com

import math
import numbers
import numpy as np
import torch
import random
import sys

import torchvision.transforms

import utils.helpers

from PIL import Image


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
        frames = torch.from_numpy(np.array(frames)).float().permute(0, 3, 1, 2)
        masks = torch.from_numpy(np.array(masks))

        return frames, masks


class ReorganizeObjectID(object):
    def __init__(self, parameters):
        self.ignore_idx = parameters['ignore_idx']

    def __call__(self, frames, masks):
        mask_indexes = np.unique(masks[0])
        mask_indexes = mask_indexes[mask_indexes != self.ignore_idx]

        for m_idx, m in enumerate(masks):
            _m = np.zeros(m.shape)
            for idx, mi in enumerate(mask_indexes):
                _m[m == mi] = idx

            masks[m_idx] = _m.astype(np.uint8)

        return frames, masks


class ToOneHot(object):
    def __init__(self, parameters):
        self.shuffle = parameters['shuffle']
        self.n_objects = parameters['n_objects']

    def __call__(self, frames, masks):
        random_permutation = np.random.permutation(self.n_objects) + 1
        random_permutation = np.insert(random_permutation, 0, 0)    # Make background ID = 0
        masks = [utils.helpers.to_onehot(m, self.n_objects + 1) for m in masks]
        if self.shuffle:
            masks = [m[random_permutation, ...] for m in masks]

        return frames, masks


class Normalize(object):
    def __init__(self, parameters):
        self.mean = parameters['mean']
        self.std = parameters['std']

    def __call__(self, frames, masks):
        for idx, (f, m) in enumerate(zip(frames, masks)):
            f = f.astype(np.float32)
            frames[idx] = (f / 255. - self.mean) / self.std
            masks[idx] = m.astype(np.uint8)

        return frames, masks


class RandomPermuteRGB(object):
    def __init__(self, parameters):
        pass

    def __call__(self, frames, masks):
        random_permutation = np.random.permutation(3)
        for idx, f in enumerate(frames):
            frames[idx] = f[..., random_permutation]

        return frames, masks


class Resize(object):
    def __init__(self, parameters):
        self.size = parameters['size']
        self.keep_ratio = parameters['keep_ratio']

    def __call__(self, frames, masks):
        img_h, img_w = masks[0].shape

        height = img_h
        width = img_w
        if self.keep_ratio:
            scale = max(self.size / img_h, self.size / img_w)
            height = int(img_h * scale + 0.5)
            width = int(img_w * scale + 0.5)
        else:
            height = self.size
            width = self.size

        frames = [
            np.array(Image.fromarray(f).resize((width, height), Image.BILINEAR)) for f in frames
        ]
        masks = [
            np.array(Image.fromarray(m).resize((width, height), Image.NEAREST)) for m in masks
        ]
        return frames, masks


class RandomCrop(object):
    def __init__(self, parameters):
        self.height = parameters['height']
        self.width = parameters['width']
        self.ignore_idx = parameters['ignore_idx']

    def __call__(self, frames, masks):
        n_frames = len(frames)
        for i in range(n_frames):
            x_min = sys.maxsize
            y_min = sys.maxsize
            x_max = 0
            y_max = 0
            # Change ignore_idx to 0 for detecting bounding boxes
            mask = masks[i].copy()
            mask[mask == 255] = 0
            # Detect bounding boxes
            for j in np.unique(mask):
                # Ignore the background
                if j == 0:
                    continue

                _x_min, _x_max, _y_min, _y_max = utils.helpers.get_bounding_boxes(mask == j)
                # Bug Fix: the object is out of current frame
                if _x_min is None or _x_max is None or _y_min is None or _y_max is None:
                    continue

                x_min = min(x_min, _x_min)
                x_max = max(x_max, _x_max)
                y_min = min(y_min, _y_min)
                y_max = max(y_max, _y_max)

            # Crop the frame and mask with the bounding box
            bbox_height = y_max - y_min + 1
            bbox_width = x_max - x_min + 1

            # Determine the top left coordinates of the crop box
            img_h, img_w = masks[i].shape
            height_diff = abs(bbox_height - self.height)
            width_diff = abs(bbox_width - self.width)

            if bbox_height <= self.height:
                y_min_lb = max(y_min - height_diff, 0)
                y_min_ub = min(img_h - self.height, y_min)
                y_min = random.randint(y_min_lb, y_min_ub) if y_min_lb < y_min_ub else 0
            else:
                y_min = random.randint(y_min, y_min + height_diff)

            if bbox_width <= self.width:
                x_min_lb = max(x_min - width_diff, 0)
                x_min_ub = min(img_w - self.width, x_min)
                x_min = random.randint(x_min_lb, x_min_ub) if x_min_lb < x_min_ub else 0
            else:
                x_min = random.randint(x_min, x_min + width_diff)

            # Crop the frame and mask
            frames[i] = frames[i][y_min:y_min + self.height, x_min:x_min + self.width, :]
            masks[i] = masks[i][y_min:y_min + self.height, x_min:x_min + self.width]

        return frames, masks


class RandomAffine(object):
    def __init__(self, parameters):
        self.degrees = parameters['degrees']
        self.translate = parameters['translate']
        self.scale = parameters['scale']
        self.shears = parameters['shears']
        self.frame_fill_color = parameters['frame_fill_color']
        self.mask_fill_color = parameters['mask_fill_color']

    def __call__(self, frames, masks):
        img_h, img_w = masks[0].shape

        for idx, (f, m) in enumerate(zip(frames, masks)):
            degrees, translate, scale, shears = torchvision.transforms.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shears,
                img_size=(img_h, img_w))

            frames[idx] = np.array(
                self._affine(Image.fromarray(f),
                             degrees,
                             translate,
                             scale,
                             shears,
                             fillcolor=tuple(self.frame_fill_color)))
            masks[idx] = np.array(
                self._affine(Image.fromarray(m),
                             degrees,
                             translate,
                             scale,
                             shears,
                             fillcolor=self.mask_fill_color))

        return frames, masks

    def _affine(self, img, degree, translate, scale, shear, resample=0, fillcolor=None):
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        matrix = self._get_inverse_affine_matrix(center, degree, translate, scale, shear)
        kwargs = {"fillcolor": fillcolor}

        return img.transform(img.size, Image.AFFINE, matrix, resample, **kwargs)

    def _get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # Helper method to compute inverse matrix for affine transformation
        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, s, (sx, sy)) =
        #       = R(a) * S(s) * SHy(sy) * SHx(sx)
        #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
        #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
        #         [ 0                    , 0                                      , 1 ]
        #
        # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
        # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
        #          [0, 1      ]              [-tan(s), 1]
        #
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError("Shear should be a single value or a tuple/list containing " +
                             "two values. Got {}".format(shear))

        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0, -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M
