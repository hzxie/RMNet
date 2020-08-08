# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 17:01:04
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-08 10:51:00
# @Email:  cshzxie@gmail.com

import cv2
import math
import matplotlib.pyplot as plt
import numbers
import numpy as np
import torch
import torchvision.transforms
import random
import sys

import utils.helpers
import flow_affine_transformation

from PIL import Image


def wrap(img, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    img = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(dim=0).float()
    flo = torch.from_numpy(-flo.copy()).permute(2, 0, 1).unsqueeze(dim=0).float()

    B, C, H, W = img.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    # grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(img, vgrid)
    mask = torch.ones(img.size())
    mask = torch.nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    output = output * mask
    return output.squeeze(dim=0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    UNKNOWN_FLOW_THRESH = 1e7

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
            })  # yapf: disable

    def __call__(self, frames, masks, optical_flows):
        for tr in self.transformers:
            transform = tr['callback']

            plt.figure(figsize=(20, 10))
            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(frames[0])
            ax1.grid(b=True, which="major", color="#ff0000", linestyle="-")

            ax2 = plt.subplot(2, 3, 2)
            ax2.imshow(frames[1])
            ax2.grid(b=True, which="major", color="#ff0000", linestyle="-")

            ax3 = plt.subplot(2, 3, 3)
            ax3.imshow(frames[2])
            ax3.grid(b=True, which="major", color="#ff0000", linestyle="-")

            ax4 = plt.subplot(2, 3, 4)
            ax4.imshow(flow_to_image(optical_flows[0]))

            ax5 = plt.subplot(2, 3, 5)
            ax5.imshow(wrap(frames[0], optical_flows[0]))
            ax5.grid(b=True, which="major", color="#ff0000", linestyle="-")

            ax6 = plt.subplot(2, 3, 6)
            ax6.imshow(wrap(frames[1], optical_flows[1]))
            ax6.grid(b=True, which="major", color="#ff0000", linestyle="-")

            plt.show()
            print(transform)

            frames, masks, optical_flows = transform(frames, masks, optical_flows)

        return frames, masks, optical_flows


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, frames, masks, optical_flows):
        frames = torch.from_numpy(np.array(frames)).float().permute(0, 3, 1, 2)
        masks = torch.from_numpy(np.array(masks))
        optical_flows = torch.from_numpy(np.array(optical_flows)).permute(0, 3, 1, 2)

        return frames, masks, optical_flows


class ReorganizeObjectID(object):
    def __init__(self, parameters):
        self.ignore_idx = parameters['ignore_idx']

    def __call__(self, frames, masks, optical_flows):
        mask_indexes = np.unique(masks[0])
        mask_indexes = mask_indexes[mask_indexes != self.ignore_idx]

        for m_idx, m in enumerate(masks):
            _m = np.zeros(m.shape)
            for idx, mi in enumerate(mask_indexes):
                _m[m == mi] = idx

            masks[m_idx] = _m.astype(np.uint8)

        return frames, masks, optical_flows


class ToOneHot(object):
    def __init__(self, parameters):
        self.shuffle = parameters['shuffle']
        self.n_objects = parameters['n_objects']

    def __call__(self, frames, masks, optical_flows):
        random_permutation = np.random.permutation(self.n_objects) + 1
        random_permutation = np.insert(random_permutation, 0, 0)    # Make background ID = 0
        masks = [utils.helpers.to_onehot(m, self.n_objects + 1) for m in masks]
        if self.shuffle:
            masks = [m[random_permutation, ...] for m in masks]

        return frames, masks, optical_flows


class Normalize(object):
    def __init__(self, parameters):
        self.mean = parameters['mean']
        self.std = parameters['std']

    def __call__(self, frames, masks, optical_flows):
        for idx, (f, m) in enumerate(zip(frames, masks)):
            frames[idx] = utils.helpers.img_normalize(f, self.mean, self.std).astype(np.float32)
            masks[idx] = m.astype(np.uint8)

        return frames, masks, optical_flows


class RandomPermuteRGB(object):
    def __init__(self, parameters):
        pass

    def __call__(self, frames, masks, optical_flows):
        random_permutation = np.random.permutation(3)
        for idx, f in enumerate(frames):
            frames[idx] = f[..., random_permutation]

        return frames, masks, optical_flows


class RandomFlip(object):
    def __init__(self, parameters):
        pass

    def __call__(self, frames, masks, optical_flows):
        rnd_value = random.random()

        if rnd_value <= 0.5:
            for idx, (f, m, of) in enumerate(zip(frames, masks, optical_flows)):
                frames[idx] = np.flip(f, axis=1)
                masks[idx] = np.flip(m, axis=1)
                optical_flows[idx] = np.flip(of, axis=1)
                optical_flows[idx][..., 0] = -optical_flows[idx][..., 0]

        return frames, masks, optical_flows


class Resize(object):
    def __init__(self, parameters):
        self.size = parameters['size']
        self.keep_ratio = parameters['keep_ratio']

    def __call__(self, frames, masks, optical_flows):
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
            cv2.resize(f, dsize=(width, height), interpolation=cv2.INTER_LINEAR) for f in frames
        ]
        masks = [
            cv2.resize(m, dsize=(width, height), interpolation=cv2.INTER_NEAREST) for m in masks
        ]
        optical_flows = [
            cv2.resize(of, dsize=(width, height), interpolation=cv2.INTER_LINEAR) * scale
            for of in optical_flows
        ]
        return frames, masks, optical_flows


class RandomCrop(object):
    def __init__(self, parameters):
        self.height = parameters['height']
        self.width = parameters['width']
        self.ignore_idx = parameters['ignore_idx']

    def __call__(self, frames, masks, optical_flows):
        prev_x_min = 0
        prev_y_min = 0

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
            optical_flows[i] = optical_flows[i][y_min:y_min + self.height,
                                                x_min:x_min + self.width, :]
            # Update the values of optical flow
            if i > 0:
                optical_flows[i - 1][..., 0] += prev_x_min - x_min
                optical_flows[i - 1][..., 1] += prev_y_min - y_min

            prev_x_min = x_min
            prev_y_min = y_min

        return frames, masks, optical_flows


class ColorJitter(object):
    def __init__(self, parameters):
        self.brightness = parameters['brightness']
        self.contrast = parameters['contrast']
        self.saturation = parameters['saturation']
        self.hue = parameters['hue']

    def __call__(self, frames, masks, optical_flows):
        jitter = torchvision.transforms.ColorJitter.get_params(brightness=self.brightness,
                                                               contrast=self.contrast,
                                                               saturation=self.saturation,
                                                               hue=self.hue)
        for idx, f in enumerate(frames):
            frames[idx] = np.array(jitter(Image.fromarray(f)))

        return frames, masks, optical_flows


class RandomAffine(object):
    def __init__(self, parameters):
        self.degrees = parameters['degrees']
        self.translate = parameters['translate']
        self.scale = parameters['scale']
        self.shears = parameters['shears']
        self.frame_fill_color = parameters['frame_fill_color']
        self.mask_fill_color = parameters['mask_fill_color']
        self.optical_flow_fill_color = parameters['optical_flow_fill_color']

    def __call__(self, frames, masks, optical_flows):
        img_h, img_w = masks[0].shape
        center = (img_h * 0.5 + 0.5, img_w * 0.5 + 0.5)

        tr_matices = []
        for idx, (f, m) in enumerate(zip(frames, masks)):
            degrees, translate, scale, shears = torchvision.transforms.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shears,
                img_size=(img_h, img_w))

            tr_matrix = self._get_inverse_affine_matrix(center, degrees, translate, scale, shears)
            tr_matices.append(tr_matrix)

            frames[idx] = self._affine(f, tr_matrix, fillcolor=tuple(self.frame_fill_color))
            masks[idx] = self._affine(m, tr_matrix, fillcolor=self.mask_fill_color)

        for idx, of in enumerate(optical_flows):
            # Skip the last frame
            if idx == len(optical_flows) - 1:
                continue

            # Update the optical flow values
            flow_affine_transformation.update_optical_flow(of, tr_matices[idx], tr_matices[idx + 1])
            optical_flows[idx] = self._affine(of,
                                              tr_matices[idx],
                                              fillcolor=tuple(self.optical_flow_fill_color))

        return frames, masks, optical_flows

    def _affine(self, img, matrix, fillcolor=None):
        return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), borderValue=fillcolor)

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
        #
        # See also: https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/
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

        return np.array(M).astype(np.float32).reshape(2, 3)
