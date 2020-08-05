# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-26 15:03:35
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-05 11:45:55
# @Email:  cshzxie@gmail.com
#
# Maintainers:
# - Maxim Berman <maxim.berman@kuleuven.be>
# - Haozhe Xie <cshzxie@gmail.com>

import math
import torch

from itertools import filterfalse


class LovaszLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super(LovaszLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        Multi-class Lovasz-Softmax loss
          input: [B, C, F, H, W] Tensor, class probabilities at each prediction (between 0 and 1).
          target: [B, F, H, W] Tensor, ground truth labels (between 0 and C - 1)
        """
        input, target = self._flatten(input, target)

        if input.numel() == 0:
            # only void pixels, the gradients should be 0
            return input * 0.

        C = input.size(1)
        losses = []
        class_to_sum = list(range(C))
        for c in class_to_sum:
            fg = (target == c).float()    # foreground for class c
            if fg.sum() == 0:
                continue

            class_pred = input[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))

        return self._mean(losses)

    def _flatten(self, input, target):
        B, C, F, H, W = input.size()
        input = input.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)    # B * F * H * W, C = P, C
        target = target.view(-1)

        valid = target.ne(self.ignore_index).nonzero(as_tuple=False).squeeze()
        vinput = input[valid]
        vtarget = target[valid]

        return vinput, vtarget

    def _mean(self, lst, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        lst = iter(lst)
        if ignore_nan:
            lst = filterfalse(math.isnan, lst)

        try:
            n = 1
            acc = next(lst)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')

            return empty

        for n, v in enumerate(lst, 2):
            acc += v

        if n == 1:
            return acc

        return acc / n

    def _lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union

        if p > 1:    # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard
