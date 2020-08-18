# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-18 16:42:37
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-18 17:45:17
# @Email:  cshzxie@gmail.com

import torch.nn
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, top_k, ignore_index):
        super(FocalLoss, self).__init__()
        self.top_k = top_k
        self.ignore_index = ignore_index

    def forward(self, input, target, step_percent=0):
        B, K, N, H, W = input.shape
        step_percent = min(step_percent, 1.0)

        input = input.view(B * N, K, H * W)
        target = target.view(B * N, H * W)

        nll_loss = F.nll_loss(input, target, ignore_index=self.ignore_index, reduction='none')

        n_pixels = H * W
        n_top_k_pixels = self.top_k * n_pixels
        n_top_k_pixels = int(step_percent * n_top_k_pixels + (1 - step_percent) * n_pixels)
        top_k_loss, _ = torch.topk(nll_loss, k=n_top_k_pixels)

        return torch.mean(top_k_loss)
