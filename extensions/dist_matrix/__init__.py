# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-17 09:06:16
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-23 09:18:16
# @Email:  cshzxie@gmail.com

import torch

import dist_matrix


class DistanceMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, prob_threshold):
        dist_mtx = dist_matrix.forward(mask, prob_threshold)
        return dist_mtx


class DistanceMatrix(torch.nn.Module):
    def __init__(self):
        super(DistanceMatrix, self).__init__()

    def forward(self, mask, prob_threshold=0.5, scale_factor=0.1):
        return torch.sqrt(DistanceMatrixFunction.apply(mask, prob_threshold)) * scale_factor
