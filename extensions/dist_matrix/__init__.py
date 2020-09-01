# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-17 09:06:16
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-09-01 15:31:54
# @Email:  cshzxie@gmail.com

import torch

import dist_matrix


class DistanceMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, prob_threshold):
        dist_mtx = dist_matrix.forward(mask, prob_threshold)
        return dist_mtx

    @staticmethod
    def backward(ctx, grad_dist_mtx):
        # TODO
        return torch.ones(grad_dist_mtx.size()).float().cuda(), None


class DistanceMatrix(torch.nn.Module):
    def __init__(self):
        super(DistanceMatrix, self).__init__()

    def forward(self, mask, prob_threshold=0.5, scale_factor=50):
        dist_mtx = DistanceMatrixFunction.apply(mask, prob_threshold)
        max_dist = torch.max(dist_mtx) + 1
        dist_mtx = dist_mtx / max_dist * scale_factor
        return torch.sigmoid(-dist_mtx) * 2
