# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-17 09:06:16
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-09-18 15:04:13
# @Email:  cshzxie@gmail.com

import torch

import utils.helpers

import dist_matrix_generator


class DistanceMatrixGeneratorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, occ_mask, prob_threshold, occ_dist_factor):
        dist_mtx = dist_matrix_generator.forward(mask, occ_mask, prob_threshold, occ_dist_factor)
        return dist_mtx

    @staticmethod
    def backward(ctx, grad_dist_mtx):
        grad_mask = utils.helpers.var_or_cuda(torch.ones(grad_dist_mtx.size()).float())
        grad_occ_mask = utils.helpers.var_or_cuda(torch.ones(grad_dist_mtx.size()).float())

        return grad_mask, grad_occ_mask, None, None


class DistanceMatrixGenerator(torch.nn.Module):
    def __init__(self):
        super(DistanceMatrixGenerator, self).__init__()

    def forward(self, mask, occ_mask, prob_threshold=0.5, occ_dist_factor=0.1, scale_factor=50):
        dist_mtx = DistanceMatrixGeneratorFunction.apply(mask, occ_mask, prob_threshold,
                                                         occ_dist_factor)
        max_dist = torch.max(dist_mtx) + 1
        dist_mtx = dist_mtx / max_dist * scale_factor
        return torch.sigmoid(-dist_mtx) * 2
