# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-17 09:06:16
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-11-03 18:02:13
# @Email:  cshzxie@gmail.com

import torch

import reg_att_map_generator


class RegionalAttentionMapGeneratorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, prob_threshold, n_pts_threshold, n_bbox_loose_pixels):
        att_map, bbox = reg_att_map_generator.forward(mask, prob_threshold, n_pts_threshold,
                                                      n_bbox_loose_pixels)
        return att_map, bbox


class RegionalAttentionMapGenerator(torch.nn.Module):
    def __init__(self):
        super(RegionalAttentionMapGenerator, self).__init__()

    def forward(self, mask, prob_threshold=0.5, n_pts_threshold=10, n_bbox_loose_pixels=64):
        return RegionalAttentionMapGeneratorFunction.apply(mask, prob_threshold, n_pts_threshold,
                                                           n_bbox_loose_pixels)
