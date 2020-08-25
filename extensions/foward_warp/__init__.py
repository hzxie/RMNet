# -*- coding: utf-8 -*-
# @Author: Zhihao Li
# @Date:   2020-08-24 19:38:09
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-24 21:51:50
# @Email:  cshzxie@gmail.com

import torch

import forward_warp


class ForwardWarpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, im0, flow, interpolation_mode):
        '''
        im0: the first image with shape [B, C, H, W]
        flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        interpolation_mode: 0 is Bilinear, 1 is Nearest
        '''
        print(interpolation_mode)
        ctx.save_for_backward(im0, flow, interpolation_mode)
        return forward_warp.forward(im0, flow, interpolation_mode)

    @staticmethod
    def backward(ctx, grad_output):
        im0, flow, interpolation_mode = ctx.saved_variables
        return forward_warp.backward(grad_output, im0, flow, interpolation_mode)


class ForwardWarp(torch.nn.Module):
    def __init__(self):
        super(ForwardWarp, self).__init__()

    def forward(self, im0, flow, interpolation_mode='bilinear'):
        assert interpolation_mode == "bilinear" or "nearest"

        interpolation_mode = 0 if interpolation_mode == 'bilinear' else 1
        return ForwardWarpFunction.apply(im0, flow, interpolation_mode)
