# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-07-09 10:17:51
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-13 20:01:08
# @Email:  cshzxie@gmail.com

import torch

import dist_matrix

class DistanceMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        dist_matrix.forward(matrix)
        return torch.sqrt(matrix)


class DistanceMatrix(torch.nn.Module):
    def __init__(self):
        super(DistanceMatrix, self).__init__()

    def forward(self, h, w):
        matrix = torch.zeros(h, w, h, w).float()

        if torch.cuda.is_available():
        	matrix = matrix.cuda()

        return DistanceMatrixFunction.apply(matrix)
