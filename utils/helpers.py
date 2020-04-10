# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:17:25
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-09 11:18:26
# @Email:  cshzxie@gmail.com

import torch


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x
