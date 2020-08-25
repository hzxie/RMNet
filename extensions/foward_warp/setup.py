# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-24 19:38:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-24 19:39:31
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='forward_warp',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('forward_warp', ['forward_warp_cuda.cpp', 'forward_warp.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})
