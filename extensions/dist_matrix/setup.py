# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-07-09 11:44:29
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-09 11:44:52
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='dist_matrix',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('dist_matrix', [
              'dist_matrix_cuda.cpp',
              'dist_matrix.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})

