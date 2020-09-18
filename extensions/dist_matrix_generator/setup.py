# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-14 16:28:08
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-09-18 10:17:11
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='dist_matrix_generator',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('dist_matrix_generator',
                        ['dist_matrix_generator_cuda.cpp', 'dist_matrix_generator.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})
