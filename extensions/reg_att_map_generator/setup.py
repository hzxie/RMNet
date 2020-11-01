# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-14 16:28:08
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-10-30 17:02:47
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='reg_att_map_generator',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('reg_att_map_generator',
                        ['reg_att_map_generator_cuda.cpp', 'reg_att_map_generator.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})
