# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-07 10:17:48
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-09 11:39:04
# @Email:  cshzxie@gmail.com

import numpy

from distutils.core import setup, Extension

# run the setup
setup(name='flow_affine_transformation',
      version='1.0.0',
      ext_modules=[
          Extension('flow_affine_transformation',
                    sources=['flow_affine_transformation.cpp'],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args=['-std=c++11'])
      ])
