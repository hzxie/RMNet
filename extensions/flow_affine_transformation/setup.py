# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-07 10:17:48
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-07 19:01:03
# @Email:  cshzxie@gmail.com
#
# References:
# - https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html

import numpy

from distutils.core import setup, Extension

# run the setup
setup(name='flow_affine_transformatio',
      version='1.0.0',
      ext_modules=[
          Extension('flow_affine_transformation',
                    sources=['flow_affine_transformation.cpp'],
                    include_dirs=[numpy.get_include()])
      ])
