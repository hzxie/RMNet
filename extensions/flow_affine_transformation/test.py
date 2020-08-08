# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-07 12:15:06
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-08 11:23:13
# @Email:  cshzxie@gmail.com

import numpy as np
import unittest

import flow_affine_transformation


class FlowAffineTransformationTestCase(unittest.TestCase):
    def test(self):
        optical_flow = np.random.rand(480, 640, 2).astype(np.float32)
        tr_matrix1 = np.random.rand(2, 3).astype(np.float32)
        tr_matrix2 = np.random.rand(2, 3).astype(np.float32)

        optical_flow = flow_affine_transformation.update_optical_flow(
            optical_flow, tr_matrix1, tr_matrix2)
        print(optical_flow)


if __name__ == '__main__':
    unittest.main()
