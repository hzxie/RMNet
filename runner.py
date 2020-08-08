#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:00:36
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-07 09:14:25
# @Email:  cshzxie@gmail.com

import argparse
import importlib
import logging
import matplotlib
import os
import sys
# Fix no $DISPLAY environment variable
matplotlib.use('Agg')

from pprint import pprint

from core.train import train_net
from core.test import test_net
from core.inference import inference_net


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of R2Net runner')
    parser.add_argument('--exp', dest='exp_name', help='Experiment Name', default=None, type=str)
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='Path to the config.py file',
                        default='config.py',
                        type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to use', default=None, type=str)
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--inference',
                        dest='inference',
                        help='Inference for benchmark',
                        action='store_true')
    parser.add_argument('--weights',
                        dest='weights',
                        help='Initialize network from the weights file',
                        default=None)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    # Read the experimental config
    exec(compile(open(args.cfg_file, "rb").read(), args.cfg_file, 'exec'))
    cfg = locals()['__C']
    pprint(cfg)

    # Parse runtime arguments
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.exp_name is not None:
        cfg.CONST.EXP_NAME = args.exp_name
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights

    # Start train/test process
    if not args.test and not args.inference:
        train_net(cfg)
    else:
        if 'WEIGHTS' not in cfg.CONST or not os.path.exists(cfg.CONST.WEIGHTS):
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)

        if args.test:
            test_net(cfg)
        else:
            inference_net(cfg)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please use Python 3.x")

    # References: https://stackoverflow.com/a/53553516/1841143
    importlib.reload(logging)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    main()
