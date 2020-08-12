#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-08 17:16:07
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-12 17:01:05
# @Email:  cshzxie@gmail.com

import argparse
import logging
import os
import requests
import shutil
import sys
import time
import torch
import threading

from bs4 import BeautifulSoup
from collections import OrderedDict

# Add STM project to sys path
PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(PROJECT_HOME)

import utils.data_loaders

from models.stm import STM
from utils.average_meter import AverageMeter
from utils.metrics import Metrics


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of eval_server')
    parser.add_argument('ckpt_dir', help='The path to folder that saves checkpoints')
    parser.add_argument('--cfg', help='The path to config file', default='config.py', type=str)
    parser.add_argument('--gpu',
                        help='The GPU IDs to use (None for auto detection)',
                        default=None,
                        type=str)
    parser.add_argument('--pavi',
                        help='The project name in PAVI (None for disabling PAVI)',
                        default=None,
                        type=str)
    parser.add_argument(
        '--remote',
        help='The remote folder that saves checkpoints (e.g., http://10.1.75.35:8000/)',
        default=None,
        type=str)
    args = parser.parse_args()
    return args


def get_summary_writer(log_dir, pavi_project_name):
    if pavi_project_name is None:
        import tensorboardX
        return tensorboardX.SummaryWriter(log_dir)
    else:
        import pavi
        _ = log_dir.split('/')
        return pavi.SummaryWriter(task=_[-2], labels=_[-1], project=pavi_project_name)


def add_scalars(summary_writer, scalars):
    ckpt_pending_removed = []
    for ckpt_name, jf_mean in scalars.items():
        if jf_mean == -1:
            break

        epoch_idx = int(ckpt_name[len('ckpt-epoch-'):-len('.pth')])
        summary_writer.add_scalar('Metric/JF-Mean', jf_mean, epoch_idx)
        ckpt_pending_removed.append(ckpt_name)

    for ckr in ckpt_pending_removed:
        del scalars[ckr]

    return scalars


def get_data_loader(cfg):
    return torch.utils.data.DataLoader(dataset=utils.data_loaders.DatasetCollector.get_dataset(
        cfg, cfg.DATASET.TEST_DATASET, utils.data_loaders.DatasetSubset.VAL),
                                       batch_size=1,
                                       num_workers=cfg.CONST.N_WORKERS,
                                       pin_memory=True,
                                       shuffle=False)


def get_networks(cfg):
    networks = []
    for i in range(torch.cuda.device_count()):
        networks.append({
            'network': STM(cfg).cuda(i),
            'data_loader': get_data_loader(cfg),
            'device': i
        })

    return networks


def test_network(cfg, network, data_loader, checkpoint, result_set):
    _checkpoint = torch.load(checkpoint)
    _checkpoint = {k.replace('module.', ''): v for k, v in _checkpoint['stm'].items()}
    network.load_state_dict(_checkpoint)
    network.eval()

    test_metrics = AverageMeter(Metrics.names())
    device, = list(set(p.device for p in network.parameters()))
    for idx, (video_name, n_objects, frames, masks, optical_flows) in enumerate(data_loader):
        with torch.no_grad():
            try:
                est_probs = network(frames, masks, optical_flows, n_objects,
                                    cfg.TEST.MEMORIZE_EVERY, device)
                est_probs = est_probs.permute(0, 2, 1, 3, 4)
                masks = torch.argmax(masks, dim=2)
                est_masks = torch.argmax(est_probs, dim=1)
            except Exception as ex:
                logging.warn('Error occurred during testing Checkpoint[Name=%s]: %s' %
                             (checkpoint, ex))
                continue

            metrics = Metrics.get(est_masks[0], masks[0])
            test_metrics.update(metrics, n_objects[0].item())

    jf_mean = test_metrics.avg(2)
    if jf_mean != 0:
        logging.info('Checkpoint[Name=%s] has been tested successfully, JF-Mean = %.4f.' %
                     (checkpoint, jf_mean))
    else:
        logging.warning('Exception occurred during testing Checkpoint[Name=%s]' % checkpoint)

    result_set['JF-Mean'] = jf_mean


def get_checkpoints(checkpoints_dir, remote_checkpoints_url, tested_checkpoints):
    # Sync checkpoints from remote server
    if remote_checkpoints_url is not None:
        checkpoints = BeautifulSoup(requests.get(remote_checkpoints_url).text, features='lxml')
        checkpoints = checkpoints.findAll('a')
        for ckpt in checkpoints:
            if not ckpt.text.startswith('ckpt-epoch') or ckpt.text in tested_checkpoints:
                continue

            ckpt_url = '%s/%s' % (remote_checkpoints_url, ckpt.text)
            logging.info('Downloading new checkpoint from %s' % ckpt_url)
            with requests.get(ckpt_url, stream=True) as r:
                with open(os.path.join(checkpoints_dir, ckpt.text), 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

    return sorted([f for f in os.listdir(checkpoints_dir) if f.startswith('ckpt-epoch')])


def get_next_checkpoints(tested_checkpoints, now_checkpoints):
    for nc in now_checkpoints:
        if nc not in tested_checkpoints:
            return nc

    return None


def main():
    # Get args from command line
    args = get_args_from_command_line()
    if args.remote is not None:
        try:
            response = requests.get(args.remote)
            if response.status_code != 200:
                raise Exception('Unexpected status code: %d' % response.status_code)
        except Exception as ex:
            logging.error(ex)
            sys.exit(127)

        ckpt_name = args.remote[args.remote.rfind('/') + 1:]
        args.ckpt_dir = args.ckpt_dir if args.ckpt_dir.find(ckpt_name) != -1 else os.path.join(
            args.ckpt_dir, ckpt_name)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    if not os.path.exists(args.ckpt_dir):
        logging.error('The checkpoints folder[Path=%s] does not exist.' % args.ckpt_dir)
        sys.exit(2)
    else:
        args.ckpt_dir = os.path.abspath(args.ckpt_dir)
        args.cfg = os.path.abspath(args.cfg)
        logging.info('Evaluation server started.')

    # Set up new summary writer
    log_dir = os.path.join(args.ckpt_dir.replace('checkpoints', 'logs'), 'val')
    test_writer = get_summary_writer(log_dir, args.pavi)
    logging.info('Tensorborad Logs available at: %s' % log_dir)

    # Read config.py
    exec(compile(open(args.cfg, "rb").read(), args.cfg, 'exec'))
    cfg = locals()['__C']
    cfg.DATASETS.DAVIS.INDEXING_FILE_PATH = os.path.abspath(
        os.path.join(PROJECT_HOME, cfg.DATASETS.DAVIS.INDEXING_FILE_PATH))

    # Initialize networks on assigned GPUs
    free_networks = get_networks(cfg)

    # Detect new checkpoints
    best_jf_mean = 0
    best_checkpoint = None
    running_threads = []
    tested_checkpoints = []
    test_results_buffer = OrderedDict()
    logging.info("Listening new checkpoints at %s ..." % args.ckpt_dir)

    while True:
        # Waiting for free GPUs & Processing test results
        while len(free_networks) == 0:
            time.sleep(15)    # Detect free gpus every 15 seconds
            for rt in running_threads:
                if not rt['thread'].is_alive():
                    running_threads.remove(rt)
                    free_networks.append(rt['network'])
                    jf_mean = rt['result_set']['JF-Mean']
                    # Add the results to the TensorBoard
                    test_results_buffer[rt["checkpoint"]] = jf_mean
                    test_results_buffer = add_scalars(test_writer, test_results_buffer)
                    # Update the best results
                    if jf_mean > best_jf_mean:
                        if best_checkpoint is not None:
                            os.remove(os.path.join(args.ckpt_dir, best_checkpoint))

                        best_jf_mean = jf_mean
                        best_checkpoint = rt['checkpoint']
                    elif jf_mean != 0:
                        os.remove(os.path.join(args.ckpt_dir, rt['checkpoint']))

        # Waiting for new checkpoints
        checkpoints = get_checkpoints(args.ckpt_dir, args.remote, tested_checkpoints)
        checkpoint = get_next_checkpoints(tested_checkpoints, checkpoints)
        if checkpoint is None:
            time.sleep(30)    # Detect new checkpoints every 30 seconds
            continue

        assigned_network = free_networks.pop(0)
        logging.info('Assign GPU[ID=%s] for the checkpoint[Name=%s].' %
                     (assigned_network['device'], checkpoint))
        result_set = {'JF-Mean': 0}
        worker_thread = threading.Thread(target=test_network,
                                         args=(cfg,
                                               assigned_network['network'],
                                               assigned_network['data_loader'],
                                               os.path.join(args.ckpt_dir, checkpoint),
                                               result_set))  # yapf: disable
        worker_thread.start()

        # Add the process to the checklist
        running_threads.append({
            'thread': worker_thread,
            'network': assigned_network,
            'checkpoint': checkpoint,
            'result_set': result_set
        })

        # Remember the tested checkpoint
        tested_checkpoints.append(checkpoint)
        test_results_buffer[checkpoint] = -1


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception('Please use Python 3.x')

    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    main()
