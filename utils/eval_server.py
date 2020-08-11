#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-08 17:16:07
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-11 17:08:28
# @Email:  cshzxie@gmail.com

import argparse
import bs4
import gpustat
import logging
import os
import requests
import time
import torch
import shutil
import subprocess
import sys

from bs4 import BeautifulSoup
from collections import OrderedDict


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of eval_server')
    parser.add_argument('ckpt_dir', help='The path to folder that saves checkpoints')
    parser.add_argument('--cfg', help='The path to config file', default='config.py', type=str)
    parser.add_argument('--gpu', help='The GPU IDs to use', default=None, type=str)
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


def get_assigned_gpus(gpu_id):
    if gpu_id is not None:
        return [int(gpu) for gpu in gpu_id.split(',')]

    free_gpus = [gpu.index for gpu in gpustat.new_query() if gpu.memory_used == 0]
    # The below statement will cost 10MB/11MB on the allocated GPUs
    torch.cuda.is_available()

    return [
        gpu.index for gpu in gpustat.new_query() if gpu.memory_used > 0 and gpu.index in free_gpus
    ]


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


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1

    return start


def get_jf_mean(process, checkpoint):
    try:
        logs = process.stdout.read().decode('utf-8').split('\n')
        for log in logs:
            if log.find('[WARNING]') != -1:
                logging.warning(log[find_nth(log, ' ', 3) + 1:])
            elif log.find('[Test Summary]') != -1:
                last_comma = log.rfind(',')
                last_quot_mark = log.rfind('\'')
                return float(log[last_comma + 3:last_quot_mark])
    except Exception as ex:
        logging.warning(ex)

    logging.warning('Failed to parse test logs for checkpoint[Name=%s].' % checkpoint)
    return 0


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

    # Detect available GPUs
    free_gpus = get_assigned_gpus(args.gpu)
    logging.info('Availble GPUs: %s' % free_gpus)

    # Detect new checkpoints
    best_jf_mean = 0
    best_checkpoint = None
    running_processes = []
    tested_checkpoints = []
    test_results_buffer = OrderedDict()
    logging.info("Listening new checkpoints at %s ..." % args.ckpt_dir)

    while True:
        # Waiting for free GPUs & Processing test results
        while len(free_gpus) == 0:
            time.sleep(15)    # Detect free gpus every 15 seconds
            for rp in running_processes:
                if rp['process'].poll() is not None:
                    free_gpus.append(rp['gpu'])
                    running_processes.remove(rp)
                    # Parse the JF-Mean from test logs
                    jf_mean = get_jf_mean(rp['process'], rp['checkpoint'])
                    if jf_mean != 0:
                        logging.info(
                            'Checkpoint[Name=%s] has been tested successfully, JF-Mean = %.4f.' %
                            (rp['checkpoint'], jf_mean))
                    else:
                        logging.warning('Exception occurred during testing Checkpoint[Name=%s]' %
                                        rp['checkpoint'])
                        continue

                    # Add the results to the TensorBoard
                    test_results_buffer[rp["checkpoint"]] = jf_mean
                    test_results_buffer = add_scalars(test_writer, test_results_buffer)
                    # Update the best results
                    if jf_mean > best_jf_mean:
                        if best_checkpoint is not None:
                            os.remove(os.path.join(args.ckpt_dir, best_checkpoint))

                        best_jf_mean = jf_mean
                        best_checkpoint = rp['checkpoint']
                    elif jf_mean != 0:
                        os.remove(os.path.join(args.ckpt_dir, rp['checkpoint']))

        # Waiting for new checkpoints
        checkpoints = get_checkpoints(args.ckpt_dir, args.remote, tested_checkpoints)
        checkpoint = get_next_checkpoints(tested_checkpoints, checkpoints)
        if checkpoint is None:
            time.sleep(30)    # Detect new checkpoints every 30 seconds
            continue

        assigned_gpu = free_gpus.pop(0)
        logging.info('Assign GPU[ID=%s] for the checkpoint[Name=%s].' % (assigned_gpu, checkpoint))
        process = subprocess.Popen([
                "python", "runner.py", "--test", "--weights",
                os.path.join(args.ckpt_dir, checkpoint), "--gpu",
                str(assigned_gpu), "--cfg",
                str(args.cfg)
            ],
            cwd=os.path.abspath(os.pardir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        # Add the process to the checklist
        running_processes.append({
            'process': process,
            'gpu': assigned_gpu,
            'checkpoint': checkpoint
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
