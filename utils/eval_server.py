#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-08-08 17:16:07
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-10 09:32:25
# @Email:  cshzxie@gmail.com

import argparse
import gpustat
import logging
import os
import tensorboardX
import time
import torch
import subprocess
import sys


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description="The argument parser of eval_server")
    parser.add_argument("ckpt_dir", help="The path to folder that saves checkpoints")
    parser.add_argument("--cfg", help="The path to config file", default="config.py", type=str)
    parser.add_argument("--gpu", help="The GPU IDs to use", default=None, type=str)
    args = parser.parse_args()
    return args


def get_summary_writer(log_dir):
    return tensorboardX.SummaryWriter(log_dir)


def get_assigned_gpus(gpu_id):
    if gpu_id is not None:
        return [int(gpu) for gpu in gpu_id.split(',')]

    free_gpus = [gpu.index for gpu in gpustat.new_query() if gpu.memory_used == 0]
    # The below statement will cost 10MB/11MB on the allocated GPUs
    torch.cuda.is_available()

    return [
        gpu.index for gpu in gpustat.new_query() if gpu.memory_used > 0 and gpu.index in free_gpus
    ]


def get_next_checkpoints(tested_checkpoints, now_checkpoints):
    for nc in now_checkpoints:
        if nc not in tested_checkpoints:
            return nc

    return None


def get_jf_mean(process, checkpoint):
    try:
        logs = process.stdout.read().decode('utf-8').split("\n")
        for log in logs:
            if log.find('[Test Summary]') != -1:
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
    if not os.path.exists(args.ckpt_dir):
        logging.error("The checkpoints folder[Path=%s] does not exist." % args.ckpt_dir)
        sys.exit(2)
    else:
        args.ckpt_dir = os.path.abspath(args.ckpt_dir)
        args.cfg = os.path.abspath(args.cfg)
        logging.info("Evaluation server started.")

    # Set up new summary writer
    log_dir = os.path.join(args.ckpt_dir.replace("checkpoints", "logs"), "val")
    test_writer = get_summary_writer(log_dir)
    logging.info("Tensorborad Logs available at: %s" % log_dir)

    # Detect available GPUs
    free_gpus = get_assigned_gpus(args.gpu)
    logging.info("Availble GPUs: %s" % free_gpus)

    # Detect new checkpoints
    best_jf_mean = 0
    best_checkpoint = None
    running_processes = []
    tested_checkpoints = []
    logging.info("Listening new checkpoints at %s ..." % args.ckpt_dir)

    while True:
        # Waiting for free GPUs & Processing test results
        while len(free_gpus) == 0:
            time.sleep(15)    # Detect free gpus every 15 seconds
            for rp in running_processes:
                if rp["process"].poll() is not None:
                    free_gpus.append(rp["gpu"])
                    running_processes.remove(rp)
                    # Parse the JF-Mean from test logs
                    jf_mean = get_jf_mean(rp["process"], rp["checkpoint"])
                    logging.info(
                        "Checkpoint[Name=%s] has been tested successfully, JF-Mean = %.4f." %
                        (rp["checkpoint"], jf_mean))
                    # Add the results to the TensorBoard
                    ckpt_epoch = int(rp["checkpoint"][len('ckpt-epoch-'):-len('.pth')])
                    test_writer.add_scalar('Metric/JF-Mean', jf_mean, ckpt_epoch)
                    # Update the best results
                    if jf_mean > best_jf_mean:
                        if best_checkpoint is not None:
                            os.remove(os.path.join(args.ckpt_dir, best_checkpoint))

                        best_jf_mean = jf_mean
                        best_checkpoint = rp["checkpoint"]
                    else:
                        os.remove(os.path.join(args.ckpt_dir, rp["checkpoint"]))

        # Waiting for new checkpoints
        checkpoints = sorted([f for f in os.listdir(args.ckpt_dir) if f.startswith("ckpt-epoch")])
        checkpoint = get_next_checkpoints(tested_checkpoints, checkpoints)
        if checkpoint is None:
            time.sleep(30)    # Detect new checkpoints every 30 seconds
            continue

        assigned_gpu = free_gpus.pop(0)
        logging.info("Assign GPU[ID=%s] for the checkpoint[Name=%s]." % (assigned_gpu, checkpoint))
        process = subprocess.Popen(
            [
                "python",
                "runner.py",
                "--test",
                "--weights",
                os.path.join(args.ckpt_dir, checkpoint),
                "--gpu",
                str(assigned_gpu),
                "--cfg",
                str(args.cfg)
            ],
            cwd=os.path.abspath(os.pardir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        # Add the process to the checklist
        running_processes.append({
            "process": process,
            "gpu": assigned_gpu,
            "checkpoint": checkpoint
        })
        # Remember the tested checkpoint
        tested_checkpoints.append(checkpoint)


if __name__ == "__main__":
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please use Python 3.x")

    logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s")
    logging.getLogger().setLevel(logging.INFO)

    main()
