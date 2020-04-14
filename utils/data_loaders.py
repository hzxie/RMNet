# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 16:43:59
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-14 14:57:10
# @Email:  cshzxie@gmail.com

import json
import numpy as np
import random
import torch.utils.data.dataset

import utils.data_transforms
import utils.helpers

from enum import Enum, unique

from utils.io import IO


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list, transforms=None, options=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.frame_step = 1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        video = self.file_list[idx]
        frames = []
        masks = []

        frame_indexes = self._get_frame_indexes(video['n_frames'], self.options['n_max_frames'])
        for fi in frame_indexes:
            frame = np.array(IO.get(video['frames'][fi]).convert('RGB'))
            frames.append(np.array(frame).astype(np.float32))
            mask = IO.get(video['masks'][fi])
            mask = mask.convert('P') if mask is not None else np.zeros(frame.shape[:-1])
            masks.append(np.array(mask).astype(np.uint8))

        # Number of objects in the masks
        n_objects = min(video['n_objects'], self.options['n_max_objects'])

        # Data preprocessing and augmentation
        if self.transforms is not None:
            frames, masks = self.transforms(frames, masks, n_objects)

        # Masks to One Hot: (H, W) -> (n_object, H, W)
        masks = torch.stack(
            [utils.helpers.to_onehot(m, self.options['n_max_objects'] + 1) for m in masks], dim=0)

        return video['name'], n_objects, frames, masks

    def _get_frame_indexes(self, n_frames, n_max_frames):
        if n_frames <= n_max_frames or n_max_frames == 0:
            return range(n_frames)

        frame_begin_idx = n_frames - n_max_frames * self.frame_step
        frame_begin_idx = random.randint(0, frame_begin_idx) if frame_begin_idx > 0 else 0
        frame_end_idx = frame_begin_idx + n_max_frames * self.frame_step
        if frame_end_idx > n_frames:
            frame_end_idx = n_frames

        return [i for i in range(frame_begin_idx, frame_end_idx, self.frame_step)]

    def set_frame_step(self, frame_step):
        self.frame_step = frame_step


class DavisDataset(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # Load the dataset indexing file
        self.videos = []
        with open(cfg.DATASETS.DAVIS.INDEXING_FILE_PATH) as f:
            self.videos = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)

        n_max_frames = self.cfg.TRAIN.N_MAX_FRAMES if subset == DatasetSubset.TRAIN else 0
        n_max_objects = self.cfg.TRAIN.N_MAX_OBJECTS if subset == DatasetSubset.TRAIN else self.cfg.DATASETS.DAVIS.N_MAX_OBJECTS
        return Dataset(file_list, transforms, {
            'n_max_frames': n_max_frames,
            'n_max_objects': n_max_objects
        })

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomCrop',
                'parameters': {
                    'height': cfg.CONST.FRAME_SIZE,
                    'width': cfg.CONST.FRAME_SIZE
                }
            }, {
                'callback': 'Normalize',
                'parameters': {
                    'mean': cfg.CONST.DATASET_MEAN,
                    'std': cfg.CONST.DATASET_STD
                }
            }, {
                'callback': 'ToTensor',
                'parameters': None
            }])
        else:
            return utils.data_transforms.Compose([
                {
                    'callback': 'Normalize',
                    'parameters': {
                        'mean': cfg.CONST.DATASET_MEAN,
                        'std': cfg.CONST.DATASET_STD
                    }
                },
                {
                    'callback': 'ToTensor',
                    'parameters': None
                },
            ])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'val'

    def _get_file_list(self, cfg, subset):
        file_list = []
        for v in self.videos[subset]:
            file_list.append({
                'name': v['name'],
                'n_objects': v['n_objects'],
                'n_frames': v['n_frames'],
                'frames': [
                    cfg.DATASETS.DAVIS.IMG_FILE_PATH % (v['name'], i)
                    for i in range(v['n_frames'])
                ],
                'masks': [
                    cfg.DATASETS.DAVIS.ANNOTATION_FILE_PATH % (v['name'], i)
                    for i in range(v['n_frames'])
                ]
            })  # yapf: disable

        return file_list


DATASET_LOADER_MAPPING = {
    'DAVIS': DavisDataset,
}  # yapf: disable
