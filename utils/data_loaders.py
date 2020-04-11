# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 16:43:59
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-11 10:31:57
# @Email:  cshzxie@gmail.com

import json
import logging
import numpy as np
import torch.utils.data.dataset

import utils.data_transforms

from enum import Enum, unique

from utils.io import IO


def collate_fn(batch_samples):
    video_names = []
    n_objects = []
    frames = []
    masks = []

    for bs in batch_samples:
        video_names.append(bs[0])
        n_objects.append(bs[1])
        frames.append(bs[2])
        masks.append(bs[3])

    return video_names, n_objects, frames, masks


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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        video = self.file_list[idx]
        frames = []
        masks = []

        for i in range(video['n_frames']):
            frame = np.array(IO.get(video['frames'][i]).convert('RGB'))
            mask = IO.get(video['masks'][i]).convert('P')
            frames.append(np.array(frame).astype(np.float32))
            masks.append(self._to_onehot(np.array(mask).astype(np.uint8), self.options['K']))

        if self.transforms is not None:
            frames, masks = self.transforms(frames, masks)

        return video['name'], video['n_objects'], frames, masks

    def _to_onehot(self, mask, k):
        h, w = mask.shape
        one_hot_masks = np.zeros((k, h, w), dtype=np.uint8)
        for k_idx in range(k):
            one_hot_masks[k_idx] = (mask == k_idx)

        return one_hot_masks


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
        return Dataset(file_list, transforms, {'K': self.cfg.DATASETS.DAVIS.K})

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([])
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
        if subset not in self.videos:
            logging.warn('The subset %s for DAVIS is not available.' % subset)
            return file_list

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
    'DAVIS': DavisDataset
}  # yapf: disable
