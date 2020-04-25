# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 16:43:59
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-25 12:09:25
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
            frames.append(np.array(frame))
            mask = IO.get(video['masks'][fi])
            mask = mask.convert('P') if mask is not None else np.zeros(frame.shape[:-1])
            masks.append(np.array(mask))

        # Number of objects in the masks
        if 'n_objects' not in video:
            mask_indexes = np.unique(masks[0])
            mask_indexes = mask_indexes[mask_indexes != self.options['ignore_idx']]
            video['n_objects'] = len(mask_indexes) - 1

        n_objects = min(video['n_objects'], self.options['n_max_objects'])

        # Data preprocessing and augmentation
        if self.transforms is not None:
            frames, masks = self.transforms(frames, masks)

        return video['name'], n_objects, frames, masks

    def _get_frame_indexes(self, n_frames, n_max_frames):
        if n_max_frames == 0:
            # Select all frames for testing
            return [i for i in range(n_frames)]
        elif n_frames <= n_max_frames:
            # Fix Bug: YouTube VOS [name=d177e9878a] only contains 2 frames
            return random.choices([i for i in range(n_frames)], k=n_max_frames)

        frame_begin_idx = n_frames - (n_max_frames - 1) * self.frame_step - 1
        frame_begin_idx = random.randint(0, frame_begin_idx) if frame_begin_idx > 0 else 0
        frame_end_idx = frame_begin_idx + (n_max_frames - 1) * self.frame_step

        # The frame_step can not be satisfied because the number of frames is not enough
        if frame_end_idx >= n_frames:
            return sorted(random.sample([i for i in range(n_frames)], n_max_frames))

        return [i for i in range(frame_begin_idx, frame_end_idx + 1, self.frame_step)]

    def set_frame_step(self, frame_step):
        self.frame_step = frame_step


class MultipleDatasets(torch.utils.data.dataset.Dataset):
    def __init__(self, datasets):
        self.frame_step = 1
        self.datasets = datasets
        # The begin and end indexes of datasets
        self.indexes = [0]
        for d in datasets:
            self.indexes.append(self.indexes[-1] + len(d))

    def __len__(self):
        return self.indexes[-1]

    def __getitem__(self, idx):
        # Determine which dataset to use in self.datasets
        dataset_idx = 0
        for i, dataset_end_idx in enumerate(self.indexes):
            if idx < dataset_end_idx:
                dataset_idx = i - 1
                break

        return self.datasets[dataset_idx][idx - self.indexes[dataset_idx]]

    def set_frame_step(self, frame_step):
        self.frame_step = frame_step
        for d in self.datasets:
            d.set_frame_step(frame_step)


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
        n_max_objects = self.cfg.TRAIN.N_MAX_OBJECTS if subset == DatasetSubset.TRAIN else self.cfg.TEST.N_MAX_OBJECTS
        return Dataset(file_list, transforms, {
            'n_max_frames': n_max_frames,
            'n_max_objects': n_max_objects
        })

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'Resize',
                'parameters': {
                    'size': cfg.TRAIN.AUGMENTATION.RESIZE_SIZE,
                    'keep_ratio': cfg.TRAIN.AUGMENTATION.RESIZE_KEEP_RATIO
                }
            }, {
                'callback': 'RandomAffine',
                'parameters': {
                    'degrees': cfg.TRAIN.AUGMENTATION.AFFINE_VIDEO_DEGREES,
                    'translate': cfg.TRAIN.AUGMENTATION.AFFINE_VIDEO_TRANSLATE,
                    'scale': cfg.TRAIN.AUGMENTATION.AFFINE_VIDEO_SCALE,
                    'shears': cfg.TRAIN.AUGMENTATION.AFFINE_VIDEO_SHEARS,
                    'frame_fill_color': cfg.TRAIN.AUGMENTATION.AFFINE_IMAGE_FILL_COLOR,
                    'mask_fill_color': cfg.TRAIN.AUGMENTATION.AFFINE_MASK_FILL_COLOR
                }
            }, {
                'callback': 'RandomCrop',
                'parameters': {
                    'height': cfg.TRAIN.AUGMENTATION.CROP_SIZE,
                    'width': cfg.TRAIN.AUGMENTATION.CROP_SIZE,
                    'ignore_idx': cfg.CONST.INGORE_IDX
                }
            }, {
                'callback': 'ReorganizeObjectID',
                'parameters': {
                    'ignore_idx': cfg.CONST.INGORE_IDX
                }
            }, {
                'callback': 'ToOneHot',
                'parameters': {
                    'shuffle': True,
                    'n_objects': cfg.TRAIN.N_MAX_OBJECTS
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
                    'callback': 'ReorganizeObjectID',
                    'parameters': {
                        'ignore_idx': cfg.CONST.INGORE_IDX
                    }
                },
                {
                    'callback': 'ToOneHot',
                    'parameters': {
                        'shuffle': False,
                        'n_objects': cfg.TEST.N_MAX_OBJECTS
                    }
                },
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


class YoutubeVosDataset(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # Load the dataset indexing file
        self.videos = []
        with open(cfg.DATASETS.YOUTUBE_VOS.INDEXING_FILE_PATH) as f:
            self.videos = json.loads(f.read())
            if 'videos' in self.videos:
                self.videos = self.videos['videos']

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg)
        transforms = self._get_transforms(self.cfg, subset)

        n_max_frames = self.cfg.TRAIN.N_MAX_FRAMES if subset == DatasetSubset.TRAIN else 0
        n_max_objects = self.cfg.TRAIN.N_MAX_OBJECTS if subset == DatasetSubset.TRAIN else self.cfg.TEST.N_MAX_OBJECTS
        return Dataset(
            file_list, transforms, {
                'ignore_idx': self.cfg.CONST.INGORE_IDX,
                'n_max_frames': n_max_frames,
                'n_max_objects': n_max_objects
            })

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'Resize',
                'parameters': {
                    'size': cfg.TRAIN.AUGMENTATION.RESIZE_SIZE,
                    'keep_ratio': cfg.TRAIN.AUGMENTATION.RESIZE_KEEP_RATIO
                }
            }, {
                'callback': 'RandomAffine',
                'parameters': {
                    'degrees': cfg.TRAIN.AUGMENTATION.AFFINE_VIDEO_DEGREES,
                    'translate': cfg.TRAIN.AUGMENTATION.AFFINE_VIDEO_TRANSLATE,
                    'scale': cfg.TRAIN.AUGMENTATION.AFFINE_VIDEO_SCALE,
                    'shears': cfg.TRAIN.AUGMENTATION.AFFINE_VIDEO_SHEARS,
                    'frame_fill_color': cfg.TRAIN.AUGMENTATION.AFFINE_IMAGE_FILL_COLOR,
                    'mask_fill_color': cfg.TRAIN.AUGMENTATION.AFFINE_MASK_FILL_COLOR
                }
            }, {
                'callback': 'RandomCrop',
                'parameters': {
                    'height': cfg.TRAIN.AUGMENTATION.CROP_SIZE,
                    'width': cfg.TRAIN.AUGMENTATION.CROP_SIZE,
                    'ignore_idx': cfg.CONST.INGORE_IDX
                }
            }, {
                'callback': 'ReorganizeObjectID',
                'parameters': {
                    'ignore_idx': cfg.CONST.INGORE_IDX
                }
            }, {
                'callback': 'ToOneHot',
                'parameters': {
                    'shuffle': True,
                    'n_objects': cfg.TRAIN.N_MAX_OBJECTS
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
                    'callback': 'ReorganizeObjectID',
                    'parameters': {
                        'ignore_idx': cfg.CONST.INGORE_IDX
                    }
                },
                {
                    'callback': 'ToOneHot',
                    'parameters': {
                        'shuffle': False,
                        'n_objects': cfg.TEST.N_MAX_OBJECTS
                    }
                },
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

    def _get_file_list(self, cfg):
        file_list = []
        for v in self.videos:
            video = self.videos[v]
            frame_indexes = set({})
            for o_idx, o_value in video['objects'].items():
                frame_indexes.update(o_value['frames'])

            frame_indexes = sorted(list(frame_indexes))
            file_list.append({
                'name': v,
                'n_frames': len(frame_indexes),
                'frames': [
                    cfg.DATASETS.YOUTUBE_VOS.IMG_FILE_PATH % (v, i)
                    for i in frame_indexes
                ],
                'masks': [
                    cfg.DATASETS.YOUTUBE_VOS.ANNOTATION_FILE_PATH % (v, i)
                    for i in frame_indexes
                ]
            })  # yapf: disable

        return file_list


class ImageDataset(object):
    def get_dataset(self, subset):
        if not subset == DatasetSubset.TRAIN:
            raise Exception('ONLY DatasetSubset.TRAIN is available for ImageDataset.')

        file_list = self._get_file_list(self.cfg)
        transforms = self._get_transforms(self.cfg)
        return Dataset(
            file_list, transforms, {
                'ignore_idx': self.cfg.CONST.INGORE_IDX,
                'n_max_frames': self.cfg.TRAIN.N_MAX_FRAMES,
                'n_max_objects': self.cfg.TRAIN.N_MAX_OBJECTS
            })

    def _get_transforms(self, cfg):
        return utils.data_transforms.Compose([{
            'callback': 'Resize',
            'parameters': {
                'size': cfg.TRAIN.AUGMENTATION.RESIZE_SIZE,
                'keep_ratio': cfg.TRAIN.AUGMENTATION.RESIZE_KEEP_RATIO
            }
        }, {
            'callback': 'RandomAffine',
            'parameters': {
                'degrees': cfg.TRAIN.AUGMENTATION.AFFINE_IMAGE_DEGREES,
                'translate': cfg.TRAIN.AUGMENTATION.AFFINE_IMAGE_TRANSLATE,
                'scale': cfg.TRAIN.AUGMENTATION.AFFINE_IMAGE_SCALE,
                'shears': cfg.TRAIN.AUGMENTATION.AFFINE_IMAGE_SHEARS,
                'frame_fill_color': cfg.TRAIN.AUGMENTATION.AFFINE_IMAGE_FILL_COLOR,
                'mask_fill_color': cfg.TRAIN.AUGMENTATION.AFFINE_MASK_FILL_COLOR
            }
        }, {
            'callback': 'RandomCrop',
            'parameters': {
                'height': cfg.TRAIN.AUGMENTATION.CROP_SIZE,
                'width': cfg.TRAIN.AUGMENTATION.CROP_SIZE,
                'ignore_idx': cfg.CONST.INGORE_IDX
            }
        }, {
            'callback': 'ReorganizeObjectID',
            'parameters': {
                'ignore_idx': cfg.CONST.INGORE_IDX
            }
        }, {
            'callback': 'ToOneHot',
            'parameters': {
                'shuffle': True,
                'n_objects': cfg.TRAIN.N_MAX_OBJECTS
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


class PascalVocDataset(ImageDataset):
    def __init__(self, cfg):
        super(PascalVocDataset, self).__init__()

        self.cfg = cfg
        # Load the dataset indexing file
        self.images = []
        with open(cfg.DATASETS.PASCAL_VOC.INDEXING_FILE_PATH) as f:
            self.images = f.read().split('\n')[:-1]

    def _get_file_list(self, cfg):
        file_list = []
        for i in self.images:
            file_list.append({
                'name': i,
                'n_frames': 1,
                'frames': [
                    cfg.DATASETS.PASCAL_VOC.IMG_FILE_PATH % i
                ],
                'masks': [
                    cfg.DATASETS.PASCAL_VOC.ANNOTATION_FILE_PATH % i
                ]
            })  # yapf: disable

        return file_list


class EcssdDataset(ImageDataset):
    def __init__(self, cfg):
        super(EcssdDataset, self).__init__()

        self.cfg = cfg
        # Load the dataset indexing file
        self.images = ['%04d' % i for i in range(1, cfg.DATASETS.ECSSD.N_IMAGES + 1)]

    def _get_file_list(self, cfg):
        file_list = []
        for i in self.images:
            file_list.append({
                'name': i,
                'n_frames': 1,
                'frames': [
                    cfg.DATASETS.ECSSD.IMG_FILE_PATH % i
                ],
                'masks': [
                    cfg.DATASETS.ECSSD.ANNOTATION_FILE_PATH % i
                ]
            })  # yapf: disable

        return file_list


class Msra10kDataset(ImageDataset):
    def __init__(self, cfg):
        super(Msra10kDataset, self).__init__()

        self.cfg = cfg
        # Load the dataset indexing file
        self.images = []
        with open(cfg.DATASETS.MSRA10K.INDEXING_FILE_PATH) as f:
            self.images = f.read().split('\n')

    def _get_file_list(self, cfg):
        file_list = []
        for i in self.images:
            file_list.append({
                'name': i,
                'n_frames': 1,
                'frames': [
                    cfg.DATASETS.MSRA10K.IMG_FILE_PATH % i
                ],
                'masks': [
                    cfg.DATASETS.MSRA10K.ANNOTATION_FILE_PATH % i
                ]
            })  # yapf: disable

        return file_list


class MscocoDataset(ImageDataset):
    def __init__(self, cfg):
        super(MscocoDataset, self).__init__()

        self.cfg = cfg
        # Load the dataset indexing file
        self.images = []
        with open(cfg.DATASETS.MSCOCO.INDEXING_FILE_PATH) as f:
            self.images = f.read().split('\n')

    def _get_file_list(self, cfg):
        file_list = []
        for i in self.images:
            file_list.append({
                'name': i,
                'n_frames': 1,
                'frames': [
                    cfg.DATASETS.MSCOCO.IMG_FILE_PATH % i
                ],
                'masks': [
                    cfg.DATASETS.MSCOCO.ANNOTATION_FILE_PATH % i
                ]
            })  # yapf: disable

        return file_list


class DatasetCollector(object):
    DATASET_LOADER_MAPPING = {
        'DAVIS': DavisDataset,
        'YOUTUBE_VOS': YoutubeVosDataset,
        'PASCAL_VOC': PascalVocDataset,
        'ECSSD': EcssdDataset,
        'MSRA10K': Msra10kDataset,
        'MSCOCO': MscocoDataset,
    }  # yapf: disable

    @classmethod
    def get_dataset(cls, cfg, dataset, subset):
        if type(dataset) == str:
            return cls.DATASET_LOADER_MAPPING[dataset](cfg).get_dataset(subset)
        elif type(dataset) == list:
            datasets = []
            for dn in dataset:
                x_index = dn.rfind('x')
                repeat_times = int(dn[x_index + 1:]) if x_index != -1 else 1
                dn = dn[:x_index] if x_index != -1 else dn

                for i in range(repeat_times):
                    datasets.append(cls.DATASET_LOADER_MAPPING[dn](cfg).get_dataset(subset))

            return MultipleDatasets(datasets)
        else:
            raise Exception('Unknown dataset format: %s' % dataset)
