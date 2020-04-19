# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:05:17
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-19 12:37:32
# @Email:  cshzxie@gmail.com

from datetime import datetime
from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.DAVIS                               = edict()
__C.DATASETS.DAVIS.N_MAX_OBJECTS                 = 11
__C.DATASETS.DAVIS.INDEXING_FILE_PATH            = './datasets/DAVIS.json'
__C.DATASETS.DAVIS.IMG_FILE_PATH                 = '/home/SENSETIME/xiehaozhe/Datasets/DAVIS/JPEGImages/480p/%s/%05d.jpg'
__C.DATASETS.DAVIS.ANNOTATION_FILE_PATH          = '/home/SENSETIME/xiehaozhe/Datasets/DAVIS/Annotations/480p/%s/%05d.png'
__C.DATASETS.YOUTUBE_VOS                         = edict()
__C.DATASETS.YOUTUBE_VOS.N_MAX_OBJECTS           = 11
__C.DATASETS.YOUTUBE_VOS.INDEXING_FILE_PATH      = '/home/SENSETIME/xiehaozhe/Datasets/ytb_train/meta.json'
__C.DATASETS.YOUTUBE_VOS.IMG_FILE_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/ytb_train/JPEGImages/%s/%s.jpg'
__C.DATASETS.YOUTUBE_VOS.ANNOTATION_FILE_PATH    = '/home/SENSETIME/xiehaozhe/Datasets/ytb_train/Annotations/%s/%s.png'
__C.DATASETS.ECSSD                               = edict()
__C.DATASETS.MSCOCO                              = edict()
__C.DATASETS.VOC2012                             = edict()
__C.DATASETS.MSRA10K                             = edict()

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: DAVIS, YOUTUBE_VOS, ECSSD, MSCOCO, VOC2012, MSRA10K
__C.DATASET.TRAIN_DATASET                        = ['YOUTUBE_VOS', 'DAVISx5']
__C.DATASET.TEST_DATASET                         = 'DAVIS'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.N_WORKERS                              = 2
__C.CONST.INGORE_IDX                             = 255
__C.CONST.DATASET_MEAN                           = [0.485, 0.456, 0.406]
__C.CONST.DATASET_STD                            = [0.229, 0.224, 0.225]
__C.CONST.EXP_NAME                               = datetime.now().isoformat()

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output'

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# PAVI
#
__C.PAVI                                         = edict()
__C.PAVI.ENABLED                                 = False
__C.PAVI.PROJECT_NAME                            = 'Semi-Video-Segmentation'
__C.PAVI.TAGS                                    = ['stm']


#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 4
__C.TRAIN.N_EPOCHS                               = 150
__C.TRAIN.N_MAX_OBJECTS                          = 3
__C.TRAIN.N_MAX_FRAMES                           = 3
__C.TRAIN.FRAME_STEP_MILESTONES                  = []
__C.TRAIN.LEARNING_RATE                          = 1e-5
__C.TRAIN.LR_MILESTONES                          = [100]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.SAVE_FREQ                              = 20
__C.TRAIN.MEMORIZE_EVERY                         = 1
__C.TRAIN.AUGMENTATION                           = edict()
__C.TRAIN.AUGMENTATION.RESIZE_SIZE               = 480
__C.TRAIN.AUGMENTATION.RESIZE_KEEP_RATIO         = True
__C.TRAIN.AUGMENTATION.CROP_SIZE                 = 480
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_DEGREES      = (-20, 20)
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_TRANSLATE    = (0, 0)
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_SCALE        = (0.9, 1.1)
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_SHEARS       = (6, 9)
__C.TRAIN.AUGMENTATION.AFFINE_VIDEO_DEGREES      = (-15, 15)
__C.TRAIN.AUGMENTATION.AFFINE_VIDEO_TRANSLATE    = (0, 0)
__C.TRAIN.AUGMENTATION.AFFINE_VIDEO_SCALE        = (0.95, 1.05)
__C.TRAIN.AUGMENTATION.AFFINE_VIDEO_SHEARS       = (-10, 10)
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_FILL_COLOR   = (255, 255, 255)
__C.TRAIN.AUGMENTATION.AFFINE_MASK_FILL_COLOR    = 255

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.VISUALIZE_EVERY                         = 10
__C.TEST.MEMORIZE_EVERY                          = 5
__C.TEST.N_TESTING_VIDEOS                        = 10
__C.TEST.MAIN_METRIC_NAME                        = 'JF-Mean'
