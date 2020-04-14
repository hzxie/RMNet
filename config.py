# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:05:17
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-14 14:39:32
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
__C.DATASETS.ECSSD                               = edict()
__C.DATASETS.MSCOCO                              = edict()
__C.DATASETS.VOC2012                             = edict()
__C.DATASETS.MSRA10K                             = edict()

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: DAVIS, YOUTUBE_VOS, ECSSD, MSCOCO, VOC2012, MSRA10K
__C.DATASET.TRAIN_DATASET                        = 'DAVIS'
__C.DATASET.TEST_DATASET                         = 'DAVIS'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.N_WORKERS                              = 2
__C.CONST.INGORE_IDX                             = 255
__C.CONST.FRAME_SIZE                             = 480
__C.CONST.DATASET_MEAN                           = [0.485, 0.456, 0.406]
__C.CONST.DATASET_STD                            = [0.229, 0.224, 0.225]
__C.CONST.EXP_NAME                               = datetime.now().isoformat()

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output'

#
# Networks
#
__C.NETWORKS                                     = edict()
__C.NETWORKS.MEMORIZE_EVERY                      = 5

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 1
__C.TRAIN.N_EPOCHS                               = 150
__C.TRAIN.N_MAX_OBJECTS                          = 1
__C.TRAIN.N_MAX_FRAMES                           = 2
__C.TRAIN.LEARNING_RATE                          = 1e-5
__C.TRAIN.LR_MILESTONES                          = [100]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.SAVE_FREQ                              = 20

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.VISUALIZE_EVERY                         = 10
__C.TEST.MAIN_METRIC_NAME                        = 'JF-Mean'
