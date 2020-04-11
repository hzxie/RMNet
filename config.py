# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:05:17
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-10 20:40:06
# @Email:  cshzxie@gmail.com

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.DAVIS                               = edict()
__C.DATASETS.DAVIS.K                             = 11
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
__C.CONST.DATASET_MEAN                           = [0.485, 0.456, 0.406]
__C.CONST.DATASET_STD                            = [0.229, 0.224, 0.225]

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output'

#
# Networks
#
__C.NETWORKS                                     = edict()
__C.NETWORKS.MEM_EVERY                           = 5

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

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 4
__C.TRAIN.N_EPOCHS                               = 150
