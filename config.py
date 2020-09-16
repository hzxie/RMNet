# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:05:17
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-09-16 13:03:26
# @Email:  cshzxie@gmail.com

from datetime import datetime
from easydict import EasyDict as edict

__C                                              = edict()
#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.DAVIS                               = edict()
__C.DATASETS.DAVIS.INDEXING_FILE_PATH            = './datasets/DAVIS.json'
__C.DATASETS.DAVIS.IMG_FILE_PATH                 = '/home/SENSETIME/xiehaozhe/Datasets/DAVIS/JPEGImages/480p/%s/%05d.jpg'
__C.DATASETS.DAVIS.ANNOTATION_FILE_PATH          = '/home/SENSETIME/xiehaozhe/Datasets/DAVIS/Annotations/480p/%s/%05d.png'
__C.DATASETS.DAVIS.OPTICAL_FLOW_FILE_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/DAVIS/OpticalFlows/480p/%s/%05d.flo'
__C.DATASETS.YOUTUBE_VOS                         = edict()
__C.DATASETS.YOUTUBE_VOS.INDEXING_FILE_PATH      = '/home/SENSETIME/xiehaozhe/Datasets/YouTubeVOS/%s/meta.json'
__C.DATASETS.YOUTUBE_VOS.IMG_FILE_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/YouTubeVOS/%s/JPEGImages/%s/%s.jpg'
__C.DATASETS.YOUTUBE_VOS.ANNOTATION_FILE_PATH    = '/home/SENSETIME/xiehaozhe/Datasets/YouTubeVOS/%s/Annotations/%s/%s.png'
__C.DATASETS.YOUTUBE_VOS.OPTICAL_FLOW_FILE_PATH  = '/home/SENSETIME/xiehaozhe/Datasets/YouTubeVOS/%s/OpticalFlows/%s/%s.flo'
__C.DATASETS.PASCAL_VOC                          = edict()
__C.DATASETS.PASCAL_VOC.INDEXING_FILE_PATH       = '/home/SENSETIME/xiehaozhe/Datasets/voc2012/trainval.txt'
__C.DATASETS.PASCAL_VOC.IMG_FILE_PATH            = '/home/SENSETIME/xiehaozhe/Datasets/voc2012/images/%s.jpg'
__C.DATASETS.PASCAL_VOC.ANNOTATION_FILE_PATH     = '/home/SENSETIME/xiehaozhe/Datasets/voc2012/masks/%s.png'
__C.DATASETS.ECSSD                               = edict()
__C.DATASETS.ECSSD.N_IMAGES                      = 1000
__C.DATASETS.ECSSD.IMG_FILE_PATH                 = '/home/SENSETIME/xiehaozhe/Datasets/ecssd/images/%s.jpg'
__C.DATASETS.ECSSD.ANNOTATION_FILE_PATH          = '/home/SENSETIME/xiehaozhe/Datasets/ecssd/masks/%s.png'
__C.DATASETS.MSRA10K                             = edict()
__C.DATASETS.MSRA10K.INDEXING_FILE_PATH          = './datasets/msra10k.txt'
__C.DATASETS.MSRA10K.IMG_FILE_PATH               = '/home/SENSETIME/xiehaozhe/Datasets/msra10k/images/%s.jpg'
__C.DATASETS.MSRA10K.ANNOTATION_FILE_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/msra10k/masks/%s.png'
__C.DATASETS.MSCOCO                              = edict()
__C.DATASETS.MSCOCO.INDEXING_FILE_PATH           = './datasets/mscoco.txt'
__C.DATASETS.MSCOCO.IMG_FILE_PATH                = '/home/SENSETIME/xiehaozhe/Datasets/coco2017/images/train2017/%s.jpg'
__C.DATASETS.MSCOCO.ANNOTATION_FILE_PATH         = '/home/SENSETIME/xiehaozhe/Datasets/coco2017/masks/train2017/%s.png'
__C.DATASETS.ADE20K                              = edict()
__C.DATASETS.ADE20K.INDEXING_FILE_PATH           = './datasets/ade20k.txt'
__C.DATASETS.ADE20K.IMG_FILE_PATH                = '/home/SENSETIME/xiehaozhe/Datasets/ADE20K_2016_07_26/images/training/%s.jpg'
__C.DATASETS.ADE20K.ANNOTATION_FILE_PATH         = '/home/SENSETIME/xiehaozhe/Datasets/ADE20K_2016_07_26/images/training/%s_seg.png'

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: DAVIS, DAVIS_FRAMES, YOUTUBE_VOS, ECSSD, MSCOCO, PASCAL_VOC, MSRA10K, ADE20K
__C.DATASET.TRAIN_DATASET                        = ['ECSSD', 'PASCAL_VOC', 'MSRA10K', 'MSCOCO']
__C.DATASET.TRAIN_DATASET                        = ['YOUTUBE_VOS', 'DAVISx5']
__C.DATASET.TEST_DATASET                         = 'DAVIS'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.RNG_SEED                               = 0
__C.CONST.N_WORKERS                              = 2
__C.CONST.IGNORE_IDX                             = 255
__C.CONST.DATASET_MEAN                           = [0.485, 0.456, 0.406]
__C.CONST.DATASET_STD                            = [0.229, 0.224, 0.225]
__C.CONST.EXP_NAME                               = datetime.now().isoformat()

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUTPUT_DIR                               = './output'

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
__C.TRAIN.N_EPOCHS                               = 200
__C.TRAIN.N_MAX_OBJECTS                          = 3
__C.TRAIN.N_MAX_FRAMES                           = 3
__C.TRAIN.USE_RANDOM_FRAME_STEPS                 = True
__C.TRAIN.USE_BATCH_NORM                         = False
__C.TRAIN.MAX_FRAME_STEPS                        = 20
__C.TRAIN.LAST_N_EPOCHES_FIXING_FRAME_STEPS      = 50
__C.TRAIN.LEARNING_RATE                          = 1e-5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.CKPT_SAVE_FREQ                         = 1
__C.TRAIN.CKPT_SAVE_THRESHOLD                    = 0.71
__C.TRAIN.MEMORIZE_EVERY                         = 1
__C.TRAIN.AUGMENTATION                           = edict()
__C.TRAIN.AUGMENTATION.RESIZE_SIZE               = 480
__C.TRAIN.AUGMENTATION.RESIZE_KEEP_RATIO         = True
__C.TRAIN.AUGMENTATION.CROP_HSIZE                = 465
__C.TRAIN.AUGMENTATION.CROP_WSIZE                = 465
__C.TRAIN.AUGMENTATION.COLOR_BRIGHTNESS          = (0.97, 1.03)
__C.TRAIN.AUGMENTATION.COLOR_CONTRAST            = None
__C.TRAIN.AUGMENTATION.COLOR_SATURATION          = None
__C.TRAIN.AUGMENTATION.COLOR_HUE                 = None
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_DEGREES      = (-20, 20)
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_TRANSLATE    = (0, 0)
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_SCALE        = (0.9, 1.1)
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_SHEARS       = (-10, 10)
__C.TRAIN.AUGMENTATION.AFFINE_VIDEO_DEGREES      = (-15, 15)
__C.TRAIN.AUGMENTATION.AFFINE_VIDEO_TRANSLATE    = (0, 0)
__C.TRAIN.AUGMENTATION.AFFINE_VIDEO_SCALE        = (1.0, 1.3)
__C.TRAIN.AUGMENTATION.AFFINE_VIDEO_SHEARS       = (-10, 10)
__C.TRAIN.AUGMENTATION.AFFINE_IMAGE_FILL_COLOR   = (255, 255, 255)
__C.TRAIN.AUGMENTATION.AFFINE_MASK_FILL_COLOR    = 255
__C.TRAIN.AUGMENTATION.AFFINE_FLOW_FILL_COLOR    = (0, 0)

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.N_MAX_OBJECTS                           = 10
__C.TEST.VISUALIZE_EVERY                         = 10
__C.TEST.MEMORIZE_EVERY                          = 5
__C.TEST.MAIN_METRIC_NAME                        = 'JF-Mean'
__C.TEST.FLIP_LR                                 = True
__C.TEST.FRAME_SCALES                            = [1.0, 1.25, 1.5]
# DAVIS
__C.TEST.TESTING_VIDEOS_INDEXES                  = [0, 2, 3, 8, 10, 18, 19, 24, 27, 29]
# YouTube VOS 2019
# __C.TEST.TESTING_VIDEOS_INDEXES                = [38, 43, 59, 156, 184, 257, 267, 503]
