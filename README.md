# RMNet

This repository contains the source code for the paper [Efficient Regional Memory Network for Video Object Segmentation](https://arxiv.org/abs/2103.12934).

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=hzxie_RMNet&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=hzxie_RMNet)
[![codefactor badge](https://www.codefactor.io/repository/github/hzxie/RMNet/badge)](https://www.codefactor.io/repository/github/hzxie/RMNet)

![Overview](https://www.infinitescript.com/projects/RMNet/RMNet-Overview.jpg)

## Cite this work

```
@inproceedings{xie2021efficient,
  title={Efficient Regional Memory Network for Video Object Segmentation},
  author={Xie, Haozhe and 
          Yao, Hongxun and 
          Zhou, Shangchen and 
          Zhang, Shengping and 
          Sun, Wenxiu},
  booktitle={CVPR},
  year={2021}
}
```

## Datasets

We use the [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [COCO](https://cocodataset.org/), [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [MSRA10K](https://mmcheng.net/msra10k/), [DAVIS](https://davischallenge.org/), and [YouTube-VOS](http://youtube-vos.org/) datasets in our experiments, which are available below:

- [ECSSD Images](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip) / [Masks](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip)
- [COCO Images](http://images.cocodataset.org/zips/train2017.zip) / [Masks](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
- [MSRA10K](http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip)
- [DAVIS 2017 Train/Val](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)
- [DAVIS 2017 Test-dev](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip)
- [YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate)

## Pretrained Models

The pretrained models for DAVIS and YouTube-VOS are available as follows:

- [RMNet for DAVIS](https://gateway.infinitescript.com/?fileName=RMNet-DAVIS.pth) (202 MB)
- [RMNet for YouTube-VOS](https://gateway.infinitescript.com/?fileName=RMNet-YouTubeVOS.pth) (202 MB)

## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/hzxie/RMNet.git
```

#### Install Python Denpendencies

```
cd RMNet
pip install -r requirements.txt
```

#### Build PyTorch Extensions

**NOTE:** PyTorch >= 1.4, CUDA >= 9.0 and GCC >= 4.9 are required.

```
RMNET_HOME=`pwd`

cd $RMNET_HOME/extensions/reg_att_map_generator
python setup.py install --user

cd $RMNET_HOME/extensions/flow_affine_transformation
python setup.py install --user
```

#### Precompute the Optical Flow

- For the DAVIS dataset, the optical flows are computed by [FlowNet2-CSS](https://github.com/NVIDIA/flownet2-pytorch) with [the model pretrained on FlyingThings3D](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing).
- For the YouTube-VOS dataset, the optical flows are computed by [RAFT](https://github.com/princeton-vl/RAFT) with [the model pretrained on Sintel](https://drive.google.com/file/d/1fubTHIa_b2C8HqfbPtKXwoRd9QsYxRL6/view?usp=sharing).

#### Update Settings in `config.py`

You need to update the file path of the datasets:

```
__C.DATASETS                                     = edict()
__C.DATASETS.DAVIS                               = edict()
__C.DATASETS.DAVIS.INDEXING_FILE_PATH            = './datasets/DAVIS.json'
__C.DATASETS.DAVIS.IMG_FILE_PATH                 = '/path/to/Datasets/DAVIS/JPEGImages/480p/%s/%05d.jpg'
__C.DATASETS.DAVIS.ANNOTATION_FILE_PATH          = '/path/to/Datasets/DAVIS/Annotations/480p/%s/%05d.png'
__C.DATASETS.DAVIS.OPTICAL_FLOW_FILE_PATH        = '/path/to/Datasets/DAVIS/OpticalFlows/480p/%s/%05d.flo'
__C.DATASETS.YOUTUBE_VOS                         = edict()
__C.DATASETS.YOUTUBE_VOS.INDEXING_FILE_PATH      = '/path/to/Datasets/YouTubeVOS/%s/meta.json'
__C.DATASETS.YOUTUBE_VOS.IMG_FILE_PATH           = '/path/to/Datasets/YouTubeVOS/%s/JPEGImages/%s/%s.jpg'
__C.DATASETS.YOUTUBE_VOS.ANNOTATION_FILE_PATH    = '/path/to/Datasets/YouTubeVOS/%s/Annotations/%s/%s.png'
__C.DATASETS.YOUTUBE_VOS.OPTICAL_FLOW_FILE_PATH  = '/path/to/Datasets/YouTubeVOS/%s/OpticalFlows/%s/%s.flo'
__C.DATASETS.PASCAL_VOC                          = edict()
__C.DATASETS.PASCAL_VOC.INDEXING_FILE_PATH       = '/path/to/Datasets/voc2012/trainval.txt'
__C.DATASETS.PASCAL_VOC.IMG_FILE_PATH            = '/path/to/Datasets/voc2012/images/%s.jpg'
__C.DATASETS.PASCAL_VOC.ANNOTATION_FILE_PATH     = '/path/to/Datasets/voc2012/masks/%s.png'
__C.DATASETS.ECSSD                               = edict()
__C.DATASETS.ECSSD.N_IMAGES                      = 1000
__C.DATASETS.ECSSD.IMG_FILE_PATH                 = '/path/to/Datasets/ecssd/images/%s.jpg'
__C.DATASETS.ECSSD.ANNOTATION_FILE_PATH          = '/path/to/Datasets/ecssd/masks/%s.png'
__C.DATASETS.MSRA10K                             = edict()
__C.DATASETS.MSRA10K.INDEXING_FILE_PATH          = './datasets/msra10k.txt'
__C.DATASETS.MSRA10K.IMG_FILE_PATH               = '/path/to/Datasets/msra10k/images/%s.jpg'
__C.DATASETS.MSRA10K.ANNOTATION_FILE_PATH        = '/path/to/Datasets/msra10k/masks/%s.png'
__C.DATASETS.MSCOCO                              = edict()
__C.DATASETS.MSCOCO.INDEXING_FILE_PATH           = './datasets/mscoco.txt'
__C.DATASETS.MSCOCO.IMG_FILE_PATH                = '/path/to/Datasets/coco2017/images/train2017/%s.jpg'
__C.DATASETS.MSCOCO.ANNOTATION_FILE_PATH         = '/path/to/Datasets/coco2017/masks/train2017/%s.png'
__C.DATASETS.ADE20K                              = edict()
__C.DATASETS.ADE20K.INDEXING_FILE_PATH           = './datasets/ade20k.txt'
__C.DATASETS.ADE20K.IMG_FILE_PATH                = '/path/to/Datasets/ADE20K_2016_07_26/images/training/%s.jpg'
__C.DATASETS.ADE20K.ANNOTATION_FILE_PATH         = '/path/to/Datasets/ADE20K_2016_07_26/images/training/%s_seg.png'

# Dataset Options: DAVIS, DAVIS_FRAMES, YOUTUBE_VOS, ECSSD, MSCOCO, PASCAL_VOC, MSRA10K, ADE20K
__C.DATASET.TRAIN_DATASET                        = ['ECSSD', 'PASCAL_VOC', 'MSRA10K', 'MSCOCO']  # Pretrain
__C.DATASET.TRAIN_DATASET                        = ['YOUTUBE_VOS', 'DAVISx5']                    # Fine-tune
__C.DATASET.TEST_DATASET                         = 'DAVIS'

# Network Options: RMNet, TinyFlowNet
__C.TRAIN.NETWORK                                = 'RMNet'
```

## Get Started

To train RMNet, you can simply use the following command:

```
python3 runner.py
```

To test RMNet, you can use the following command:

```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```

## License

This project is open sourced under MIT license.
