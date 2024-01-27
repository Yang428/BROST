# BROST - Accurate and Robust Visual Tracking Using Bounding Box Refinement and Online Sample Filtering

## Publication
Yijin Yang and Xiaodong Gu.
Accurate and robust visual tracking using bounding box refinement and online sample filtering.
Signal Processing: Image Communication, 116(116981), 2023.[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0923596523000632)

# Abstract
Discriminative correlation trackers have currently achieved excellent performance in terms of tracking robustness. However, these trackers still suffer from limited precision of bounding box estimation due to the challenging factors of occlusion, deformation and rotation. In this paper, in order to address these issues, we propose a three-stage tracking framework called BROST. The proposed tracker is mainly composed of DCF module, segmentation module and box refinement module. Firstly, the proposed tracker roughly locates the center position of the object through the DCF module, then utilizes the segmentation module to estimate the scale of the object and finally employs the box refinement module to improve the accuracy of target box estimation. In order to achieve high tracking robustness, we develop a confidence function of correlation response map to filter out the corrupted or occluded training samples of DCF module. Besides, we introduce a new mask initialization network into the segmentation module to make it more suitable for tracking task. The comprehensive experimental results on six challenging visual tracking benchmarks show that the proposed BROST tracker outperforms most of the state-of-the-art trackers and achieves favorable tracking performance on VOT benchmarks.

## Running Environments
* Pytorch 1.1.0, Python 3.6.12, Cuda 9.0, torchvision 0.3.0, cudatoolkit 9.0, Matlab R2016b.
* Ubuntu 16.04, NVIDIA GeForce GTX 1080Ti, i7-6700K CPU @ 4.00GHz.

## Installation
The instructions have been tested on an Ubuntu 16.04 system. In case of issues, we refer to these two links [1](https://github.com/alanlukezic/d3s) and [2](https://github.com/visionml/pytracking) for details.

#### Clone the GIT repository
```
git clone https://github.com/Yang428/BROST.git.
```

#### Install dependent libraries
Run the installation script 'install.sh' to install all dependencies. We refer to [this link](https://github.com/visionml/pytracking/blob/master/INSTALL.md) for step-by-step instructions.
```
bash install.sh conda_install_path pytracking
```

#### Or step by step install
```
conda create -n pytracking python=3.6
conda activate pytracking
conda install -y pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
conda install -y matplotlib=2.2.2
conda install -y pandas
pip install opencv-python
pip install tensorboardX
conda install -y cython
pip install pycocotools
pip install jpeg4py 
sudo apt-get install libturbojpeg
```

#### Or copy my environment directly.

You can download the packed conda environment from the [Baidu cloud link](https://pan.baidu.com/s/1gMQOB2Zs1UPj6n8qzJc4Lg?pwd=qjl2), the extraction code is 'qjl2'.

#### Download the pre-trained networks
You can download the models from the [Baidu cloud link](https://pan.baidu.com/s/10JTBM-aL_SlWGPZGsEUM5w), the extraction code is 'kb82'. Then put the model files 'SegmNet.pth.tar, SegmNet_maskInitNet.pth.tar and IoUnet.pth.tar' to the subfolder 'pytracking/networks'.

## Testing the tracker
There are the [raw resullts](https://github.com/Yang428/BROST/tree/master/RawResultsOnBenchmarks) on six datasets. 
1) Download the testing datasets Got-10k, TrackingNet, VOT2016, VOT2018, VOT2019 and VOT2020 from the following Baidu cloud links.
* [Got-10k](https://pan.baidu.com/s/1t_PvpIicHc0U9yR4upf-cA), the extraction code is '78hq'.
* [TrackingNet](https://pan.baidu.com/s/1BKtc4ndh_QrMiXF4fBB2sQ), the extraction code is '5pj8'.
* [VOT2016](https://pan.baidu.com/s/1iU88Aqq9mvv9V4ZwY4gUuw), the extraction code is '8f6w'.
* [VOT2018](https://pan.baidu.com/s/1ztAfNwahpDBDssnEYONDuw), the extraction code is 'jsgt'.
* [VOT2019](https://pan.baidu.com/s/1vf7l4sQMCxZY_fDsHkuwTA), the extraction code is '61kh'.
* [VOT2020](https://pan.baidu.com/s/16PFiEdnYQDIGh4ZDxeNB_w), the extraction code is 'kdag'.
* Or you can download almost all tracking datasets from this web [link](https://blog.csdn.net/laizi_laizi/article/details/105447947#VisDrone_77).

2) Change the following paths to you own paths.
```
Network path: pytracking/parameters/segm/default_params.py  params.segm_net_path.
Results path: pytracking/evaluation/local.py  settings.network_path, settings.results_path, dataset_path.
```
3) Run the BROST tracker on Got10k and TrackingNet datasets.
```
cd pytracking
python run_experiment.py myexperiments got10k
python packed_got10k.py
python run_experiment.py myexperiments trackingnet
python packed_trackingnet.py
```

## Evaluation on VOT16, VOT18 and VOT19 using Matlab R2016b
We provide a [VOT Matlab toolkit](https://github.com/votchallenge/toolkit-legacy) integration for the BROST tracker. There is the [tracker_BROST.m](https://github.com/Yang428/BROST/tree/master/pytracking/utils) Matlab file in the 'pytracking/utils', which can be connected with the toolkit. It uses the 'pytracking/vot_wrapper.py' script to integrate the tracker to the toolkit.

## Evaluation on VOT2020 using Python Toolkit
We provide a [VOT Python toolkit](https://github.com/votchallenge/toolkit) integration for the BROST tracker. There is the [trackers.ini](https://github.com/Yang428/BROST/tree/master/pytracking/utils) setting file in the 'pytracking/utils', which can be connected with the toolkit. It uses the 'pytracking/vot20_wrapper.py' script to integrate the tracker to the toolkit.
```
cd pytracking/workspace_vot2020
pip install git+https://github.com/votchallenge/vot-toolkit-python
vot initialize <vot2020> --workspace ./workspace_vot2020/
vot evaluate BROST
vot analysis --workspace ./workspace_vot2020/ BROST
```

## Training the networks
The segmentation network in BROST tracker is trained only on the YouTube VOS dataset. Download the VOS training dataset (2018 version) and copy the files vos-list-train.txt and vos-list-val.txt from ltr/data_specs to the training directory of the VOS dataset. Download the bounding boxes from [this link](http://data.vicos.si/alanl/d3s/rectangles.zip) and copy them to the corresponding training sequence directories.
1) Download the training dataset from [this link](https://youtube-vos.org/challenge/2018/).

2) Change the following paths to you own paths.
```
Workspace: ltr/admin/local.py  workspace_dir.
Dataset: ltr/admin/local.py  vos_dir.
```
3) Taining the segmentation network
```
cd ltr
python run_training.py segm segm_default
```
4) Taining the mask initialization network
```
cp ./BROST/ ./BROST_maskInit
cd ./BROST_maskInit/ltr
move the file 
 (./actors/segm_actor_maskInitNet.py; ./data/segm_processing_maskInitNet.py; ./train_seetings/segm/segm_default_maskInitNet.py)
 to (./actors/segm_actor.py; ./data/segm_processing.py; ./train_seetings/segm/segm_default.py) respectively.
python run_training.py segm segm_default
```

## Acknowledgement
This a modified version of [LEAST](https://github.com/Yang428/LEAST) tracker which is based on the [pytracking](https://github.com/visionml/pytracking) framework. We would like to thank the author Martin Danelljan of pytracking and the author Alan Lukežič of D3S.

## Citation
If you find this project useful in your research, please consider cite:
```BibTeX
@ARTICLE{Yijin2023,<br>
title = {Accurate and Robust Visual Tracking Using Bounding Box Refinement and Online Sample Filtering},<br>
author = {Yijin, Yang. and Xiaodong, Gu.},<br>
journal = {Signal Processing: Image Communication},<br>
volume  = {116},<br>
number = {116981},<br>
year    = {2023},<br>
doi = {10.1016/j.image.2023.116981}<br>
}
```
