# pytorch-superpoint

This is a PyTorch implementation of  "SuperPoint: Self-Supervised Interest Point Detection and Description." Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. [ArXiv 2018](https://arxiv.org/abs/1712.07629).
This code is partially based on the tensorflow implementation
https://github.com/rpautrat/SuperPoint.

Please be generous to star this repo if it helps your research.
This repo is a bi-product of the paper [deepFEPE(IROS 2020)](https://github.com/eric-yyjau/pytorch-deepFEPE.git).

* Note: this repo is based on the work of eric-yyjau: https://github.com/eric-yyjau/pytorch-superpoint

## Installation
### Requirements
- python == 3.6
- pytorch >= 1.1 (tested in 1.3.1)
- torchvision >= 0.3.0 (tested in 0.4.2)
- cuda (tested in cuda10)

```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install -r requirements_torch.txt # install pytorch
```

### Path setting
- paths for datasets ($DATA_DIR), logs are set in `setting.py`

### Dataset
Datasets should be downloaded into $DATA_DIR. The folder structure for [SimCol dataset](https://arxiv.org/abs/2204.04968) should look like (use SyntheticColon_I as example):
```
datasets/ ($DATA_DIR)
`-- SyntheticColon_I (accumulated folders from raw data)
|   |-- Frames_S1
|   |   |-- Depth_0000.png/
|   |   |-- ...
|   `-- Frames_S2
|   |   |-- Depth_0000.png/
|   |   |-- ...
...
|   |-- Frames_S14
|   |   |-- Depth_0000.png/
|   |   |-- ...
|   |SavedPosition_S1.txt
|   |SavedPosition_S2.txt
...
|   |SavedPosition_S14.txt
|   |SavedRotationQuaternion_S1.txt
|   |SavedRotationQuaternion_S2.txt
...
|   |SavedRotationQuaternion_S14.txt
|   |train.txt
|   |val.txt

```
- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- SimCol
    - [SimCol link (Permission needed)](https://www.synapse.org/#!Synapse:syn28548633/wiki/617130)



## run the code
- Notes:
    - Start from any steps (1-4) by downloading some intermediate results
    - Training usually takes 8-10 hours on one 'NVIDIA 2080Ti'.
    - Currently Support training on 'COCO' dataset (original paper), 'SimCol' dataset.
- Tensorboard:
    - log files is saved under 'runs/<\export_task>/...'
    
`tensorboard --logdir=./runs/ [--host | static_ip_address] [--port | 6008]`

### 1) Training MagicPoint on Synthetic Shapes
```
python train4.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
```
you don't need to download synthetic data. You will generate it when first running it.
Synthetic data is exported in `./datasets`. You can change the setting in `settings.py`.

### 2) Exporting detections on MS-COCO / SimCol
This is the step of homography adaptation(HA) to export pseudo ground truth for joint training.
- make sure the pretrained model in config file is correct
- make sure COCO dataset is in '$DATA_DIR' (defined in setting.py)
<!-- - you can export hpatches or coco dataset by editing the 'task' in config file -->
- config file:
```
export_folder: <'train' | 'val'>  # set export for training or validation
```
#### General command:
```
python export.py <export task>  <config file>  <export folder> [--outputImg | output images for visualization (space inefficient)]
```
#### export coco - do on training set 
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### export coco - do on validation set 
- Edit 'export_folder' to 'val' in 'magicpoint_coco_export.yaml'
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### export SimCol
- config
  - check the 'root' in config file 
  - train/ val split_files are included in ($DATA_DIR).
```
python export.py export_detector_homoAdapt configs/magicpoint_SyntheticColon1_export.yaml magicpoint_base_SyntheticColon1F1
```
<!-- #### export tum
- config
  - check the 'root' in config file
  - set 'datasets/tum_split/train.txt' as the sequences you have
```
python export.py export_detector_homoAdapt configs/magicpoint_tum_export.yaml magicpoint_base_homoAdapt_tum
``` -->


### 3) Training Superpoint on MS-COCO/ SimCol
You need pseudo ground truth labels to traing detectors. Labels can be exported from step 2) or downloaded from [link](https://drive.google.com/drive/folders/1nnn0UbNMFF45nov90PJNnubDyinm2f26?usp=sharing). Then, as usual, you need to set config file before training.
- config file
  - root: specify your labels root
  - root_split_txt: where you put the train.txt/ val.txt split files (no need for COCO, needed for SimCol)
  - labels: the exported labels from homography adaptation
  - pretrained: specify the pretrained model (you can train from scratch)
- 'eval': turn on the evaluation during training 

#### General command
```
python train4.py <train task> <config file> <export folder> --eval
```

#### COCO
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```
#### SimCol
```
python train4.py train_joint configs/superpoint_synth_train_heatmap.yaml superpoint_synth1 --eval --debug
```

- set your batch size (originally 1)
- refer to: 'train_tutorial.md'


- specify the pretrained model

## Pretrained models
### Current best model
- *COCO dataset*
```logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar```
- *SimCol*
```pretrained\superPointNet_114000_checkpoint.pth.tar```
### model from magicleap
```pretrained/superpoint_v1.pth```

## Citations
Please cite the original paper.
```
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={224--236},
  year={2018}
}
```

Please also cite our DeepFEPE paper.
```
@misc{2020_jau_zhu_deepFEPE,
Author = {You-Yi Jau and Rui Zhu and Hao Su and Manmohan Chandraker},
Title = {Deep Keypoint-Based Camera Pose Estimation with Geometric Constraints},
Year = {2020},
Eprint = {arXiv:2007.15122},
}
```

# Credits
This implementation is developed by [You-Yi Jau](https://github.com/eric-yyjau) and [Rui Zhu](https://github.com/Jerrypiglet). Please contact You-Yi for any problems. 
Again the work is based on Tensorflow implementation by [Rémi Pautrat](https://github.com/rpautrat) and [Paul-Edouard Sarlin](https://github.com/Skydes) and official [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
Thanks to Daniel DeTone for help during the implementation.

## Posts
[What have I learned from the implementation of deep learning paper?](https://medium.com/@eric.yyjau/what-have-i-learned-from-the-implementation-of-deep-learning-paper-365ee3253a89)
