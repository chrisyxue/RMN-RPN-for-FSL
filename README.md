# Relative Position and Map Networks in Few-shot Learning for Image Classification
Accepted in CVPR 2020 VL3 Workshop (https://www.learning-with-limited-labels.com/home)

Download the paper (http://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Xue_Relative_Position_and_Map_Networks_in_Few-Shot_Learning_for_Image_CVPRW_2020_paper.pdf)
## Citation
If you find our work useful, please consider citing our work using the bibtex:

```
@InProceedings{Xue_2020_CVPR_Workshops,
author = {Xue, Zhiyu and Xie, Zhenshan and Xing, Zheng and Duan, Lixin},
title = {Relative Position and Map Networks in Few-Shot Learning for Image Classification},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
```
## Requirements
Pytorch 1.4.0

Torchvision 0.5.0

1 GPU (e.g. NVIDIA TITAN Xp)


## Data
[Mini-Imagenet](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE): Put it in ./datas/mini-Imagenet and run proc_image.py to preprocess generate train/val/test datasets.

[CIFAR-FS](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view?usp=sharing): Put it in ./datas/cifar/, and unzip the package.

## Run
#### Mini-ImageNet(5-way 5-shot):
```
cd miniimagenet/
python main.py -w 5 -s 5
```
#### CIFAR-FS(5-way 5-shot)
```
cd CIFAR-FS/
python main.py -w 5 -s 5
```
## Reference
1. [Relation Networks](https://github.com/floodsung/LearningToCompare_FSL)
2. [R2-D2](https://github.com/bertinetto/r2d2)

Please contact xzy990228@gmail.com if you have any problem.
