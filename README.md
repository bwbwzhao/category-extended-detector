# Towards a category-extended object detector with limited data

>Object detectors are typically learned on fully-annotated training data with fixed predefined categories. However, categories are often required to be increased progressively. Usually, only the original training set annotated with old classes and some new training data labeled with new classes are available in such scenarios. Based on the limited datasets, a unified detector that can handle all categories is strongly needed. We propose a practical scheme to achieve it in this work. A conflict-free loss is designed to avoid label ambiguity, leading to an acceptable detector in one training round. To further improve performance, we propose a retraining phase in which Monte Carlo Dropout is employed to calculate the localization confidence to mine more accurate bounding boxes, and an overlap-weighted method is proposed for making better use of pseudo annotations during retraining. Extensive experiments demonstrate the effectiveness of our method.

## Requirements
The requirements are exactly the same as mmdetection [v2.0.0](https://github.com/open-mmlab/mmdetection/tree/v2.0.0). We tested on on the following settings:

- Python 3.6.8
- CUDA 10.1
- GCC 7.3
- PyTorch 1.5.0+cu101
- TorchVision 0.6.0+cu101
- NumPy 1.16.1

To install this repository:
```
# create a conda virtual environment 
conda create -n mmdet python=3.6.8 -y
conda activate mmdet

# install mmcv
cd mmcv 
pip install -e .

# install mmdet
cd ..
pip install -e .
```
Users of this repo are highly recommended to read the README of [mmcv](https://github.com/open-mmlab/mmcv/tree/v0.5.9) and [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v2.0.0). 


## Prepare datasets

It is recommended to symlink the dataset root to [`./data/`](./data/). If your folder structure is different from the following one, you may need to change the corresponding paths in config files.
```
./
├── mmdet
├── mmcv
├── configs
├── requirements
├── tools
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── voc
│   │   ├── VOCdevkit
│   │       ├──VOC2007
│   ├── wider_face
│   │   ├── WIDER_train
│   │   ├── WIDER_val
│   │   ├── WIDER_test
│   ├── UOD
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
```


## Training
All config files are under [`./configs/category_extended/`](./configs/category_extended/).
```
./configs/category_extended/
│   ├── coco75_coco5
│   │   ├── retinanet_r50_fpn_1x_coco75coco5_conflictfree.py
│   │   ├── ...
│   ├── coco79_coco1
│   │   ├── ...
│   ├── coco60_voc
│   │   ├── ...
│   ├── coco_widerface
│   │   ├── ...
```

The experiments on COCO75-train and COCO5-train are used as examples to introduce the training process.

- Training with Conflict-Free Loss
```
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_conflictfree.py 8 \
        --work-dir ./work_dirs/retinanet_r50_fpn_1x_coco75coco5_conflictfree/
```
- Unlabeled Ground-Truth Mining
```
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_conflictfree.py \
        ./work_dirs/retinanet_r50_fpn_1x_coco75coco5_conflictfree/epoch_12.pth 8 \
        --prefix CLC
```
- Retraining with Pseudo Annotations
```
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_clcposweightedneg.py 8 \
        --work-dir ./work_dirs/retinanet_r50_fpn_1x_coco75coco5_clcposweightedneg/
```

Commands for other experiments are provided in [`./train.sh`](./train.sh).


## Acknowledgements
This code is based on mmdetection [v2.0.0](https://github.com/open-mmlab/mmdetection/tree/v2.0.0) and mmcv [v0.5.9](https://github.com/open-mmlab/mmcv/tree/v0.5.9).