# GAFARv2

This repository contains the code necessary to train and evaluate the work presented in 
    
    GAFAR Revisited - Exploring the Limits of Point Cloud Registration on Sparse Subsets

which is based upon our original work presented in

    GAFAR: Graph-Attention Feature-Augmentation for Registration
    A Fast and Light-weight Point Set Registration Algorithm

For the code of the original ECMR'23 publication, please check out branch [GAFARv1](https://github.com/mordecaimalignatius/GAFAR/tree/GAFARv1) and use the models in release [GAFARv1](https://github.com/mordecaimalignatius/GAFAR/releases/tag/GAFARv1).

## Training
### ModelNet40
To train a model download the sub-sampled ModelNet40 point clouds with the official training/testing split from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and unzip it to a location of your choosing.
You'll need to update the path to the hdf5 archives in the dataset config files in gafar/config/data accordingly.

To train a model for the respective experiment run e.g.:

    python3 matching.py ./config/model/GAFARv2_train_ModelNet40.json ./experiments/mnet_unseen_crop --model ./config/model/GAFARv2.json --dataset ./config/data/ModelNet40_unseen.json

Select the dataset config according to the experiment you want to run.
All training parameters available can be found in the default configuration in `matching.py`, all model parameters will be listed in the file `config.json` in the output directory of the experiment as soon as model training has started.

Validation using the same code as in training can be done with:

    python3 matching.py ./config/model/GAFARv2.json ./experiments/mnet_unseen/eval/ --model ./config/weights/GAFARv2_ModelNet40_unseen.pt --dataset ./config/data/ModelNet40_unseen.json --validate

For the remaining test cases as presented in [RGM](https://github.com/fukexue/RGM/), please select the according `ModelNet40_<*>.json` dataset configurations.

### Kitti Odometry Benchmark
To train GAFAR on [Kitti Odometry Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) please follow [PREDATOR](https://github.com/prs-eth/OverlapPredator) for dataset download and preparation.
In order to run model training adjust the dataset path in `./config/data/KittiOdometry.json` and run:

    python3 matching.py ./config/model/GAFARv2_train_KittiOdometry.json ./experiments/kitti --model ./config/model/GAFARv2.json --dataset ./config/data/KittiOdometry.json

Validation using the training code and evaluation routines can be done with:

    python3 matching.py ./config/model/GAFARv2_train_KittiOdometry.json ./experiments/kitti/eval/ --model ./config/weights/GAFARv2_KittiOdometry.pt --dataset ./config/data/KittiOdometry.json --validate

### 3DMatch Dataset
To train GAFAR on [3DMatch](https://3dmatch.cs.princeton.edu/) please follow [PREDATOR](https://github.com/prs-eth/OverlapPredator) for dataset download and preparation.
In order to run model training adjust the dataset path in `./config/data/KittiOdometry.json` and run:

    python3 matching.py ./config/model/GAFARv2_train_3DMatch.json ./experiments/3dm --model ./config/model/GAFARv2.json --dataset ./config/data/3DMatch.json

Validation using the training code and evaluation routines can be done with:

    python3 matching.py ./config/model/GAFARv2_train_3DMatch.json ./experiments/3dm/eval/ --model ./config/weights/GAFARv2_3DMatch.pt --dataset ../config/data/3DMatch.json --validate

## Testing
### ModelNet40
To recreate the results published in the paper please use the evaluation code adapted from [RGM](https://github.com/fukexue/RGM/) as follows:

    python3 evaluate_modelnet.py --cfg ./config/data/RGM_Unseen_Crop_modelnet40.yaml --model ./config/weights/GAFARv2_ModelNet40_unseen.pt --output ./experiments/mnet_unseen/eval/rgm/ --dataset path/to/ModelNet40/ --discard-bin

### Kitti Odometry Benchmark
For recreation of results on Kitti Odometry Benchmark, please run the code adapted from [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) as follows:

    python3 evaluate.py ./config/weights/GAFARv2_KittiOdometry.pt ./experiments/kitti/eval/geot/ --benchmark Kitti --dataset <path_to_kitti_dataset_root>

### 3DMatch and 3DLoMatch
For 3DMatch and 3DLoMatch, please run:

    python3 evaluate.py ./config/weights/GAFARv2_3DMatch.pt ./experiments/3dm/eval/geot/ --benchmark 3DMatch --dataset <path_to_3dmatch_dataset_root>

Select 3DLoMatch by also passing the flag `--split 3DLoMatch` as follows:

    python3 evaluate.py ./config/weights/GAFARv2_3DMatch.pt ./experiments/3dm/eval/geot/ --benchmark 3DMatch --dataset <path_to_3dmatch_dataset_root> --split 3DLoMatch

Please note that changes in batch size, different versions of packages used, differences in GPU driver or NVIDIA CUDA versions (non-deterministic execution) results may vary slightly.

## Pre-trained weights
All model weights can be found in release [GAFARv2](https://github.com/mordecaimalignatius/GAFAR/releases/tag/GAFARv2) for the `Robotics and Autonomous Systems` article and release [GAFARv1](https://github.com/mordecaimalignatius/GAFAR/releases/tag/GAFARv1) for the `ECMR'23` paper.

## Acknowledgement
In this work, parts or adaptions of the implementations of the following works are used:
* [DGCNN](https://github.com/WangYueFt/dgcnn)
* [RPM-Net](https://github.com/yewzijian/RPMNet/)
* [RGM](https://github.com/fukexue/RGM/)
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
* [LightGlue](https://github.com/cvg/LightGlue/)
* [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)

## Citation
If you use this code in your work or project, please reference:

    @inproceedings{mohr2023gafar
      title={{GAFAR: Graph-Attention Feature-Augmentation for Registration. A Fast and Light-weight Point Set Registration Algorithm}},
      author={{Mohr, Ludwig and Geles, Ismail and Fraundorfer, Friedrich}},
      booktitle={{Proceedings of the 11th European Conference on Mobile Robots ({ECMR})}},
      year={2023}
    }

and:

    @article{mohr2024gafar-revisited,
      title = {GAFAR revisited—Exploring the limits of point cloud registration on sparse subsets},
      journal = {Robotics and Autonomous Systems},
      volume = {185},
      pages = {104870},
      year = {2025},
      doi = {https://doi.org/10.1016/j.robot.2024.104870},
      url = {https://www.sciencedirect.com/science/article/pii/S0921889024002549},
      author = {Ludwig Mohr and Ismail Geles and Friedrich Fraundorfer},
    }
