# GAFAR 

This repository contains the code necessary to train and evaluate the work presented in 

    GAFAR: Graph-Attention Feature-Augmentation for Registration
    A Fast and Light-weight Point Set Registration Algorithm
a well as to recreate the results stated in the paper.

## Training

To train a model download the sub-sampled ModelNet40 point clouds with the official training/testing split from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and unzip it to a location of your choosing.
You'll need to update the path to the hdf5 archives in the dataset config files in gafar/config/data accordingly.

To train a model for the respective experiment run e.g.:

    python3 matching.py ./config/model/gafar.json ./experiments/unseen_crop/ --dataset ./config/data/unseen_crop.json

Select the dataset config according to the experiment you want to run.
All training parameters available can be found in the default configuration in `matching.py`, all model parameters will be listed in the file `config.json` in the output directory of the experiment as soon as model training has started.

Validation using the same code as in training can be done with:

    python3 matching.py ./config/model/gafar.json ./experiments/unseen_crop/eval/ --model ./weights/gafar_unseen_crop.t7 --dataset ./config/data/unseen_crop.json 

## Testing

To recreate the results published in the paper please use the evaluation code adapted from [RGM](https://github.com/fukexue/RGM/) as follows:

    python3 evaluate.py --cfg ./config/data/RGM_Unseen_Crop_modelnet40.yaml --model ./config/weights/gafar_crop_unseen.pt --output ./eval/rgm/unseen/ --dataset path/to/ModelNet40/

Please note that changes in batch size, different versions of packages used, differences in GPU driver or NVIDIA CUDA versions (non-deterministic execution) results may vary slightly.

## Acknowledgement
In this work, parts or adaptions of the implementations of the following works are used:
* [DGCNN](https://github.com/WangYueFt/dgcnn)
* [RPM-Net](https://github.com/yewzijian/RPMNet/)
* [RGM](https://github.com/fukexue/RGM/)
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)

## Citation
If you use this code in your work or project, please reference:

    @inproceedings{mohr2023gafar
      title={{GAFAR: Graph-Attention Feature-Augmentation for Registration. A Fast and Light-weight Point Set Registration Algorithm}},
      author={{Mohr, Ludwig and Geles, Ismail and Fraundorfer, Friedrich}},
      booktitle={{Proceedings of the 11th European Conference on Mobile Robots ({ECMR})}},
      year={2023}
    }
