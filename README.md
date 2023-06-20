# Installation

Our codebase is built upon [detectron2](https://github.com/facebookresearch/detectron2). You only need to install [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) following their [instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

Please note that we used initially detectron 0.2.1 in this project. Higher versions of detectron might report errors.

To remove cuda and nvidia: https://stackoverflow.com/questions/56431461/how-to-remove-cuda-completely-from-ubuntu
https://gist.github.com/ksopyla/bf74e8ce2683460d8de6e0dc389fc7f5

For torch 1.6 cuda 10.2 and detectron 0.2.1:
- create new environment using conda: conda create -n <<NAME>>
- Activate environment: conda activate <<NAME>>
- Install pytorch+cudatoolkit: conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
- Install detectron2: python -m pip install detectron2==0.2.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
- Install pandas: conda install pandas(ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/aavlachos/anaconda3/envs/newmeta/lib/python3.8/site-packages/pandas/_libs/window/aggregations.cpython-38-x86_64-linux-gnu.so), To solve:
  - sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
  - sudo apt install -y g++-11
  - Test with: strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX if GLIBCXX_3.4.29 is listed there
- conda install -c conda-forge pyyaml
- conda install -c anaconda scikit-image 
  
  Problem that may appear multiple times:

  AttributeError: module 'numpy' has no attribute 'str'.
  `np.str` was a deprecated alias for the builtin `str`. To avoid this error in existing code, use `str` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.str_` here.
Problem: AssertionError: Checkpoint detectron2://ImageNetPretrained/MSRA/R-101.pkl not found!
pip install fvcore==0.1.1.dev200512
  

  
I suggest using cuda 11.8 to be easier to train in cloud
  
  DONE!

  Can work with last version of detectron2 and last pytorch 2.0.1 - cudatoolkit 11.8:
  
  1: Set up the environment
  conda create -n newmeta
  conda activate newmeta
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia (latest pytorch 2.0.1)
  
  2:Install detectron2 0.6
  git clone https://github.com/facebookresearch/detectron2.git
  python -m pip install -e detectron2
  
  3:Download Meta Faster and other important packages
  git clone https://github.com/GuangxingHan/Meta-Faster-R-CNN
  conda install -c anaconda pandas
  conda install -c anaconda scikit-image 
  
(to fix problem with first stride: https://github.com/fanq15/FewX/issues/40)
  
(to fix problem with None type weight decay: https://github.com/facebookresearch/detectron2/issues/3964, change ./configs/fsod/Base-FSOD-C4.yaml)
  
# Data Preparation

We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [meta_faster_rcnn/data/datasets/builtin_meta_pascal_voc.py](meta_faster_rcnn/data/datasets/builtin_meta_pascal_voc.py).

  For a few datasets that meta faster rcnn natively supports, the datasets are assumed to exist in a directory called "datasets/", under the directory where you launch the program. They need to have the following directory structure:

## Expected dataset structure for Pascal VOC:
```
VOC2007/
  Annotations/
  ImageSets/
  JPEGImages/
```
```
VOC2012/
  Annotations/
  ImageSets/
  JPEGImages/
```
- You can download the splits from [Pascal VOC 2007 training/validation data + annotated test data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) and [Pascal VOC 2012 training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
  !!!Be careful, the pascal 2007 test set must be in the same folder as pascal 2007 train!!!
  
## Create Few-shot datasets

For each dataset, we additionally create few-shot versions by sampling shots for each novel category. For better comparisons, we sample multiple groups of training shots in addition to the ones provided in previous works. We include the sampling scripts we used for better reproducibility and extensibility. The few-shot dataset files can be found here. They should have the following directory structure:

Pascal VOC:
```
vocsplit/
  box_{1,2,3,5,10}shot_{category}_train.txt
  seed{1-29}/
    # shots
```

Each file contains the images for K shots for a specific category. There may be more instances in the images than K; in these cases, we randomly sample K instances.

The shots in the vocsplit directory are the same shots used by previous works. We additionally sample 29 more groups of shots for a total of 30 groups, which can be generated by using prepare_voc_few_shot.py.

See [prepare_voc_few_shot.py](datasets/pascal_voc/prepare_voc_few_shot.py) for generating the seeds yourself.
prepare_voc_few_shot.py is from [FSDet](https://github.com/ucbdrive/few-shot-object-detection) so I used it from there and copied the generated files.

Then run the scripts in ./datasets/pascal_voc step by step to generate the support images for both many-shot base classes (used during meta-training) and few-shot classes (used during few-shot fine-tuning).
python 2_gen_support_pool.py .

  
# Code Structure

    configs: Configuration files
    datasets: Dataset files (see Data Preparation for more details)
    meta_faster_rcnn
        checkpoint: Checkpoint code.
        config: Configuration code and default configurations.
        engine: Contains training and evaluation loops and hooks.
        layers: Implementations of different layers used in models.
        modeling: Code for models, including backbones, proposal networks, and prediction heads.
    tools
        train_net.py: Training script.
        test_net.py: Testing script.


  
# Model training and evaluation on PASCAL VOC

  Detectron2 configs: https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py

  
  To test model training:
  python3 fsod_train_net.py --num-gpus 1 --dist-url auto --config-file configs/fsod/meta_training_pascalvoc_split1_resnet101_stage_1.yaml

- We evaluate our model on the three splits as [TFA](https://github.com/ucbdrive/few-shot-object-detection).
- Similar as MSCOCO, we have three training stages, and three training steps during meta-training. 
- The training scripts for VOC split1 is 
```
sh scripts/meta_training_pascalvoc_split1_resnet101_multi_stages.sh
sh scripts/faster_rcnn_with_fpn_pascalvoc_split1_base_classes_branch.sh
sh scripts/few_shot_finetune_pascalvoc_split1_resnet101.sh
```
- The training scripts for VOC split2 is 
```
sh scripts/meta_training_pascalvoc_split2_resnet101_multi_stages.sh
sh scripts/faster_rcnn_with_fpn_pascalvoc_split2_base_classes_branch.sh
sh scripts/few_shot_finetune_pascalvoc_split2_resnet101.sh
```
- The training scripts for VOC split3 is 
```
sh scripts/meta_training_pascalvoc_split3_resnet101_multi_stages.sh
sh scripts/faster_rcnn_with_fpn_pascalvoc_split3_base_classes_branch.sh
sh scripts/few_shot_finetune_pascalvoc_split3_resnet101.sh
```

## Model Zoo 

We provided the meta-trained models over base classes for both MSCOCO dataset and the 3 splits on VOC dataset. The model links are [Google Drive](https://drive.google.com/drive/u/0/folders/11ODEuV1iaKRZp_XQgEfnuwmIK00FIv1S) and [Tencent Weiyun](https://share.weiyun.com/PeBdgBLY).

