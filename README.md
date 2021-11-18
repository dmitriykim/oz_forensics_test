# Facial recognition on grayscale images (32x32)

This repo is a reimplementation of Insightface([github](https://github.com/deepinsight/insightface)).

I used Arcface_torch method with ResNet50 backbone. 

Training parameters are default except Learning rate and Batch size and placed in configs/ms1mv3_r50.py.

Training logs and checkpoints are located in work_dirs/ms1mv3_r50


## Requirements

- Install CUDA 10.2
- `pip3 install -r requirements.txt`.

## Datasets preparing
Train: `python3 resize.py -i <path_to_train_set> -o datasets/casia_faces_32`

Test: `python3 resize.py -i <path_to_test_set> -o datasets/lfw_32`


## How to Training

To train a model, run `python3 train.py configs/ms1mv3_r50`.

## Evaluation on LFW

Run `python3 eval_lfw.py`.

## Evaluation results

|Backbone|Threshold|EER   |Accuracy|
|:---    |:---     |:---  |:---    |  
|r50     |1.259    |0.099 |0.901   |         

<img src="https://github.com/dmitriykim/oz_forensics_test/blob/master/metrics.png" width="760"/>