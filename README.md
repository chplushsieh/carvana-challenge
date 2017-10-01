# carvana-challenge
Out of 737 teams, our solution for [Carvana Image Masking Challenge on Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge) ranked 9th place (top 1.2%) on Public Leaderboard and 31th place (top 4.2%) on Private Leaderboard. It is made by [Chia-Hao Hsieh](https://github.com/chplushsieh) and [Shao-Wen Lai](https://github.com/judichunt).

### Problem

![Problem: remove background from car image](https://github.com/chplushsieh/carvana-challenge/blob/master/figures/problem.png "Problem")

> In this competition, you’re challenged to develop an algorithm that automatically removes the photo studio background. 

### Train Data

![Train Data contains 5088 pairs of train image and label. ](https://github.com/chplushsieh/carvana-challenge/blob/master/figures/data.png "Train Data")

### Test Data

There are 100064 test images.

## Solution Overview

Our solution is an ensemble of 5 modified U-Net models using 1280x1280 image patch as input, along with test time augmentation. We used a combination loss function of soft DICE loss and Binary Cross Entropy loss. During train, we used data augmentations, including flipping, shifting, scaling, HSV color augmentation, and fancy PCA.

Training of one single model takes about 45-50 hours on a single GPU P5000 machine. Testing takes about 6-8 hours.

### Our best performing single model

(https://github.com/chplushsieh/carvana-challenge/blob/master/figures/U-net%20Structure.png)
(https://github.com/chplushsieh/carvana-challenge/blob/master/figures/Blocks.png)


## Result

TODO

## Requirements
* python 3.6
* numpy
* pytorch
* pandas
* pyyaml
* [crayon](https://github.com/torrvision/crayon)
* scikit-image
* [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)
    * `pip install cython` and then `pip install pydensecrf`
    

## Usage

### Train/Test
1. Extract data downloaded from Kaggle to `./data`:

   ```
   data
   ├── metadata.csv
   ├── sample_submission.csv
   ├── test_hq
   │   ├── 0004d4463b50_01.jpg
   │   ├── 0004d4463b50_02.jpg
           ...
   │   └── 846faa0eb79f_04.jpg
   ├── train_hq
   │   ├── 00087a6bd4dc_01.jpg
   │   ├── 00087a6bd4dc_02.jpg
           ...
   │   └── fff9b3a5373f_16.jpg
   ├── train_masks
   │   ├── 00087a6bd4dc_01_mask.gif
   │   ├── 00087a6bd4dc_02_mask.gif
           ...
   │   └── fff9b3a5373f_16_mask.gif
   └── train_masks.csv
   ```

   Images are all of size 1918 x 1280    

2. Before training, start `crayon` by running `docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon`

3. Run `python train.py`

4. Run `python test.py <experiment_name>`

   For example, running `python test.py PeterUnet3_dropout`

   :warning: Before you run `test.py` the first time, make sure you have at least `250GB` free disk space to save prediction results.

5. [Optional] Run `python run_ensemble.py --pred_dirs <exp_output_dir_1> <exp_output_dir_2> ... <exp_output_dir_n>`

   For example, run `python run_ensemble.py --pred_dirs 0921-05:59:53 0921-06:00:00 0921-06:00:05` to ensemble three predictions

6. Run `python run_rle.py <exp_output_dir>` to generate submission at `./output/<exp_output_dir>/submission.csv`

7. [Optional] Run `python run_rle_ensemble.py --pred_dirs <exp_output_dir_1> <exp_output_dir_2> ... <exp_output_dir_n>` to ensemble run-length encoded submission.csv files.

   For example, run `python run_rle_ensemble.py --pred_dirs 0923-05:59:53 0921-06:00:00` to ensemble two predictions

### Other Scripts

* To find numbers that are divislbe by `2^n`, run `python scripts/divisble.py <start_number> <end_number>`

   For instance, `python scripts/divisble.py 900 1300`

## To-dos

- [x] load data
- [x] train/val split
- [x] prepare small dataset
- [x] implement run length encoding
- [x] add experiment config loading
- [x] try a simple UNet
- [x] visualize groundtruth and prediction
- [x] add DICE validation
- [x] add data augmentation: random horizontal flipping
- [x] add data augmentation: padding
- [x] try the original UNet
- [x] train/predict-by-tile
- [x] add DICE score during training
- [x] add validation loss and DICE score during training
- [x] add optimizer and loss to experiment setting in .yml
- [x] try modified UNet with UpSampling layers
- [x] improve tile.py: make it able to cut image into halves
- [x] complete util/tile.py: stitch_predictions()
- [x] complete util/submit.py
- [x] add data augmentation: random shift
- [x] add boundary weighted loss
- [ ] experimenting with UNet parameters and architectures/modules
    - [ ] larger UNets
    - [ ] different UNet mos
    - [ ] try Adam or other optimizers
    - [ ] try different losses
- [x] add CRF
    - it didn't help
- [ ] try a memory-efficient DenseNet
