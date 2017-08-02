# carvana-challenge
My Solution for Carvana Image Masking Challenge on Kaggle: https://www.kaggle.com/c/carvana-image-masking-challenge

## Requirements
* python 3.6
* pytorch
* pyyaml

## Usage

1. Extract data downloaded from Kaggle to `./data`:

```
data
├── metadata.csv
├── sample_submission.csv
├── test
│   ├── 0004d4463b50_01.jpg
│   ├── 0004d4463b50_02.jpg
        ...
│   └── 846faa0eb79f_04.jpg
├── train
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

2. Run `python train.py`

3. [Not yet supported] Run `python test.py`

## To-dos

- [x] load data
- [x] train/val split
- [x] prepare small dataset
- [x] implement run length encoding
- [x] add experiment config loading
- [x] try a simple UNet
- [ ] visualize groundtruth and prediction
- [x] add DICE validation
- [x] add data augmentation: random horizontal flipping
- [ ] add data augmentation: padding
- [ ] try a standard UNet
- [ ] make first submission (complete test.py)
- [ ] hold the entire dataset in memory for train/val
- [ ] experimenting with UNet parameters and architectures
- [ ] try a memory-efficient DenseNet
