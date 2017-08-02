import torch
import torch.utils.data

import os
import os.path
import random
import numpy as np
from PIL import Image

import util.const as const
import util.load as load


__all__ = [
    'get_small_loader',

    'get_train_loader',
    'get_val_loader',
    'get_test_loader',
]


class LargeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, ids=None, mask_dir=None, transform=None):
        self.data_dir = data_dir

        if not ids:
            self.data_files = load.list_img_in_dir(data_dir)
        else:
            self.data_files = ids

        self.mask_dir = mask_dir
        self.transform = transform

        return

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):

        img_name = self.data_files[idx]

        img = load.load_train_image(self.data_dir, img_name, transform=self.transform)

        if self.is_test():
            target = load.load_train_mask(self.mask_dir, img_name, transform=self.transform)
        else:
            target = None

        return img_name, img, target

    def is_test(self):
        return (self.mask_dir is None)


def get_test_loader(batch_size):
    test_dir = const.TEST_DIR

    test_ids = load.list_img_in_dir(test_dir)

    print('Number of Test Images:', len(test_ids))

    transformations = transforms.Compose([
        transforms.Pad(padding=[1, 0, 1, 0], fill=0),
    ])  # No random flipping for inference

    test_dataset = LargeDataset(
        test_dir,
        ids=test_ids,
        transform=transformations,
    )

    test_loader = torch.utils.data.dataloader.DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                shuffle=False, # For inference
                                num_workers=8,
                            )
    return test_loader

def get_trainval_loader(batch_size, car_ids):
    train_dir = const.TRAIN_DIR
    train_mask_dir = const.TRAIN_MASK_DIR

    print('Number of Images:', len(car_ids))

    transformations = transforms.Compose([
        transforms.Pad(padding=[1, 0, 1, 0], fill=0),
        transforms.RandomHorizontalFlip(),
    ])

    # TODO make a parent class CarDataset of LargeDataset
    # which, unlike LargeDataset, read the entire data in memory for faster access
    # use that class for trainval use
    dataset = LargeDataset(
        train_dir,
        ids=car_ids,
        mask_dir=train_mask_dir,
        transform=transformations,
    )

    loader = torch.utils.data.dataloader.DataLoader(
                                dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                            )
    return loader

def get_train_loader(batch_size):
    train_imgs = load.load_train_imageset()
    return get_trainval_loader(batch_size, train_imgs)

def get_val_loader(batch_size):
    val_imgs = load.load_val_imageset()
    return get_trainval_loader(batch_size, val_imgs)

def get_small_loader(batch_size):
    small_imgs = load.load_small_imageset()
    return get_trainval_loader(batch_size, small_imgs)
