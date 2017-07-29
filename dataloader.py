import torch

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
    def __init__(self, data_dir, ids=None):
        self._data_dir = data_dir

        if not ids:
            self.data_files = load.list_img_in_dir(data_dir)
        else:
            self.data_files = ids

        return

    def __getitem__(self, idx):

        img_name = self.data_files[idx]
        img   = load.load_img(self._data_dir, img_name, 'jpg')
        # print('img.shape:', img.shape)

        return img_name, img

    def __len__(self):
        return len(self.data_files)


def get_test_loader(batch_size):
    test_dir = const.TEST_DIR

    test_ids = load.list_img_in_dir(test_dir)

    print('Number of Test Images:', len(test_ids))

    test_dataset = LargeDataset(
        test_dir,
        ids=test_ids,
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

    print('Number of Images:', len(car_ids))

    data   = load.load_train(train_dir, car_ids)
    target = load.load_train_mask(train_mask_dir, car_ids)

    data_tensor   = torch.from_numpy(data)
    target_tensor = torch.from_numpy(target)
    dataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)

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
