import torch

import os
import os.path
import random
import numpy as np
from PIL import Image

import util.load as load


__all__ = [
    'get_small_loader', # TODO use a small dataset for quick experiments

    'get_train_loader',
    'get_val_loader',   # TODO use a random split
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
    DATA_DIR = './data'
    test_dir = os.path.join(DATA_DIR, 'test')

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

def get_train_loader(batch_size):
    DATA_DIR = './data'
    train_dir = os.path.join(DATA_DIR, 'train')

    train_ids = load.list_img_in_dir(train_dir)

    print('Number of Train Images:', len(train_ids))

    data   = load.load_train(train_dir, train_ids)
    target = load.load_train_mask(train_mask_dir, train_ids)

    data_tensor   = torch.from_numpy(data)
    target_tensor = torch.from_numpy(target)
    train_dataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)

    train_loader = torch.utils.data.dataloader.DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                            )
    return train_loader
