import torch
import torch.utils.data

import os

import numpy as np

import util.const as const
import util.load as load
import util.submit as submit
import util.run_length as run_length


class RLErunner(torch.utils.data.dataset.Dataset):
    def __init__(self, pred_dir):
        self.pred_dir = pred_dir

        pred_dir_path = os.path.join(const.OUTPUT_DIR, self.pred_dir, const.PROBS_DIR_NAME)
        self.img_names = load.list_npy_in_dir(pred_dir_path)
        return

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img_name = self.img_names[idx]

        pred_path = os.path.join(const.OUTPUT_DIR, self.pred_dir, const.PROBS_DIR_NAME, img_name + '.npy')
        img_prob = np.load(pred_path)

        # generate image mask
        img_mask = np.zeros(img_prob.shape)
        img_mask[img_prob > 50] = 1
        # prob maps are saved in int8 with values ranging from 0 to 100
        # The threshold for image mask is 50 instead of 0.5

        rle = run_length.encode(img_mask)
        return img_name, rle


def get_rle_loader(pred_dir):

    dataset = RLErunner(pred_dir)

    loader = torch.utils.data.dataloader.DataLoader(
                                dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                            )
    return loader
