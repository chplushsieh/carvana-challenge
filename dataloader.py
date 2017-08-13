import torch
import torch.utils.data
from torchvision import transforms

import random

import util.const as const
import util.load as load
import util.tile as tile

__all__ = [
    'get_small_loader',

    'get_train_loader',
    'get_val_loader',
    'get_test_loader',
]


class LargeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, ids=None, mask_dir=None, hflip_enabled=False, shift_enabled=False, paddings=None, tile_size=None):
        self.data_dir = data_dir

        if not ids:
            self.data_files = load.list_img_in_dir(data_dir)
        else:
            self.data_files = ids

        if tile_size:
            img_height, img_width = const.img_size
            padded_img_size = img_height + 2 * paddings[0], img_width + 2 * paddings[1]
            self.data_files = tile.generate_tile_names(self.data_files, tile_size, padded_img_size)
            _, self.tile_borders = tile.get_tile_layout(tile_size, padded_img_size)

        self.mask_dir = mask_dir
        self.hflip_enabled = hflip_enabled
        self.shift_enabled = shift_enabled
        self.paddings = paddings
        self.tile_size = tile_size

        return

    def remove_tile_borders(self, image_with_border):
        '''
        input:
          image_with_border: a numy array of shape (num_channels, height, width)
        output:
          image: a numy array of shape (num_channels, height - 2 * tile_height_border, width - 2 * tile_width_border)
        '''
        tile_height_border, tile_width_border = self.tile_borders
        image  =  image_with_border[:, tile_height_border:-tile_height_border, tile_width_border:-tile_width_border]
        return image

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):

        img_name = self.data_files[idx]

        # decide if we will flip the image and the target
        is_hflip = self.hflip_enabled and (random.random() < 0.5)

        is_shift = self.shift_enabled and (random.random() < 0.5)

        img = load.load_train_image(self.data_dir, img_name, is_hflip=is_hflip, is_shift=is_shift, paddings=self.paddings, tile_size=self.tile_size)

        if self.is_test():
            target = -1
        else:
            target = load.load_train_mask(self.mask_dir, img_name, is_hflip=is_hflip, is_shift=is_shift, paddings=self.paddings, tile_size=self.tile_size)

        return img_name, img, target

    def is_test(self):
        return (self.mask_dir is None)


def get_test_loader(batch_size, tile_size):
    test_dir = const.TEST_DIR

    test_ids = load.list_img_in_dir(test_dir)

    print('Number of Test Images:', len(test_ids))

    test_dataset = LargeDataset(
        test_dir,
        ids=test_ids,
        hflip_enabled=False, # No random flipping for inference
        shift_enabled=False,
        tile_size=tile_size,
    )

    test_loader = torch.utils.data.dataloader.DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                shuffle=False, # For inference
                                num_workers=8,
                            )
    return test_loader

def get_trainval_loader(batch_size, car_ids, paddings, tile_size, hflip_enabled=False, shift_enabled=False):
    train_dir = const.TRAIN_DIR
    train_mask_dir = const.TRAIN_MASK_DIR

    print('Number of Images:', len(car_ids))

    # TODO make a parent class CarDataset of LargeDataset
    # which, unlike LargeDataset, read the entire data in memory for faster access
    # use that class for trainval use
    dataset = LargeDataset(
        train_dir,
        ids=car_ids,
        mask_dir=train_mask_dir,
        hflip_enabled=hflip_enabled,
        shift_enabled=shift_enabled,
        paddings=paddings,
        tile_size=tile_size,
    )

    loader = torch.utils.data.dataloader.DataLoader(
                                dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                            )
    return loader

def get_train_loader(batch_size, paddings, tile_size, hflip, shift):
    train_imgs = load.load_train_imageset()
    return get_trainval_loader(batch_size, train_imgs, paddings, tile_size, hflip_enabled=hflip, shift_enabled=shift)

def get_val_loader(batch_size, paddings, tile_size, hflip, shift):
    val_imgs = load.load_val_imageset()
    return get_trainval_loader(batch_size, val_imgs, paddings, tile_size, hflip_enabled=hflip, shift_enabled=shift)

def get_small_loader(batch_size, paddings, tile_size, hflip, shift):
    small_imgs = load.load_small_imageset()
    return get_trainval_loader(batch_size, small_imgs, paddings, tile_size, hflip_enabled=False, shift_enabled=False)
