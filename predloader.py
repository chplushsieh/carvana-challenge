import torch
import torch.utils.data

import util.const as const
import util.load as load

__all__ = [
    'get_small_loader',

    'get_train_loader',
    'get_val_loader',
    'get_test_loader',
]


class Ensembler(torch.utils.data.dataset.Dataset):
    def __init__(self, pred_dirs):
        self.pred_dirs = pred_dirs

        # TODO verify img_names in pred_dirs
        self.data_files = load.list_img_in_dir(pred_dirs[0])

        return

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):

        img_name = self.data_files[idx]
        target = load.load_train_mask(img_name)

        return img_name, img, target


def get_ensembler_loader():
    print('Number of Images:', len(car_ids))

    dataset = Ensembler(
        train_dir
    )
    tile_borders = dataset.get_tile_borders()

    loader = torch.utils.data.dataloader.DataLoader(
                                dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                            )
    return loader
