import torch
import torch.utils.data

import util.const as const
import util.load as load
import util.submit as submit
import util.run_length as run_length



class Ensembler(torch.utils.data.dataset.Dataset):
    def __init__(self, pred_dirs):
        self.pred_dirs = pred_dirs

        img_names = load.list_npy_in_dir(pred_dirs[0])

        # verify that all models have predicted for the same set of images
        for pred_dir in pred_dirs:
            cur_img_names = load.list_npy_in_dir(pred_dir)
            assert set(img_names) == set(cur_img_names)

        self.img_names = img_names

        return

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img_name = self.img_names[idx]

        ensembled = np.zeros(const.img_size)

        for pred_dir in self.pred_dirs:
            pred_path = os.path.join(pred_dir, img_name + '.npy')
            pred = np.load(pred_path)
            # TODO handle the case if file not found

            ensembled = np.add(ensembled, pred)

        ensembled = np.divide(ensembled, len(self.pred_dirs))

        # generate image mask
        img_mask = np.zeros(ensembled.shape)
        img_mask[ensembled > 0.5] = 1

        rle = run_length.encode(img_mask)

        return img_name, rle


def get_ensembler_loader(exp_names):

    pred_dirs = [ submit.get_pred_dir(exp_name) for exp_name in exp_names ]

    dataset = Ensembler(
        pred_dirs
    )

    loader = torch.utils.data.dataloader.DataLoader(
                                dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                            )
    return loader
