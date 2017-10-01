import torch
import torch.utils.data

import os
import numpy as np

import util.const as const
import util.ensemble as ensemble
import util.submit as submit
import util.run_length as run_length
import matplotlib.pyplot as plt

def load_submissions(pred_dirs):

    submissions = []
    for pred_dir in pred_dirs:
        exp_names, test_time_aug_names = ensemble.get_models_ensembled(pred_dir)
        print('The predictions in {} are predicted by {}. '.format(pred_dir, list(zip(exp_names, test_time_aug_names))))

        rles = submit.load_predictions(pred_dir)
        submissions.append(rles)

    return submissions

class RleEnsembleRunner(torch.utils.data.dataset.Dataset):
    def __init__(self, pred_dirs, ensemble_dir):
        self.pred_dirs = pred_dirs
        self.submissions = load_submissions(pred_dirs)

        self.img_names = list(self.submissions[0].keys())

        self.weights = ensemble.get_ensemble_weights(self.pred_dirs)

        self.ensemble_dir = ensemble_dir
        ensemble.create_models_ensembled(self.pred_dirs, self.ensemble_dir)
        return

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img_name = self.img_names[idx]

        ensembled_mask = np.zeros((const.img_size[0], const.img_size[1]))

        for i, submission in enumerate(self.submissions):

            rle = submission[img_name]
            mask = run_length.decode(rle)
            weighted_mask = np.multiply(mask, self.weights[i])
            ensembled_mask = np.add(ensembled_mask, weighted_mask)

        ensembled_mask[ ensembled_mask > 0.5 ] = 1
        ensembled_mask[ ensembled_mask <= 0.5 ] = 0

        # plt.imshow(ensembled_mask)
        # plt.show()

        ensembled_rle = run_length.encode(ensembled_mask)

        return img_name, ensembled_rle


def get_rle_ensemble_loader(pred_dirs, ensemble_dir):

    dataset = RleEnsembleRunner(pred_dirs, ensemble_dir)

    loader = torch.utils.data.dataloader.DataLoader(
                                dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                            )
    return loader
