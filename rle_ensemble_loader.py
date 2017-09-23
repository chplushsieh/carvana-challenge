import torch
import torch.utils.data

import os

import numpy as np

import util.const as const
import util.load as load
import util.ensemble as ensemble
import util.submit as submit
import util.get_time as get_time
import util.run_length as run_length
import util.exp as exp
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
    def __init__(self, pred_dirs):
        self.pred_dirs = pred_dirs
        self.submissions = load_submissions(pred_dirs)

        # TODO verify same number in each dir
        img_names = list(submissions[0].keys())

        # TODO self.weights = ensemble.get_ensemble_weights(self.pred_dirs)

        # create self.ensemble_dir/models_ensembled.txt
        # for pred_dir in self.pred_dirs:
        #     exp_names, test_time_aug_names = ensemble.get_models_ensembled(pred_dir)
        #
        #     for exp_name, test_time_aug_name in zip(exp_names, test_time_aug_names):
        #         ensemble.mark_model_ensembled(self.ensemble_dir, exp_name, test_time_aug_name)
        return

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img_name = self.img_names[idx]


        masks = np.zeros((len(self.submissions), const.img_size[0], const.img_size[1]))

        for i, submission in enumerate(self.submissions):

            rle = submission[img_name]
            mask = run_length.decode(rle)
            masks[i] = mask

        ensembled_mask, _ = stats.mode(masks)
        # plt.imshow(ensembled_mask)
        # plt.show()
        ensembled_rle = run_length.encode(ensembled_mask)

        return img_name, ensembled_rle


def get_ensemble_loader(pred_dirs):

    dataset = RleEnsembleRunner(pred_dirs)

    loader = torch.utils.data.dataloader.DataLoader(
                                dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                            )
    return loader
