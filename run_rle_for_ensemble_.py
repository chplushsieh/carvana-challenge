import numpy as np

import time

import util.exp as exp
import util.submit as submit
import util.const as const

import rle_loader

def apply_rle(rle_loader):
    img_rles = {}

    for i, (img_name, rle) in enumerate(rle_loader):
        img_rles[img_name] = rle

    # save submission.csv
    submit.save_predictions(const.ENSEMBLE_DIR_NAME, img_rles)
    return


if __name__ == "__main__":

    exp_names = args.exps

    print('The predictions made by {} will be ensembled. '.format(exp_names))

    rle_loader = rle_loader.get_rle_loader()

    apply_rle(rle_loader)
