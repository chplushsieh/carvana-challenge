import time
import argparse

from scipy import stats
import numpy as np

import util.ensemble as ensemble
import util.submit as submit
import util.const as const
import util.run_length as run_length
import util.get_time as get_time

import rle_ensemble_loader

def apply_ensemble(ensemble_loader):
    output_dir = get_time.get_current_time()
    ensembled_rles = {}

    iter_timer = time.time()
    for i, (img_name, rle) in enumerate(ensemble_loader):
        assert len(img_name) == 1
        assert len(rle) == 1

        img_name = img_name[0]
        rle = rle[0]

        ensembled_rles[img_name] = rle

        if (i % 1000) == 0:
            print('Iter {} / {}, time spent since last logging: {} sec'.format(i, len(ensemble_loader), time.time() - iter_timer))
            iter_timer = time.time()

    # save into submission.csv
    submit.save_predictions(output_dir, ensembled_rles)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_dirs', nargs='+')
    args = parser.parse_args()

    pred_dirs = args.pred_dirs

    rle_ensemble_loader = rle_ensemble_loader.get_rle_ensemble_loader(pred_dirs)

    apply_ensemble(rle_ensemble_loader)
