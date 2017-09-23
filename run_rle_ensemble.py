import time
import argparse

from scipy import stats
import numpy as np

import util.ensemble as ensemble
import util.submit as submit
import util.const as const
import util.run_length as run_length
import util.get_time as get_time

def load_submissions(pred_dirs):

    submissions = []
    for pred_dir in pred_dirs:
        exp_names, test_time_aug_names = ensemble.get_models_ensembled(pred_dir)
        print('The predictions in {} are predicted by {}. '.format(pred_dir, list(zip(exp_names, test_time_aug_names))))

        rles = submit.load_predictions(pred_dir)
        submissions.append(rles)

    return submissions

def rle_ensemble(submissions):

    ensembled_rles = {}
    img_names = list(submissions[0].keys())
    for i, img_name in enumerate(img_names):
        iter_start = time.time()
        masks = np.zeros((len(submissions), const.img_size[0], const.img_size[1]))

        for i, submission in enumerate(submissions):

            rle = submission[img_name]
            mask = run_length.decode(rle)
            masks[i] = mask

        ensembled_mask, _ = stats.mode(masks)
        ensembled_rle = run_length.encode(ensembled_mask)
        ensembled_rles[img_name] = ensembled_rle
        print('{}/{} : {:.2f} sec '.format(i, len(img_names), time.time() - iter_start))
    return ensembled_rles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_dirs', nargs='+')
    args = parser.parse_args()

    pred_dirs = args.pred_dirs
    output_dir = get_time.get_current_time()

    submissions = load_submissions(pred_dirs)  # TODO load weightes from models.txt as well
    ensembled_rles = rle_ensemble(submissions)

    # save into submission.csv
    submit.save_predictions(output_dir, ensembled_rles)
