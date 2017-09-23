import time
import argparse

import util.ensemble as ensemble
import util.submit as submit
import util.const as const
import util.run_length
import util.get_time as get_time

def load_submissions(pred_dirs):

    submissions = []
    for pred_dir in pred_dirs:
        exp_names, test_time_aug_names = ensemble.get_models_ensembled(pred_dir)
        print('The predictions in {} are predicted by {}. '.format(pred_dir, list(zip(exp_names, test_time_aug_names))))

        rles = None # TODO
        submissions.append(rles)

    return submissions

def rle_ensemble(submissions):

    ensembled_rles = {}
    img_names = None # TODO
    for img_name in img_names:
        masks = np.zeros()

        for i, submission in enumerate(submissions):

            rle = submission[img_name]
            mask = run_length.decode(rle)
            masks[i] = mask

        ensembled_mask = None # TODO average of masks
        ensembled_rle = run_length.encode(ensembled_mask)
        ensembled_rles[img_name] = ensembled_rle
    return ensembled_rles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_dirs', nargs='+')
    args = parser.parse_args()

    pred_dirs = args.pred_dirs
    output_dir = get_time.get_current_time()

    submissions = load_submissions(pred_dirs)
    ensembled_rles = rle_ensemble(submissions)

    # save into submission.csv
    submit.save_predictions(output_dir, ensembled_rles)
    return
