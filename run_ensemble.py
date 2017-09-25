import time
import argparse

import util.ensemble as ensemble
import util.submit as submit
import util.const as const

import ensemble_loader

def apply_ensemble(ensemble_loader):

    iter_timer = time.time()

    for i, (img_name, img_prob) in enumerate(ensemble_loader):

        assert len(img_name) == 1
        assert len(img_prob) == 1

        img_name = img_name[0]
        img_name = img_prob[0]

        if (i % 1000) == 0:
            print('Iter {} / {}, time spent since last logging: {} sec'.format(i, len(ensemble_loader), time.time() - iter_timer))
            iter_timer = time.time()

    return


if __name__ == "__main__":
    program_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_dirs', nargs='+')
    args = parser.parse_args()

    pred_dirs = args.pred_dirs

    for pred_dir in pred_dirs:
        exp_names, test_time_aug_names = ensemble.get_models_ensembled(pred_dir)
        print('The predictions in {} are predicted by {}. '.format(pred_dir, list(zip(exp_names, test_time_aug_names))))

    ensemble_loader = ensemble_loader.get_ensemble_loader(pred_dirs)

    apply_ensemble(ensemble_loader)
    print('Total time spent: {} sec = {} hours'.format(time.time() - program_start, (time.time() - program_start) / 3600))
