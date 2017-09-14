import numpy as np

import time
import argparse

import util.exp as exp
import util.evaluation as evaluation
import util.visualization as viz
import util.submit as submit


def ensemble(ensembler_loader):
    img_rles = {}

    for i, (img_name, rle) in enumerate(ensembler_loader):
        img_rles[img_name] = rle

    # create ./output/ensemble/ folder
    ensemble_dir = os.path.join(const.OUTPUT_DIR, 'ensemble')
    exp.create_if_not_exist(ensemble_dir)

    # save submission.csv
    submit.save_predictions('ensemble', img_rles)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--exps', nargs='+', help='<Required> experiments of which results will be ensembled', required=True)

    # Usage:
    # python ensemble.py -e PeterUnet3
    # python ensemble.py -e PeterUnet3 PeterUnet4
    # python ensemble.py -e PeterUnet3 PeterUnet4 DenseUnet

    args = parser.parse_args()
    exp_names = args.exps

    print('The predictions made by {} will be ensembled. '.format(exp_names))

    ensembler_loader = get_ensembler_loader(exp_names)

    ensemble(ensembler_loader)
