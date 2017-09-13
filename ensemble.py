import numpy as np

import time
import argparse

import util.exp as exp
import util.evaluation as evaluation
import util.visualization as viz
import util.submit as submit

# TODO
# load predictions from multiple output/<exp_name>/predictions folders
# take average of them
# convert from prob to mask
# apply RLE
# save into submission.csv


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
