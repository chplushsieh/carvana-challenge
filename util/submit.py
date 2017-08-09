import os
import csv

import util.run_length as run_length
import util.const as const


def save_predictions(exp_name, preds):
    '''
    input:
      exp_name: a string which is the experiemnt name
      preds: a dict of numpy arrays, with image names as keys and predicted masks as values
    '''
    save_path = os.path.join(const.OUTPUT_DIR, exp_name, 'submission.csv')

    # TODO iterate thru preds, use run_length.run_length_encode(mask) and save into a csv file

    return
