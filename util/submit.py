import os
import csv
import pandas as pd

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
    preds=pd.DataFrame(list(preds.items()), columns=['img', 'rle_mask'])
    preds['img'] = preds['img'].apply(lambda x: x+'.jpg')
    preds['rle_mask']=preds['rle_mask'].apply(lambda x: run_length.run_length_encode(x))
    preds.to_csv(save_path, index= False )

    return
