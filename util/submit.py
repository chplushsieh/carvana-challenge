import os
import time

import pandas as pd

import util.const as const


def save_predictions(exp_name, preds):
    '''
    input:
      exp_name: a string which is the experiemnt name
      preds: a dict of strings, with image names as keys and predicted run-length-encoded masks as values
    '''
    func_start = time.time()

    save_path = os.path.join(const.OUTPUT_DIR, exp_name, 'submission.csv')

    preds=pd.DataFrame(list(preds.items()), columns=['img', 'rle_mask'])
    preds['img'] = preds['img'].apply(lambda x: x+'.jpg')
    preds.to_csv(save_path, index= False )

    func_end = time.time()
    print('{:.2f} sec spent saving into .csv file'.format(func_end - func_start))
    return
