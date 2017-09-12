import os
import time

import numpy as np
import pandas as pd

import util.exp as exp
import util.const as const

def save_prob_map(exp_name, img_name, img_prob):
    '''
    input:
      exp_name: a string, name of the experiemnt
      img_name: a string, name of the image
      img_prob: an numpy array of probability of each pixel being foreground(car)
    '''

    assert img_prob.shape == const.img_size  # image shape: (1280, 1918)

    save_dir = os.path.join(const.OUTPUT_DIR, exp_name, const.SAVED_PREDS_DIR_NAME)
    exp.create_if_not_exist(save_dir)

    save_path = os.path.join(save_dir, img_name + '.npy')
    np.save(save_path, img_prob)

    return


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
