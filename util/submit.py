import os
import time

import numpy as np
import pandas as pd

import util.exp as exp
import util.const as const
import util.ensemble as ensemble

def get_pred_dir(exp_name):
    pred_dir = os.path.join(const.OUTPUT_DIR, exp_name, const.SAVED_PREDS_DIR_NAME)
    return pred_dir

def save_prob_map(ensemble_dir, img_name, img_prob):
    '''
    input:
      img_name: a string, name of the image
      img_prob: an numpy array of probability of each pixel being foreground(car)
    '''
    # func_start = time.time()

    assert img_prob.shape == const.img_size  # image shape: (1280, 1918)

    probs_dir = os.path.join(ensemble_dir, 'probs')

    exp.create_if_not_exist(probs_dir)
    save_path = os.path.join(probs_dir, img_name + '.npy')

    # convert from probability in percentage
    # ex: 0.92 -> 92(%)
    img_prob = np.multiply(img_prob, 100)

    if os.path.isfile(save_path):
        print('Warning: {} already exists'.format(save_path))

    # img_prob.dtype == np.float64
    img_prob = img_prob.astype(np.int8) # which takes 0.014 sec while casting to np.float16 takes about 0.2 sec
    # One int8 1280x1918 image takes about 2.5 MB storage
    # while One float16 1280x1918 image takes about 4.9 MB storage

    np.save(save_path, img_prob)
    # func_end = time.time()
    # print('Saving probability map takes {:.4f} sec. '.format(func_end - func_start))
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
