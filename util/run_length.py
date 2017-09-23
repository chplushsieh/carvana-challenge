
import numpy as np
from PIL import Image

'''
Carvana competition uses Run Length encoding to reduce the size of submission:
https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
'''

def encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

def decode(mask_rle):
    '''
    mask_rle: run-length as string formated (start length)

    Returns numpy array, 1 - mask, 0 - background

    '''
    shape = const.img_size

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
