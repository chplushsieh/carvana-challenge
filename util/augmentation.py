import os
import csv
import os.path
from os import listdir
from os.path import isfile, join
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

import util.const as const
import util.tile as tile
import util.color as color
import util.scale as scale
import util.fancy_pca as fancy_pca
import cv2
from random import randrange

def get_TTA_funcs(is_TTA):

    if not is_TTA:
        funcs = [ ("nothing", None, None),  ]
    else:
        # TODO add more test time augs
        funcs = [
                    ("nothing",   None,                   None),
                    ("color",     lambda x:color(x),      None),
                    ("fancy_pca", lambda x: fancy_pca(x), None),
        ]

    return funcs

def color(img):
    '''
    input:
      img: (3, height, width)
    '''
    img = np.swapaxes(img, 0, 2)
    # img.shape: (height, width, 3)

    img = color.transform(img)

    img = np.swapaxes(img, 0, 2)
    # img.shape: (height, width, 3)

    return img

def fancy_pca(img):
    img = np.swapaxes(img, 0, 2)
    # img.shape: (height, width, 3)

    img = fancy_pca.rgb_shift(img)

    img = np.swapaxes(img, 0, 2)
    # img.shape: (height, width, 3)

    return img
