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
        vshift, hshift = randrange(-120, 120), randrange(-25, 25)
        scale_size = randrange(90, 110) / 100
        funcs = [
                    ("nothing",   None,                   None),
                    ("color",     lambda x:color_enable(x),      None),
                    ("fancy_pca", lambda x: fancy_pca_enable(x), None),
                    ("shift",     lambda x: shift(x,hshift, vshift),     lambda x: shift(x,-hshift, -vshift)),
                    ("scale",     lambda x: scale_enable(x,scale_size),     lambda x: scale_enable(x,1/scale_size))
        ]

    return funcs

def color_enable(img):
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

def fancy_pca_enable(img):
    img = np.swapaxes(img, 0, 2)
    # img.shape: (height, width, 3)

    img = fancy_pca.rgb_shift(img)

    img = np.swapaxes(img, 0, 2)
    # img.shape: (height, width, 3)

    return img

def shift(img,hshift, vshift):
    img = np.roll(img, hshift, axis=2).copy()

    img = np.roll(img, vshift, axis=1).copy()

    return img

def scale_enable(img, scale_size):
    img = scale.resize_TTA(img, scale_size).copy()


    return img

