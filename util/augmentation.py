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
        vshift, hshift = 70, 15
        scale_size = 92/100

        funcs = [
                    ("nothing",   None,                                 None),
                    ("color",     lambda x: color_enable(x),            None),
                    ("fancy_pca", lambda x: fancy_pca_enable(x),        None),
                    ("hflip",     lambda x: hflip(x),                   lambda x: hflip(x)),
                    ("shift",     lambda x: shift(x , hshift, vshift),  lambda x: shift(x, -hshift, -vshift)),
                    ("scale",     lambda x: scale_enable(x, scale_size),lambda x: scale_enable(x,1/scale_size))
        ]

    return funcs

def color_enable(img):
    '''
    input:
      img: (3, height, width)
    '''
    #print(img.shape)
    img = np.moveaxis(img, 0, 2)
    # img.shape: (height, width, 3)

    img = color.transform(img)

    img = np.moveaxis(img, 2, 0)
    # img.shape: (3, height, width)

    return img

def fancy_pca_enable(img):
    #print(img.shape)
    img = np.moveaxis(img, 0, 2)
    # img.shape: (height, width, 3)

    img = fancy_pca.rgb_shift(img)

    img = np.moveaxis(img, 2, 0)
    # img.shape: (3, height, width)

    return img


def hflip(img):
    if img.shape == (3, 1280, 1918):
        img = np.swapaxes(img, 0, 2)  # img.shape: (width, height, num of channels)
        img = np.flipud(img).copy()
        img = np.swapaxes(img, 0, 2)
    else:
        assert img.shape == (1280,1918)
        img = np.swapaxes(img, 0, 1)
        img = np.flipud(img).copy()
        img = np.swapaxes(img, 0, 1)

    return img

def shift(img,hshift, vshift):
    if img.shape == (3,1280,1918):
        img = np.roll(img, hshift, axis=2).copy()
        img = np.roll(img, vshift, axis=1).copy()

    else:
        img = np.roll(img, hshift, axis=1).copy()
        img = np.roll(img, vshift, axis=0).copy()

    return img

def scale_enable(img, scale_size):
    if img.shape == (3,1280,1918):
        img = scale.resize_TTA(img, scale_size).copy()

    else:
        img = np.expand_dims(img, axis=0)
        img = scale.resize_TTA(img, scale_size).copy()
        #img = np.squeeze(img, axis=0)

    return img
