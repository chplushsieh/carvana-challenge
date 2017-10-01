import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import os.path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import csv
import cv2

# TODO refactor (or delete)


def multi_color_aug(img):
    FrequencyNoise = iaa.FrequencyNoiseAlpha(
        exponent=(-4, 0),
        first=iaa.Multiply((0.8, 1.2), per_channel=True),
        second=iaa.ContrastNormalization((0.5, 2.0))
    )
    ContrastNorm = iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5)
    Emboss = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        sometimes(FrequencyNoise),      sometimes(ContrastNorm)]    )

    #img=np.moveaxis(img,0,2)
    img1 = np.expand_dims(img, axis=0)
    images_aug = np.squeeze(seq.augment_images(img1), axis=0)

    return images_aug




def multi_shape_aug(img, coeff):
    #shear = iaa.Affine(shear=(-12, 12))
    #piecewiseAff = iaa.PiecewiseAffine(scale=(0.01, 0.03))
    #PerspectiveTrans = iaa.PerspectiveTransform(scale=(0.01, 0.05))

    shear = iaa.Affine(shear=coeff[0])
    piecewiseAff = iaa.PiecewiseAffine(scale=coeff[1]/100)
    PerspectiveTrans = iaa.PerspectiveTransform(scale=coeff[2]/100)


    seq = iaa.Sequential([
        shear, piecewiseAff]
    )
    #seq = iaa.Sequential([
    #    shear, piecewiseAff, PerspectiveTrans]
    #)

    img=np.swapaxes(img,0,2)
    img1 = np.expand_dims(img, axis=0)
    #print(img1.shape)
    #if img1.shape ==(1, 1918, 1280, 1):
    #    seq.augment_images(img1)
    images_aug = np.squeeze(seq.augment_images(img1), axis=0)
    images_aug = np.swapaxes(images_aug, 2, 0)
    #print(images_aug.shape)

    return images_aug



if __name__ == "__main__":

    def load_imageset(imageset_path):
        img_names = []
        with open(imageset_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                img_names.append('/home/paperspace/git/carvana-challenge/data/train_hq/' + row[0] + '.jpg')

        img_names.sort()
        return img_names


    DATA_DIR = '/home/paperspace/git/carvana-challenge/data/'
    TRAIN_IMAGESET_PATH = os.path.join(DATA_DIR, 'train.csv')

    img_names = load_imageset(TRAIN_IMAGESET_PATH)

    imlist = (io.imread_collection(img_names))

    img = imlist[0]

    shear = iaa.Affine(shear=(-12, 12))
    piecewiseAff = iaa.PiecewiseAffine(scale=(0.01, 0.03))
    PerspectiveTrans = iaa.PerspectiveTransform(scale=(0.01, 0.05))
    FrequencyNoise = iaa.FrequencyNoiseAlpha(
        exponent=(-4, 0),
        first=iaa.Multiply((0.8, 1.2), per_channel=True),
        second=iaa.ContrastNormalization((0.5, 2.0))
    )
    ContrastNorm = iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
    Emboss = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        sometimes(shear), sometimes(piecewiseAff), sometimes(PerspectiveTrans), sometimes(FrequencyNoise),
        sometimes(ContrastNorm)]
    )

    img1 = np.expand_dims(img, axis=0)
    images_aug = np.squeeze(shear.augment_images(img1), axis=0)
    images_aug2 = np.squeeze(piecewiseAff.augment_images(img1), axis=0)
    images_aug3 = np.squeeze(PerspectiveTrans.augment_images(img1), axis=0)
    images_aug4 = np.squeeze(FrequencyNoise.augment_images(img1), axis=0)
    images_aug5 = np.squeeze(ContrastNorm.augment_images(img1), axis=0)
    # images_aug6 = np.squeeze(Emboss.augment_images(img1), axis=0)
    images_aug6 = np.squeeze(seq.augment_images(img1), axis=0)
    # images_aug= np.squeeze(images_aug, axis=0)

    translater = iaa.Affine(translate_px={"x": -16})  # move each input image by 16px to the left
    # images_aug2 = translater.augment_image(img)
    print(img.shape)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.imshow(images_aug)
    plt.subplot(2, 2, 3)
    plt.imshow(images_aug2)
    plt.subplot(2, 2, 4)
    plt.imshow(images_aug3)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.imshow(images_aug4)
    plt.subplot(2, 2, 3)
    plt.imshow(images_aug5)
    plt.subplot(2, 2, 4)
    plt.imshow(images_aug6)
    plt.show()
