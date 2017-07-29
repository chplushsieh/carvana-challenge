
import os
import numpy as np
import csv
import os.path
from os import listdir
from os.path import isfile, join
from PIL import Image

def load_img(data_dir, img_name, img_ext):
    img_path = os.path.join(data_dir, img_name + '.' + img_ext)
    img = np.asarray(Image.open(img_path)) # img.shape: (height, width, 3)
    img = np.moveaxis(img, 2, 0) # img.shape: (3, height, width)
    # print('Image {} Shape: {}'.format(img_name, img.shape))
    return img

def load_train(dir, img_names):
    return load_data(dir, img_names, 'jpg')

def load_train_mask(dir, img_names):
    return load_data(dir, img_names, 'gif')

def load_data(data_dir, img_names, img_ext):

    first_img_path = os.path.join(data_dir, img_names[0] + '.' + img_ext)
    img_height, img_width= get_img_shape(first_img_path)
    imgs = np.zeros((len(img_names), img_height, img_width))

    for i, img_name in numerate(img_names):
        img = load_img(data_dir, img_name, img_ext)
        imgs[i] = img

    return imgs

def get_filename(path):
    base = os.path.basename(path)
    filename, ext = os.path.splitext(base)[0], os.path.splitext(base)[1]
    return filename, ext

def list_img_in_dir(dir):
    onlyfiles = [ f for f in listdir(dir) if isfile(join(dir, f))]
    onlyjpgs = [os.path.splitext(f)[0] for f in onlyfiles if os.path.splitext(f)[1] == '.jpg']
    return onlyjpgs

def list_csv_in_dir(dir):
    onlyfiles = [ f for f in listdir(dir) if isfile(join(dir, f))]
    onlyjpgs = [os.path.splitext(f)[0] for f in onlyfiles if os.path.splitext(f)[1] == '.csv']
    return onlyjpgs

def get_img_shape(image_path):
    im = Image.open(image_path)
    width, height =  im.size
    return (height, width)
