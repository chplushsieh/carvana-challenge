import os.path
import sys
import csv

import numpy as np

import util.const as const
import util.load as load


def save_imageset(img_names, savepath):

    img_names.sort()

    # one image name per row
    img_names = [ [img_name] for img_name in img_names ]

    # save into csv
    with open(savepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(img_names)
    return



train_imageset_path = const.TRAIN_IMAGESET_PATH
val_imagest_path = const.VAL_IMAGESET_PATH
train_dir = const.TRAIN_DIR

if os.path.isfile(train_imageset_path) and os.path.isfile(val_imagest_path):
    print("train/val split already exists: {} and {}".format(train_imageset_path, val_imagest_path))
    sys.exit()

img_names = load.list_img_in_dir(train_dir)
car_ids = load.get_car_ids(img_names)
num_cars = len(car_ids)

num_val = int(num_cars / 5)
np.random.shuffle(car_ids)
val_ids   = car_ids[:num_val]
train_ids = car_ids[num_val:]
print("{} cars for Training and {} cars for Validation".format(len(train_ids), len(val_ids)))

train_imgs = load.get_img_names_from_car_ids(train_ids)
val_imgs   = load.get_img_names_from_car_ids(val_ids)

# save into files
save_imageset(train_imgs, train_imageset_path)
save_imageset(val_imgs, val_imagest_path)
