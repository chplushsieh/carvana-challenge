
import numpy as np

import util.load as load

DATA_DIR = './data'
train_dir = os.path.join(DATA_DIR, 'train')

ids = load.list_img_in_dir(train_dir)
np.random.shuffle(ids)

num_imgs = len(ids)
print("There are {} images in {}. ".format(num_imgs, train_dir))
num_val = int(num_imgs / 5)

val_ids   = ids[:num_val]
train_ids = ids[num_val:]

print("{} images for Training and {} images for Validation".format(len(train_ids), len(val_ids)))

# TODO save both into data/imageset/train.csv and data/imageset/val.csv
