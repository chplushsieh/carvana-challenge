import os.path

DATA_DIR = './data'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train_masks')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMAGESET_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_IMAGESET_PATH   = os.path.join(DATA_DIR, 'val.csv')

# Images are of size 1918 * 1280
img_size = (1280, 1918) # (height, width)

# 1918 = 2 * 7 * 137
# 1920 = 2^7 * 3 * 5
# 1280 = 2^8 * 5
