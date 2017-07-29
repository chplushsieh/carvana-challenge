import os.path

DATA_DIR = './data'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train_masks')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMAGESET_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_IMAGESET_PATH   = os.path.join(DATA_DIR, 'val.csv')
