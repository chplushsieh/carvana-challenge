import os.path

DATA_DIR = './data'

TRAIN_DIR = os.path.join(DATA_DIR, 'train_hq')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train_masks')
TEST_DIR = os.path.join(DATA_DIR, 'test_hq')

TRAIN_IMAGESET_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_IMAGESET_PATH   = os.path.join(DATA_DIR, 'val.csv')

OUTPUT_DIR = './output'

ENSEMBLE_DIR_NAME = 'ensemble'
ENSEMBLE_DIR = os.path.join(OUTPUT_DIR, ENSEMBLE_DIR_NAME)
ENSEMBLED_MODELS_PATH = os.path.join(ENSEMBLE_DIR, 'models_ensembled.txt')
ENSEMBLE_PROB_DIR = os.path.join(ENSEMBLE_DIR, 'probs')


# Images are of size 1918 * 1280
img_size = (1280, 1918) # (height, width)

# 1918 = 2 * 7 * 137
# 1920 = 2^7 * 3 * 5
# 1280 = 2^8 * 5
