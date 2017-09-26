# Run-Length Encode and Decode

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib.pyplot as plt
from util.run_length import decode as rle_decode


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)


# Time Test
masks = pd.read_csv('./output/0921-18:30:23/submission.csv')
#masks = pd.read_csv('./output/PeterUnet3DUC/submission.csv')
num_masks = masks.shape[0]
print('Total masks to encode/decode =', num_masks)

time_enc = 0.0  # seconds
time_dec = 0.0  # seconds

for r in masks.itertuples():
    t0 = time.clock()
    mask = rle_decode(r.rle_mask)
    plt.imshow(mask)
    plt.show()
    time_dec += time.clock() - t0
    t0 = time.clock()
    mask_rle = rle_encode(mask)
    time_enc += time.clock() - t0
    # assert (mask_rle == r.rle_mask)

print('Time full encoding = {:.4f} ms per mask'.format(1000 * time_enc / num_masks))
print('Time full decoding = {:.4f} ms per mask'.format(1000 * time_dec / num_masks))
