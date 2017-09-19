# Run-Length Encode and Decode

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib.pyplot as plt


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


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# Time Test
masks = pd.read_csv('./output/ensemble/submission.csv')
#masks = pd.read_csv('./output/PeterUnet3DUC/submission.csv')
num_masks = masks.shape[0]
print('Total masks to encode/decode =', num_masks)

time_enc = 0.0  # seconds
time_dec = 0.0  # seconds

for r in masks.itertuples():
    t0 = time.clock()
    mask = rle_decode(r.rle_mask, (1280, 1918))
    #plt.imshow(mask)
    #plt.show()
    time_dec += time.clock() - t0
    t0 = time.clock()
    mask_rle = rle_encode(mask)
    time_enc += time.clock() - t0
    # assert (mask_rle == r.rle_mask)

print('Time full encoding = {:.4f} ms per mask'.format(1000 * time_enc / num_masks))
print('Time full decoding = {:.4f} ms per mask'.format(1000 * time_dec / num_masks))