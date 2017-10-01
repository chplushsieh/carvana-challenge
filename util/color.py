import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def transform(image):
    '''
    input:
      image: numpy array of shape (channels, height, width), in RGB code
    output:
      transformed: numpy array of shape (channels, height, width), in RGB code
    '''
    transformed = image

    hue_shift_limit = (-50, 50)
    sat_shift_limit = (-5, 5)
    val_shift_limit = (-15, 15)

    if np.random.random() < 0.5:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(transformed)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        transformed = cv2.merge((h, s, v))
        transformed = cv2.cvtColor(transformed, cv2.COLOR_HSV2BGR)

    return transformed
