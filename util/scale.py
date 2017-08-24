import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2



def resize_image(im, sz):
  h = im.shape[0]
  w = im.shape[1]
  I_out = np.zeros((h, w, 3), dtype = np.float);
  I = cv2.resize(im, None, None, fx = np.float(sz), fy = np.float(sz), interpolation=cv2.INTER_LINEAR);
  h_out = min(im.shape[0],I.shape[0])
  w_out = min(im.shape[1],I.shape[1])
  out_start=(int((h-h_out)/2), int((w-w_out)/2))
  in_start=(int((I.shape[0]-h_out)/2), int((I.shape[1]-w_out)/2))
  I_out[out_start[0]:out_start[0]+h_out, out_start[1]:out_start[1]+w_out,:] = I[in_start[0]:in_start[0]+h_out, in_start[1]:in_start[1]+w_out,:];
  return I_out.astype('uint8')
