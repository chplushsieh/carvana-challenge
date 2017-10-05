import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def resize_image(im, sz):
  '''
    input:
      im: numpy array of shape (num of channels, height, width)
    output:
      I_out: (num of channels, height, width)
  '''  
  im = np.swapaxes(im, 0, 2)

  h = im.shape[0]
  w = im.shape[1]
  c = im.shape[2]
  
  # identify if im is a color image or a mask
  if c == 3:
    I_out = np.zeros((h, w, 3), dtype = np.float)
  else :
    I_out = np.zeros((h, w, 1), dtype = np.float)
  # resize
  I = cv2.resize(im, None, None, fx = np.float(sz), fy = np.float(sz), interpolation=cv2.INTER_LINEAR)
  h_out = min(im.shape[0],I.shape[0])
  w_out = min(im.shape[1],I.shape[1])
  out_start=(int((h-h_out)/2), int((w-w_out)/2))
  in_start=(int((I.shape[0]-h_out)/2), int((I.shape[1]-w_out)/2))
  # set resized image in the center of output
  if c==3:
    I_out[out_start[0]:out_start[0] + h_out, out_start[1]:out_start[1] + w_out, :] = I[in_start[0]:in_start[0] + h_out, in_start[1]:in_start[1] + w_out, :]
  else :
    I_out[out_start[0]:out_start[0] + h_out, out_start[1]:out_start[1] + w_out, 0] = I[in_start[0]:in_start[0] + h_out, in_start[1]:in_start[1] + w_out]
  I_out = np.swapaxes(I_out, 0, 2).astype('uint8')
  #del im, I
  #assert I_out.shape == np.swapaxes(im, 0, 2).shape
  return I_out


def resize_TTA(im, sz):
  im = np.swapaxes(im, 0, 2)
  h = im.shape[0]
  w = im.shape[1]
  c = im.shape[2]
  # identify if im is a color image or a mask
  if c == 3:
    I_out = np.zeros((h, w, 3), dtype=np.float)
  else :
    I_out = np.zeros((h, w), dtype=np.float)
  # resize
  I = cv2.resize(im, None, None, fx = np.float(sz), fy = np.float(sz), interpolation=cv2.INTER_LINEAR)
  h_out = min(im.shape[0],I.shape[0])
  w_out = min(im.shape[1],I.shape[1])
  out_start=(int((h-h_out)/2), int((w-w_out)/2))
  in_start=(int((I.shape[0]-h_out)/2), int((I.shape[1]-w_out)/2))
  # set resized image in the center of output
  if c==3:
    I_out[out_start[0]:out_start[0] + h_out, out_start[1]:out_start[1] + w_out, :] = I[in_start[0]:in_start[0] + h_out, in_start[1]:in_start[1] + w_out, :]
    I_out = np.swapaxes(I_out, 0, 2).astype('uint8')
  else :
    I_out[out_start[0]:out_start[0] + h_out, out_start[1]:out_start[1] + w_out] = I[in_start[0]:in_start[0] + h_out, in_start[1]:in_start[1] + w_out]
    I_out=np.swapaxes(I_out, 0, 1)
    #plt.imshow(I_out)
    #plt.show()


  #del im, I
  #assert I_out.shape == np.swapaxes(im, 0, 2).shape
  return I_out
