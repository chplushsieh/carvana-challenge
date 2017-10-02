import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize(image, pred, target=None):
    '''
    input:
      image:  a numpy array of shape (num of channels, Height, Width)
      pred: a numpy array of shape (1, Height, Width) with numbers between 0 and 1 in it
      target: a numpy array of shape (1, Height, Width) with only 1's and 0's in it
    '''
    image = image.astype('uint8')

    def change_index_ord(img):
        img=np.swapaxes(img,0,2)
        img=np.swapaxes(img,0,1)
        return img

    image=change_index_ord(image)
    pred=np.squeeze(change_index_ord(pred) , axis=2)

    assert image.shape[:-1] == pred.shape

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)

    plt.subplot(1,3,2)
    plt.imshow(image, 'gray', interpolation='none')
    plt.imshow(pred, 'BuGn', interpolation='none', alpha=0.3)


    plt.subplot(1,3,3)
    plt.imshow(pred, 'BuGn', interpolation='none')


    if target is not None:
        target=np.squeeze(change_index_ord(target) , axis=2)
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(image, 'gray', interpolation='none')
        plt.imshow(pred, 'BuGn', interpolation='none', alpha=0.4)
        plt.imshow(target, 'Purples', interpolation='none', alpha=0.3)


        plt.subplot(2,2,2)
        plt.imshow(pred, 'BuGn', interpolation='none')

        plt.subplot(2,2,3)
        plt.imshow(target, 'Purples', interpolation='none')

        plt.subplot(2, 2, 4)
        plt.imshow((pred+target*0.5), 'Accent', interpolation='none')

    plt.show()
    return
