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
    #print(image.shape)
    #print(pred.shape)


    assert image.shape[:-1] == pred.shape

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, 'gray', interpolation='none')
    plt.imshow(pred, 'BuGn', interpolation='none', alpha=0.3)


    plt.subplot(1,2,2)
    plt.imshow(pred, 'BuGn', interpolation='none')
    plt.show()

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

# example
# input_path='/home/judichunt/Downloads/'
# target = np.array(Image.open(input_path+'train_masks/00087a6bd4dc_01_mask.gif'), dtype=np.uint8)
# image = np.array(Image.open(input_path+'train/00087a6bd4dc_01.jpg'), dtype=np.uint8)
# pred = np.array(Image.open(input_path+'train_masks/0cdf5b5d0ce1_01_mask.gif'), dtype=np.uint8)
# visualize(image, pred, target)

    # image, target, image overlaied with target
    # like so: https://kaggle2.blob.core.windows.net/forum-message-attachments/208218/6916/starter-kit.png
