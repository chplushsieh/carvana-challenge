import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize(image, target):
    '''
    input:
      image:  a numpy array of shape (num of channels, Height, Width)
      target: a numpy array of shape (1, Height, Width) with only 1's and 0's in it
    '''
       
    assert image.shape[:-1] == target.shape
    


    
    
    plt.imshow(target, 'gray', interpolation='none')
    plt.imshow(image, 'gray', interpolation='none', alpha=0.6)
    plt.show()  


#example
#input_path='/home/judichunt/Downloads/'
#target = np.array(Image.open(input_path+'train_masks/00087a6bd4dc_01_mask.gif'), dtype=np.uint8)
#image = np.array(Image.open(input_path+'train/00087a6bd4dc_01.jpg'), dtype=np.uint8)
#visualize(image, target)

    # image, target, image overlaied with target
    # like so: https://kaggle2.blob.core.windows.net/forum-message-attachments/208218/6916/starter-kit.png
