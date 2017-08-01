import numpy as np

'''
This Kaggle competition is evaluated on the mean Dice coefficient:
https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
'''

def dice(x, y):
    '''
    input:
      x: an numpy array with only 0's and 1's in it
      y: an numpy array with only 0's and 1's in it, which is of the same size as x

    output:
      dice: Dice coefficient for x and y, which is a decimal number between 0 and 1
    '''
    assert x.shape == y.shape

    smooth =0            # smooth should be replaced by 1 for stability

    dice = None
    # TODO
    x=x.flatten()
    y=y.flatten()
    intersection = sum(x*y)
    
    dice= (2. * intersection + smooth) / (x.sum() + y.sum() + smooth)
    return dice

if __name__ == "__main__":

    pred = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1]
    ])

    groundtruth = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0]
    ])

    assert dice(pred, groundtruth) == 2*6/(8+10)
