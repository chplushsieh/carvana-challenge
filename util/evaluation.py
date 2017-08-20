import torch

# import numpy as np
# import time
'''
This Kaggle competition is evaluated on the mean Dice coefficient:
https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
'''

#https://github.com/pytorch/pytorch/issues/1249
def dice_loss(m1, m2 ):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = score.sum()/num
    return score

# def batch_dice(x, y):
#     '''
#     (batch_size, 1, height, width)
#     '''
#     # batch_start = time.time()
#     assert len(x) == len(y)
#
#     scores = np.zeros(len(x))
#     for i in range(len(x)):
#         scores[i] = dice(x[i], y[i])
#     # print(time.time() - batch_start)
#     return np.mean(scores)
#
#
# def dice(x, y):
#     '''
#     input:
#       x: an numpy array with only 0's and 1's in it
#       y: an numpy array with only 0's and 1's in it, which is of the same size as x
#
#     output:
#       dice: Dice coefficient for x and y, which is a decimal number between 0 and 1
#     '''
#     # func_start = time.time()
#     smooth = 1            # for stability
#
#     dice = None
#     x=x.flatten()
#     y=y.flatten()
#     assert x.shape == y.shape
#     intersection = sum(x*y)
#
#     dice= (2. * intersection + smooth) / (x.sum() + y.sum() + smooth)
#     # print(time.time() - func_start)
#     return dice
#
# if __name__ == "__main__":
#
#     pred = np.array([
#         [0, 0, 1, 0, 0],
#         [0, 1, 1, 1, 0],
#         [1, 1, 0, 1, 1]
#     ])
#
#     groundtruth = np.array([
#         [0, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1],
#         [0, 1, 0, 1, 0]
#     ])
#
#     assert dice(pred, groundtruth) == 2*6/(8+10)
