import torch

'''
This Kaggle competition is evaluated on the mean Dice coefficient:
https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
'''

def dice_loss(m1, m2 ):
    '''
    input:
      m1: a pytorch Variable of size  (batch_size, num_channels, heiht, width)
      m2:
    output:
      score: float

    ref: https://github.com/pytorch/pytorch/issues/1249
    '''
    m1 = m1.contiguous()
    m2 = m2.contiguous()

    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = score.sum()/num  # a Variable of FloatTensor of size 1
    return score.data[0]
