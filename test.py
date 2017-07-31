import torch
from torch.autograd import Variable
import numpy as np

import time

import util.exp as exp
import util.val as val
import util.load as load

from dataloader import *
import config


exp_name = 'smallunet'

cfg = config.load_config_file(exp_name)

net, _, start_epoch = exp.load_exp(exp_name)

data_loader = get_small_loader(
    cfg['train']['batch_size']
)
IS_TEST = True

# Testing setting
DEBUG = cfg['DEBUG']

if torch.cuda.is_available():
    net.cuda()

net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

pred_counts_by_img = [ {} for i in range(len(threshold_sets)) ]
gt_counts_by_img = {}

for i, (images, targets) in enumerate(data_loader):
    iter_start = time.time()
    print('Iter {}/{}, Image {} with size {}'.format(i, len(data_loader), img_name, images.size()))

    images = images.float()  # convert from DoubleTensor   to FloatTensor
    images = Variable(images, volatile=True)

    if torch.cuda.is_available():
        images = images.cuda()

    outputs = net(images)
    output = outputs.data[0].cpu().numpy()

    # For Validation Dataset
    if not IS_TEST:
        label = None
        # load counts of whole image from annotation
        gt_counts = load.get_counts_from_annotations(whole_img_name)
        # ... which can't be directly compared against image patch predicted counts

    # remove border from label and output
    output = output[:, tile_border:-tile_border, tile_border:-tile_border]

    # Save predictions
    # exp.save_prediction(exp_name, start_epoch, img_name[0], output)

    iter_end = time.time()
    print("It took {:.2f} sec".format(iter_end - iter_start))

    # compute rmse

    # save after rounding for submission
