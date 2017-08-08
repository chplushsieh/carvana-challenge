import torch
from torch.autograd import Variable
import numpy as np

import time

import util.exp as exp
import util.evaluation as evaluation

from dataloader import *
import config



def tester(exp_name, data_loader):
    cfg = config.load_config_file(exp_name)

    net, _, _ = exp.load_exp(exp_name)

    if torch.cuda.is_available():
        net.cuda()
    net.eval()  # Change model to 'eval' mode

    # Testing setting
    DEBUG = cfg['DEBUG']

    # initialize stats
    val_accuracy = 0

    epoch_start = time.time()

    for i, (img_name, images, targets) in enumerate(data_loader):
        iter_start = time.time()

        images = images.float()  # convert to FloatTensor
        images = Variable(images, volatile=True) # no need to compute gradients

        if torch.cuda.is_available():
            images = images.cuda()

        outputs = net(images)

        # compute dice
        masks = (outputs > 0.5).float()
        accuracy = evaluation.dice(masks.data[0].cpu().numpy(), targets.data[0].cpu().numpy())

        # Update stats
        val_accuracy += accuracy

        iter_end = time.time()
        print('Iter {}/{}, Image {}, Accuracy:{:.5f}, {:.2f} sec spent'.format(i, len(data_loader), img_name, accuracy, iter_end - iter_start))

    val_accuracy /= len(data_loader)

    epoch_end = time.time()

    print('Avg. Accuracy:{:.5f}, {:.2f} sec spent'.format(val_accuracy, epoch_end - epoch_start))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', nargs='?', default='upsamplingUnet')
    args = parser.parse_args()

    exp_name = args.exp_name

    data_loader = get_small_loader(
    # data_loader = get_val_loader(
    # data_loader = get_test_loader(
        cfg['train']['batch_size'],
        cfg['train']['paddings'],
        cfg['train']['tile_size']
    )

    tester(exp_name, data_loader)
