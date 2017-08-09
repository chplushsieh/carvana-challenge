import torch
from torch.autograd import Variable
import numpy as np

import time

import util.exp as exp
import util.evaluation as evaluation
import util.visualization as viz
import util.submit as submit
import util.tile as tile

from dataloader import *
import config



def tester(exp_name, data_loader, is_val=False):
    cfg = config.load_config_file(exp_name)

    net, _, criterion, _ = exp.load_exp(exp_name)

    if torch.cuda.is_available():
        net.cuda()
        criterion = criterion.cuda()
    net.eval()  # Change model to 'eval' mode

    # Testing setting
    DEBUG = cfg['DEBUG']

    # initialize stats
    if is_val:
        epoch_val_loss = 0
        epoch_val_accuracy = 0
    else:
        predictions = {}

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

        if is_val:
            accuracy = evaluation.dice(masks.data[0].cpu().numpy(), targets.data[0].cpu().numpy())
            loss = criterion(outputs, targets)

            # Update stats
            epoch_val_loss     += loss.data[0]
            epoch_val_accuracy += accuracy
        else:
            predictions[img_name] = masks.data[0].cpu().numpy()

        iter_end = time.time()

        print('Iter {}/{}, Image {}: {:.2f} sec spent'.format(i, len(data_loader), img_name, accuracy, iter_end - iter_start))

        if DEBUG:
            if val:
                print('Iter {}, {}: Loss {:.3f}, Accuracy: {:.4f}'.format(i, img_name, loss.data[0], accuracy))
                viz.visualize(images.data[0].cpu().numpy(), masks.data[0].cpu().numpy(), targets.data[0].cpu().numpy())
            else:
                viz.visualize(images.data[0].cpu().numpy(), masks.data[0].cpu().numpy())


    if is_val:
        epoch_val_loss     /= len(data_loader)
        epoch_val_accuracy /= len(data_loader)
        print('Validation Loss: {:.3f} Validation Accuracy:{:.5f}'.format(epoch_val_loss, val_accuracy))
    else:
        pass
        # TODO haven't implement yet:
        # predictions = tile.stitch_predictions(predictions)
        # submit.save_predictions(exp_name, predictions)

    epoch_end = time.time()
    print('{:.2f} sec spent'.format(epoch_end - epoch_start))

    if is_val:
        return epoch_val_loss, epoch_val_accuracy
    else:
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

    # TODO load net and pass it to tester
    # TODO and do the same thing to trainer as well
    tester(exp_name, data_loader, is_val=False)
