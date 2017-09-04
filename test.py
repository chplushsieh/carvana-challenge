import torch
from torch.autograd import Variable
import numpy as np

import time
import argparse

import util.exp as exp
import util.evaluation as evaluation
import util.visualization as viz
import util.submit as submit
import util.tile as tile
import util.crf as crf

from dataloader import *
import config



def tester(exp_name, data_loader, tile_borders, net, criterion, is_val=False, paddings=(0, 0), use_crf=False, DEBUG=False):

    if torch.cuda.is_available():
        net.cuda()
        criterion = criterion.cuda()
    net.eval()  # Change model to 'eval' mode

    # initialize stats
    if is_val:
        epoch_val_loss = 0
        epoch_val_accuracy = 0
    else:
        tile_masks = {}
        img_rles = {}

    epoch_start = time.time()

    for i, (img_name, images, targets) in enumerate(data_loader):
        iter_start = time.time()

        images = images.float()  # convert to FloatTensor
        targets = targets.float()

        images = Variable(images, volatile=True) # no need to compute gradients
        targets = Variable(targets, volatile=True)

        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        outputs = net(images)

        # remove tile borders
        images = tile.remove_tile_borders(images, tile_borders)
        outputs = tile.remove_tile_borders(outputs, tile_borders)

        if is_val:
            targets = tile.remove_tile_borders(targets, tile_borders)

        # compute dice
        masks = (outputs > 0.5).float()

        # apply CRF to image tiles
        if use_crf:
            for img_idx in range(len(img_name)):
                img = images.data[img_idx].cpu().numpy()
                prob = outputs.data[img_idx].cpu().numpy()
                crf_mask[img_idx] = crf.apply_crf(img, prob)

                # convert CRF results back into Variable in GPU
                masks = Variable(torch.from_numpy(crf_mask), volatile=True)
                if torch.cuda.is_available():
                    masks = masks.cuda()

                # TODO refactor the above block of code

        iter_end = time.time()

        if is_val:
            accuracy = evaluation.dice_loss(masks, targets)
            loss = criterion(outputs, targets)

            # Update stats
            epoch_val_loss     += loss.data[0]
            epoch_val_accuracy += accuracy
        else:
            for img_idx in range(len(img_name)):
                tile_masks[img_name[img_idx]] = masks.data[img_idx].cpu().numpy()

            # merge tile predictions into image predictions
            tile.merge_preds_if_possible(tile_masks, img_rles, paddings)

            print('Iter {}/{}: {:.2f} sec spent'.format(i, len(data_loader), iter_end - iter_start))

        if DEBUG:
            # convert to numpy array
            image = images.data[0].cpu().numpy()
            mask = masks.data[0].cpu().numpy()
            target = targets.data[0].cpu().numpy()

            if is_val and accuracy < 0.98:
                print('Iter {}, {}: Loss {:.4f}, Accuracy: {:.5f}'.format(i, img_name, loss.data[0], accuracy))
                viz.visualize(image, mask, target)
            else:
                viz.visualize(image, mask)
    # for loop ends

    if is_val:
        epoch_val_loss     /= len(data_loader)
        epoch_val_accuracy /= len(data_loader)
        print('Validation Loss: {:.4f} Validation Accuracy:{:.5f}'.format(epoch_val_loss, epoch_val_accuracy))
    else:
        assert len(tile_masks) == 0  # all tile predictions should now be merged into image predictions now
        submit.save_predictions(exp_name, img_rles)

    epoch_end = time.time()
    print('{:.2f} sec spent'.format(epoch_end - epoch_start))

    if is_val:
        net.train()  # Change model bacl to 'train' mode
        return epoch_val_loss, epoch_val_accuracy
    else:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', nargs='?', default='PeterUnet')
    args = parser.parse_args()

    exp_name = args.exp_name

    cfg = config.load_config_file(exp_name)
    # data_loader, tile_borders = get_small_loader(
    data_loader, tile_borders = get_val_loader(
    # data_loader, tile_borders = get_test_loader(
        cfg['test']['batch_size'],
        cfg['test']['paddings'],
        cfg['test']['tile_size'],
        cfg['test']['hflip'],
        cfg['test']['shift'],
        cfg['test']['color'],
        cfg['test']['rotate'],
        cfg['test']['scale']
    )

    net, _, criterion, _ = exp.load_exp(exp_name)

    tester(exp_name, data_loader, tile_borders, net, criterion, paddings=cfg['test']['paddings'], use_crf=True)

    # TODO try both of the following to see if CRF improves performance for validation
    # epoch_val_loss, epoch_val_accuracy = tester(exp_name, data_loader, tile_borders, net, criterion, is_val=True)
    # epoch_val_loss, epoch_val_accuracy = tester(exp_name, data_loader, tile_borders, net, criterion, is_val=True, use_crf=True)
