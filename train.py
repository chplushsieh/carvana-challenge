import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

import util.exp as exp
import util.evaluation as evaluation
import util.visualization as viz
import util.tile as tile

from dataloader import *
import config
import test


def trainer(exp_name, train_data_loader, train_tile_borders, cfg, val_data_loader=None, val_tile_borders=None, DEBUG=False, use_tensorboard=True):
    net, optimizer, criterion, start_epoch = exp.load_exp(exp_name)

    if torch.cuda.is_available():
        net.cuda()
        criterion = criterion.cuda()
    net.train()  # Change model to 'train' mode

    # set up TensorBoard
    experiment, use_tensorboard = exp.setup_crayon(use_tensorboard, CrayonClient, exp_name)

    # Training setting
    log_iter_interval     = cfg['log_iter_interval']
    snapshot_epoch_interval = cfg['snapshot_epoch_interval']
    num_epochs = cfg['num_epochs']
    accumulated_batch_size = cfg['train']['accumulated_batch_size']

    # Train the Model
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # initialize epoch stats
        epoch_train_loss = 0
        epoch_train_accuracy   = 0
        accumulated_batch_loss = 0

        print('Epoch [%d/%d] starts'
              % (epoch, num_epochs))

        for i, (img_name, images, targets) in enumerate(train_data_loader):
            iter_start = time.time()

            # convert to FloatTensor
            images = images.float()
            targets = targets.float()

            images = Variable(images)
            targets = Variable(targets)

            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()

            outputs = net(images)
            loss = criterion(outputs, targets)

            # generate prediction
            masks = (outputs > 0.5).float()

            # convert to numpy array
            image = images.data[0].cpu().numpy()
            mask = masks.data[0].cpu().numpy()
            target = targets.data[0].cpu().numpy()

            # remove tile borders
             # TODO modify for case with batch size > 1
            image = tile.remove_tile_borders(image, train_tile_borders)
            mask = tile.remove_tile_borders(mask, train_tile_borders)
            target = tile.remove_tile_borders(target, train_tile_borders)
            accuracy  = evaluation.dice(mask, target) # TODO modify for case with batch size > 1
            epoch_train_accuracy += accuracy

            # Backward pass
            loss.backward()
            accumulated_batch_loss += (loss.data[0] / accumulated_batch_size)

            # Update epoch stats
            epoch_train_loss     += loss.data[0]  # TODO modify for case with batch size > 1

            # Log Training Progress
            if (i + 1) % log_iter_interval == 0:
                print('Epoch [%d/%d] Iter [%d/%d] Loss: %.2f Accumd Loss:%.2f Accuracy: %.4f Time Spent: %.2f sec'
                      % (epoch, num_epochs, i + 1, len(train_data_loader), loss.data[0], accumulated_batch_loss, accuracy, iter_end - iter_start))

            if DEBUG and accuracy < 0.8:
                print('Epoch {}, Iter {}, {}: Loss {:.3f}, Accuracy: {:.4f}'.format(epoch, i, img_name, loss.data[0], accuracy))
                viz.visualize(image, mask, target)

            if (i+1) % accumulated_batch_size == 0:
                optimizer.step()

                # reset
                optimizer.zero_grad()
                accumulated_batch_loss = 0

            iter_end = time.time()

        # inner for loop ends

        epoch_train_loss     /= len(train_data_loader)
        epoch_train_accuracy /= len(train_data_loader)

        # Validate
        if val_data_loader is not None:
            epoch_val_loss, epoch_val_accuracy = test.tester(exp_name, val_data_loader, val_tile_borders, net, criterion, is_val=True)

        if use_tensorboard:
            experiment.add_scalar_value('train loss', epoch_train_loss, step=epoch)
            experiment.add_scalar_value('train accuracy', epoch_train_accuracy, step=epoch)

            if val_data_loader is not None:
                experiment.add_scalar_value('val loss', epoch_val_loss, step=epoch)
                experiment.add_scalar_value('val accuracy', epoch_val_accuracy, step=epoch)

            # experiment.add_scalar_value('learning_rate', lr, step=epoch)

        # Save the trained model
        if epoch % snapshot_epoch_interval == 0:
            exp.save_checkpoint(exp_name, epoch, net.state_dict(), optimizer.state_dict())

        epoch_end = time.time()
        print('Epoch [%d/%d] Loss: %.3f Accuracy: %.4f Time Spent: %.2f sec'
              % (epoch, num_epochs, epoch_train_loss, epoch_train_accuracy, epoch_end - epoch_start))

    # outer for loop ends

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', nargs='?', default='DynamicUnet')
    args = parser.parse_args()

    exp_name = args.exp_name

    cfg = config.load_config_file(exp_name)
    # train_data_loader, train_tile_borders = get_small_loader(
    train_data_loader, train_tile_borders = get_train_loader(
        cfg['train']['batch_size'],
        cfg['train']['paddings'],
        cfg['train']['tile_size'],
        cfg['train']['hflip'],
        cfg['train']['shift']
    )

    # val_data_loader, val_tile_borders = get_small_loader(
    val_data_loader, val_tile_borders = get_val_loader(
        cfg['test']['batch_size'],
        cfg['test']['paddings'],
        cfg['test']['tile_size'],
        cfg['test']['hflip'],
        cfg['test']['shift']
    )

    assert cfg['train']['batch_size'] == 1
    trainer(exp_name, train_data_loader, train_tile_borders, cfg, val_data_loader=val_data_loader, val_tile_borders=val_tile_borders, DEBUG=False)
