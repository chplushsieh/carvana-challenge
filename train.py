import torch
import torch.nn as nn
from torch.autograd import Variable

import time

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

import model.loss as loss
import util.exp as exp
import util.evaluation as evaluation
from dataloader import *
from util.visualization import *
import config

exp_name = 'upsamplingUnet_6G'

cfg = config.load_config_file(exp_name)

net, optimizer, start_epoch = exp.load_exp(exp_name)

data_loader = get_small_loader(
    cfg['train']['batch_size'],
    cfg['train']['paddings'],
    cfg['train']['tile_size']
)

if torch.cuda.is_available():
    net.cuda()
net.train()

# Loss and Optimizer
criterion = loss.StableBCELoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()

# set up TensorBoard
use_tensorboard = cfg['use_tensorboard']
experiment, use_tensorboard = exp.setup_crayon(use_tensorboard, CrayonClient, exp_name)

# Training setting
DEBUG = cfg['DEBUG']
log_iter_interval     = cfg['log_iter_interval']
snapshot_epoch_interval = cfg['snapshot_epoch_interval']
num_epochs = cfg['num_epochs']

# Train the Model
for epoch in range(start_epoch, num_epochs + 1):
    epoch_start = time.time()
    epoch_train_loss = 0
    epoch_train_accuracy   = 0

    print('Epoch [%d/%d] starts'
          % (epoch, num_epochs))

    for i, (img_name, images, targets) in enumerate(data_loader):
        iter_start = time.time()
        # print('Epoch {}, Iter {}, Image {}'.format(epoch, i, img_name))

        # convert from DoubleTensor to FloatTensor
        images = images.float()
        targets = targets.float()

        images = Variable(images)
        targets = Variable(targets)

        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        # Forward + Compute Loss
        outputs = net(images)
        # TODO remove tile borders from both outputs and targets
        loss = criterion(outputs, targets)

        # generate prediction
        masks = (outputs > 0.5).float()
        accuracy  = evaluation.dice( masks.data[0].cpu().numpy(), targets.data[0].cpu().numpy())

        # Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update epoch stats
        epoch_train_loss += loss.data[0]
        epoch_train_accuracy   += accuracy

        # Save the trained model
        if (i + 1) == 1 and epoch % snapshot_epoch_interval == 0:
            exp.save_checkpoint(exp_name, epoch, net.state_dict(), optimizer.state_dict())

        iter_end = time.time()

        # Log Training Progress
        if (i + 1) % log_iter_interval == 0:
            print('Epoch [%d/%d] Iter [%d/%d] Loss: %.2f Accuracy: %.4f Time Spent: %.2f sec'
                  % (epoch, num_epochs, i + 1, len(data_loader), loss.data[0], accuracy, iter_end - iter_start))

        if DEBUG:
            print('Epoch {}, Iter {}, {}: Loss {:.3f}, Accuracy: {:.4f}'.format(epoch, i, img_name, loss.data[0], accuracy))
            visualize(images.data[0].cpu().numpy(), masks.data[0].cpu().numpy(), targets.data[0].cpu().numpy())



    epoch_train_loss /= len(data_loader)
    epoch_train_accuracy   /= len(data_loader)

    if use_tensorboard:
        experiment.add_scalar_value('train loss', epoch_train_loss, step=epoch)
        # experiment.add_scalar_value('val loss', epoch_val_loss, step=epoch)
        experiment.add_scalar_value('accuracy', epoch_train_accuracy, step=epoch)
        # experiment.add_scalar_value('learning_rate', lr, step=epoch)

    epoch_end = time.time()

    print('Epoch [%d/%d] Loss: %.2f Accuracy: %.4f Time Spent: %.2f sec'
          % (epoch, num_epochs, epoch_train_loss, epoch_train_accuracy, epoch_end - epoch_start))
