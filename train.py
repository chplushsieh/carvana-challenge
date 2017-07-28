import torch
import torch.nn as nn
from torch.autograd import Variable

import time

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

import util.exp as exp
import util.tool as tool
from dataloader import *
import config

exp_name = 'denseunet_on_trainval'

cfg = config.load_config_file(exp_name)

net, optimizer, start_epoch = exp.load_exp(exp_name)

data_loader = get_trainval_patch_loader(
    cfg['train']['batch_size']
)

if torch.cuda.is_available():
    net.cuda()
net.train()

# Loss and Optimizer
criterion = nn.MSELoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()

# set up TensorBoard
use_tensorboard = cfg['use_tensorboard']
experiment, use_tensorboard = exp.setup_crayon(use_tensorboard, CrayonClient, exp_name)

# Training setting
DEBUG = cfg['DEBUG']
log_iter_interval     = cfg['log_iter_interval']
display_iter_interval = cfg['display_iter_interval']
snapshot_epoch_interval = cfg['snapshot_epoch_interval']
num_epochs = cfg['num_epochs']

# Train the Model
for epoch in range(start_epoch, num_epochs + 1):

    print('Epoch [%d/%d] starts'
          % (epoch, num_epochs))

    for i, (img_name, images, labels) in enumerate(data_loader):
        iter_start = time.time()
        # print('Epoch {}, Iter {}, Image {}'.format(epoch, i, img_name))

        images = images.float()  # convert from ByteTensor   to FloatTensor

        images = Variable(images)
        labels = Variable(labels)

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Compute Loss
        outputs = net(images)
        loss = criterion(outputs, labels)

        # Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_tensorboard and (i + 1) % display_iter_interval == 0:
            step = len(data_loader) * (epoch - 1) + (i + 1)
            loss_in_np = loss.data[0]
            experiment.add_scalar_value('train_loss', loss_in_np, step=step)
            # exp.add_scalar_value('learning_rate', lr, step=step)

        # Save the trained model
        if (i + 1) == 1 and epoch % snapshot_epoch_interval == 0:
            exp.save_checkpoint(exp_name, epoch, net.state_dict(), optimizer.state_dict())

        iter_end = time.time()
        # Display Training Progress
        if (i + 1) % log_iter_interval == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.2f Time Spent: %.2f sec'
                  % (epoch, num_epochs, i + 1, len(data_loader), loss.data[0], iter_end - iter_start))

        if DEBUG:
            print('Epoch {}, Iter {}, Image {}, Loss {}'.format(epoch, i, img_name, loss.data[0]))
            tool.show_denmap(2, images.data[0].cpu().byte().numpy(), exp_name, labels.data[0].cpu().numpy(), outputs.data[0].cpu().numpy())
