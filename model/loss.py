import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch

class CrossEntropy2dLoss(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2dLoss, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()

class BCELoss2d(nn.Module):
    '''
    from https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    '''
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, inputs, targets):
        inputs_flat   = inputs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(inputs_flat, targets_flat)

class WeightedBCELoss2d(nn.Module):
    '''
    from
    https://kaggle2.blob.core.windows.net/forum-message-attachments/212369/7045/weighted_dice_2.png
    '''
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, inputs, targets, weights):
        inputs   = inputs.view (-1)
        targets = targets.view(-1)
        weights   = weights.view (-1)

        loss = weights * inputs.clamp(min=0) - weights * inputs * targets + weights * torch.log(1 + torch.exp(-inputs.abs()))
        loss = loss.sum() / weights.sum()
        return loss


class SoftDiceLoss(nn.Module):
    '''
    from https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    '''
    def __init__(self):
        super(SoftDiceLoss, self).__init__()


    def forward(self, inputs, targets):
        num = targets.size(0)
        m1  = inputs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1 - score.sum()/num
        return score

class WeightedSoftDiceLoss(nn.Module):
    '''
    from
    https://kaggle2.blob.core.windows.net/forum-message-attachments/212369/7045/weighted_dice_2.png
    '''
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()


    def forward(self, inputs, targets, weights):
        num = targets.size(0)
        m1  = inputs.view(num,-1)
        m2  = targets.view(num,-1)
        w   = weights.view(num,-1)
        w2  = w * w
        intersection = (m1 * m2)

        score = 2. * ((w2 * intersection).sum(1)+1) / ((w2 * m1).sum(1) + (w2 * m2).sum(1)+1)
        score = 1 - score.sum()/num
        return score

class HengLoss(nn.Module):
    def __init__(self):
        super(HengLoss, self).__init__()
        self.dice = SoftDiceLoss()
        self.bce = BCELoss2d()

    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + self.bce(inputs, targets)

class StableHengLoss(nn.Module):
    def __init__(self):
        super(StableHengLoss, self).__init__()
        self.dice = SoftDiceLoss()
        self.stable_bce = StableBCELoss()

    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + self.stable_bce(inputs, targets)

class BoundaryWeightedLoss(nn.Module):
    def __init__(self):
        super(BoundaryWeightedLoss, self).__init__()
        self.weighted_bce = WeightedBCELoss2d()
        self.weighted_dice = WeightedSoftDiceLoss()

    def forward(self, inputs, targets):
        avg_neighbors = F.avg_pool2d(targets, kernel_size=11, padding=5, stride=1)
        is_boundary = avg_neighbors.ge(0.01) * avg_neighbors.le(0.99)
        is_boundary = is_boundary.float()

        weights = Variable(torch.tensor.torch.ones(avg_neighbors.size())).cuda()

        w0 = weights.sum()
        weights = weights + is_boundary * 2
        w1 = weights.sum()
        weights = weights * w0 / w1

        return self.weighted_dice(inputs, targets, weights) + self.weighted_bce(inputs, targets, weights)
