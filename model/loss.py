import torch.nn.functional as F
import torch.nn as nn


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


class SoftDiceLoss(nn.Module):
    '''
    from https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    '''
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, inputs, targets):
        num = targets.size(0)
        m1  = inputs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1 - score.sum()/num
        return score

class HengLoss(nn.Module):
    def __init__(self):
        super(HengLoss, self).__init__()
        self.dice = SoftDiceLoss()
        self.bce = BCELoss2d()


    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + self.bce(inputs, targets)
