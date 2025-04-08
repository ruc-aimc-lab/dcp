import torch
import torch.nn as nn
import torch.nn.functional as F


e = 1-10


def dice_loss(pred, target, need_sigmoid=True):
    assert target.size() == pred.size()
    if need_sigmoid:
        pred = torch.sigmoid(pred)
    intersect = 2 * (pred * target).sum() + e
    union = (pred * pred).sum() + (target * target).sum() + e
    return 1 - intersect / union


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return dice_loss(pred=pred, target=target)
    

class DiceBCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return  0.5 * dice_loss(pred=pred, target=target) + \
              0.5 * F.binary_cross_entropy_with_logits(input=pred, target=target)
              



