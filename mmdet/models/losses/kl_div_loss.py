import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def kl_div_loss(pred, target):
    return F.kl_div(pred, target, reduction='none')


@LOSSES.register_module()
class KLDivLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, loss_weight_multi=None):
        loss_weight = loss_weight_multi*self.loss_weight if loss_weight_multi!=None else self.loss_weight
        loss = loss_weight * kl_div_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
        return loss
