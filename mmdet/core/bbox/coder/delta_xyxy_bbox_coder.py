import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYXYBBoxCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds, max_shape, wh_ratio_clip)
        return decoded_bboxes


def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()

    px1 = proposals[..., 0]
    py1 = proposals[..., 1]
    px2 = proposals[..., 2]
    py2 = proposals[..., 3]
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx1 = gt[..., 0]
    gy1 = gt[..., 1]
    gx2 = gt[..., 2]
    gy2 = gt[..., 3]

    dx1 = (gx1 - px1) / pw
    dy1 = (gy1 - py1) / ph
    dx2 = (gx2 - px2) / pw
    dy2 = (gy2 - py2) / ph
    deltas = torch.stack([dx1, dy1, dx2, dy2], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000):

    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means

    dx1 = denorm_deltas[:, 0::4]
    dy1 = denorm_deltas[:, 1::4]
    dx2 = denorm_deltas[:, 2::4]
    dy2 = denorm_deltas[:, 3::4]

    px1 = (rois[:, 0]).unsqueeze(1).expand_as(dx1)
    py1 = (rois[:, 1]).unsqueeze(1).expand_as(dy1)
    px2 = (rois[:, 2]).unsqueeze(1).expand_as(dx2)
    py2 = (rois[:, 3]).unsqueeze(1).expand_as(dy2)
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dx1)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dx1)

    x1 = px1 + pw * dx1
    y1 = py1 + ph * dy1
    x2 = px2 + pw * dx2
    y2 = py2 + ph * dy2

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes