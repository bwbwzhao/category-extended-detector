import torch.nn as nn
import torch

from mmdet.core import bbox2result, bbox_overlaps_np
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import numpy as np


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      cur_iter, 
                      max_iters,
                      gt_confidences=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # just for cpu debug
        if not torch.is_tensor(img):
            img = img.data[0]
            img_metas = img_metas.data[0]
            gt_bboxes = gt_bboxes.data[0]
            gt_labels = gt_labels.data[0]
            if gt_confidences:
                gt_confidences = gt_confidences.data[0]

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if gt_confidences:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_confidences, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, cur_iter=cur_iter, max_iters=max_iters)
        return losses

    def simple_test(self, img, img_metas, test_type=None, rescale=False):
        if test_type=='drop':
            return self.drop_test(img, img_metas, rescale)

        # just for cpu debug
        if not torch.is_tensor(img):
            img = img.data[0]
        if not isinstance(img_metas, list):
            img_metas = img_metas.data[0]

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
            raise NotImplementedError

    def drop_test(self, img, img_metas, rescale=False):
        # just for cpu debug
        if not torch.is_tensor(img):
            img = img.data[0]
        if not isinstance(img_metas, list):
            img_metas = img_metas.data[0]

        x = self.extract_feat(img)
        # get multiple bbox_results
        test_runs = 20
        self.bbox_head.dropout.train()
        outs = self.bbox_head.forward_multidrop(x, test_runs)
        img_metas_expand = [img_metas[0] for i in range(test_runs)]
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas_expand, rescale=rescale)
        bbox_list = (
            torch.cat([bbox_list[i][0] for i in range(test_runs)]).cpu().numpy(), 
            torch.cat([bbox_list[i][1] for i in range(test_runs)]).cpu().numpy(),
        )

        # clustering
        if bbox_list[0].shape[0] == 0:
            return [np.zeros((0, 7), dtype=np.float32) for i in range(self.bbox_head.num_classes)]
        else:
            class_results = [self.clustering(bbox_list[0][bbox_list[1] == i, :], 
                                            test_runs, 
                                            culter_thre=self.test_cfg['nms']['iou_thr'], 
                                            min_ele=1) 
                            for i in range(self.bbox_head.num_classes)]
            return class_results

    def clustering(self, bboxes, test_runs, culter_thre, min_ele):
        bbox_num, bbox_dim = bboxes.shape[0], bboxes.shape[1]
        assert bbox_dim==5
        if bbox_num==0:
            return np.zeros((0, 7), dtype=np.float32)

        clusters = [bboxes[0:1]]
        for i in range(1, bbox_num):
            i_bbox = bboxes[i:i+1]
            ious = bbox_overlaps_np(
                        i_bbox[:,:4], 
                        np.concatenate(clusters)[:,:4], 
                        is_aligned=False)[0]
            min_ious = []
            nums = 0
            for cluster in clusters:
                min_ious.append(ious[nums:nums+len(cluster)].min())
                nums+=len(cluster)
            min_ious = np.array(min_ious)
            max_min_iou, max_min_idx = min_ious.max(), min_ious.argmax()
            if max_min_iou > culter_thre:
                clusters[max_min_idx] = np.concatenate((clusters[max_min_idx], i_bbox), axis=0) # update cluster
            else:
                clusters.append(i_bbox) # new cluster
        clusters = [cluster for cluster in clusters if len(cluster) >= min_ele]
        if len(clusters) < 1:
            return np.zeros((0, 7), dtype=np.float32)
        
        results = []
        for cluster in clusters:
            cluster_mean = cluster.mean(axis=0)
            iou_mean = bbox_overlaps_np(
                                cluster[:,:4], 
                                cluster[:,:4], 
                                is_aligned=False).mean()
            freq = len(cluster) / test_runs
            results.append(np.concatenate((
                cluster_mean[:4],
                cluster_mean[4:5], 
                np.array([iou_mean]),
                np.array([freq]),
            )))
        return np.stack(results)
