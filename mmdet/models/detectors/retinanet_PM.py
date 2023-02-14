from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import math


@DETECTORS.register_module()
class RetinaNet_PM(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_PM, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

    def forward_train(self, ori_data, aug_data, epoch=None,
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
        # TODO: check this
        if gt_bboxes_ignore!=None:
            NotImplementedError

        gt_bboxes, gt_labels, img, img_metas = \
            ori_data['gt_bboxes'], ori_data['gt_labels'], ori_data['img'], ori_data['img_metas']
        
        aug_gt_bboxes, aug_gt_labels, aug_img, aug_img_metas = \
            aug_data['gt_bboxes'], aug_data['gt_labels'], aug_data['img'], aug_data['img_metas']

        # for cpu train
        if not torch.is_tensor(img):
            img = img.data[0]
            img_metas = img_metas.data[0]
            gt_bboxes = gt_bboxes.data[0]
            gt_labels = gt_labels.data[0]
        if not torch.is_tensor(aug_img):
            aug_img = aug_img.data[0]
            aug_img_metas = aug_img_metas.data[0]
            aug_gt_bboxes = aug_gt_bboxes.data[0]
            aug_gt_labels = aug_gt_labels.data[0]
        
        # check ori_data and aug_data
        assert len(aug_img)==len(img)
        assert len(aug_img_metas)==len(img_metas)
        assert len(aug_gt_bboxes)==len(gt_bboxes)
        assert len(aug_gt_labels)==len(gt_labels)
        for i in range(len(img_metas)):
            assert aug_img_metas[i]['filename']==img_metas[i]['filename']
            assert torch.equal(aug_gt_labels[i], gt_labels[i])

        x = self.extract_feat(img)
        cls_score, bbox_pred = self.bbox_head(x)

        aug_img = torch.flip(aug_img, [3])
        aug_x = self.extract_feat(aug_img)
        aug_cls_score, aug_bbox_pred = self.bbox_head(aug_x)

        loss_inputs = (cls_score, bbox_pred, gt_bboxes, gt_labels, img_metas,
                    aug_cls_score, aug_bbox_pred, None, None, None, epoch)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        return losses
