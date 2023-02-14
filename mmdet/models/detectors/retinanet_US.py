from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import math


@DETECTORS.register_module()
class RetinaNet_US(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 dropout_ratio=None,
                 ):
        super(RetinaNet_US, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        self.dropout = torch.nn.Dropout2d(p=dropout_ratio)

    def forward_train(self, data, p_h, cur_iter=None, max_iters=None, gt_bboxes_ignore=None):
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
            data[0]['gt_bboxes'], data[0]['gt_labels'], data[0]['img'], data[0]['img_metas']
        
        unlabeled_img= data[1]['img']

        # for cpu train
        if torch.is_tensor(img)==False:
            img = img.data[0]
            img_metas = img_metas.data[0]
            gt_bboxes = gt_bboxes.data[0]
            gt_labels = gt_labels.data[0]
            unlabeled_img = unlabeled_img.data[0]

        losses = {}

        x_sup = self.extract_feat(img) # feature
        cls_score_sup, bbox_pred_sup = self.bbox_head(x_sup)
        sup_losses = self.bbox_head.loss(cls_score_sup, bbox_pred_sup, 
                                    gt_bboxes, 
                                    gt_labels, 
                                    img_metas,
                                    gt_bboxes_ignore=gt_bboxes_ignore)
        losses['sup_loss_cls'] = sup_losses.pop('loss_cls')
        losses['sup_loss_bbox'] = sup_losses.pop('loss_bbox')

        x_unsup = self.extract_feat(unlabeled_img)
        cls_score_unsup, bbox_pred_unsup = self.bbox_head(x_unsup)
        aug_x_unsup = self.extract_feat(torch.flip(unlabeled_img, [3])) # flip
        # aug_x_unsup = tuple([self.dropout(x_i) for x_i in x_unsup]) # dropout
        aug_cls_score_unsup, aug_bbox_pred_unsup = self.bbox_head(aug_x_unsup)
        unsup_losses = self.bbox_head.cr_loss(aug_cls_score_unsup, aug_bbox_pred_unsup, cls_score_unsup, bbox_pred_unsup,
                                        cur_iter=cur_iter, max_iters=max_iters, gt_bboxes_ignore=gt_bboxes_ignore)
        losses['unsup_cr_loss_cls'] = unsup_losses.pop('cr_loss_cls')
        losses['unsup_cr_loss_bbox'] = unsup_losses.pop('cr_loss_bbox')

        aug_x_sup = self.extract_feat(torch.flip(img, [3])) # flip
        # aug_x_unsup = tuple([self.dropout(x_i) for x_i in x_unsup]) # dropout
        aug_cls_score_sup, aug_bbox_pred_sup = self.bbox_head(aug_x_sup)
        cr_sup_losses = self.bbox_head.cr_loss(aug_cls_score_sup, aug_bbox_pred_sup, cls_score_sup, bbox_pred_sup,
                                        cur_iter=cur_iter, max_iters=max_iters, gt_bboxes_ignore=gt_bboxes_ignore)
        losses['sup_cr_loss_cls'] = cr_sup_losses.pop('cr_loss_cls')
        losses['sup_cr_loss_bbox'] = cr_sup_losses.pop('cr_loss_bbox')

        return losses
