from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import math


@DETECTORS.register_module()
class SSD_US(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(SSD_US, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        self.dropout = torch.nn.Dropout2d(p=0.5)

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

        cat_img = torch.cat([img, unlabeled_img])
        cat_img_flip = torch.flip(cat_img, [3])

        cat_x = self.extract_feat(cat_img)
        cat_cls_score, cat_bbox_pred = self.bbox_head(cat_x)
        sup_cls_score = [s[:len(img)].clone() for s in cat_cls_score]
        sup_bbox_pred = [b[:len(img)].clone() for b in cat_bbox_pred]

        cat_x_aug = self.extract_feat(cat_img_flip)
        cat_cls_score_aug, cat_bbox_pred_aug = self.bbox_head(cat_x_aug)
        cat_cls_score_aug = [torch.flip(s, [3]) for s in cat_cls_score_aug]
        cat_bbox_pred_aug = [torch.flip(b, [3]) for b in cat_bbox_pred_aug]

        losses = {}

        sup_losses = self.bbox_head.loss(sup_cls_score, sup_bbox_pred, 
                                    gt_bboxes, 
                                    gt_labels, 
                                    img_metas,
                                    gt_bboxes_ignore=gt_bboxes_ignore)
        losses['sup_loss_cls'] = sup_losses.pop('loss_cls')
        losses['sup_loss_bbox'] = sup_losses.pop('loss_bbox')

        cr_losses = self.bbox_head.cr_loss(cat_cls_score_aug, cat_bbox_pred_aug, cat_cls_score, cat_bbox_pred,
                                        cur_iter=cur_iter, max_iters=max_iters, gt_bboxes_ignore=gt_bboxes_ignore)
        # losses['cr_loss_cls'] = cr_losses.pop('cr_loss_cls')
        losses['cr_loss_bbox'] = cr_losses.pop('cr_loss_bbox')

        return losses

