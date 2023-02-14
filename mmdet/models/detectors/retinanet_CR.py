from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import math


@DETECTORS.register_module()
class RetinaNet_CR(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 dropout_ratio=None):
        super(RetinaNet_CR, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        self.dropout = torch.nn.Dropout2d(p=dropout_ratio)

    def forward_train(self, data, p_h, cur_iter, max_iters, gt_bboxes_ignore=None):
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
        
        aug_imgs = [data[i]['img'] for i in range(1, len(data))]

        # for cpu train
        if torch.is_tensor(img)==False:
            img = img.data[0]
            img_metas = img_metas.data[0]
            gt_bboxes = gt_bboxes.data[0]
            gt_labels = gt_labels.data[0]
        if len(aug_imgs)!=0 and torch.is_tensor(aug_imgs[0])==False:
            for i in range(len(aug_imgs)):
                aug_imgs[i] = aug_imgs[i].data[0]

        # class splits
        class_splits = [[1,80], [0,1]]
        img_splits = [[], []]
        for i, img_meta in enumerate(img_metas):
            gt_label = gt_labels[i]
            if img_meta['class_set']=='A' and len(gt_label):
                img_splits[0].append(i)
            elif img_meta['class_set']=='B' and len(gt_label):
                img_splits[1].append(i)
            else:
                print(img_meta)

        x = self.extract_feat(img) # feature
        cls_score, bbox_pred = self.bbox_head(x)

        aug_x_list = []
        if len(aug_imgs)==0:
            for i in range(1):
                aug_x = tuple([self.dropout(x_i) for x_i in x])
                aug_x_list.append(aug_x)
        else:
            for aug_img in aug_imgs:
                # aug_img = torch.flip(aug_img, [3])
                aug_x = self.extract_feat(aug_img)
                aug_x_list.append(aug_x)

        combine_cls_scores, combine_bbox_preds = [cls_score], [bbox_pred]
        for aug_x in aug_x_list:
            aug_cls_score, aug_bbox_pred = self.bbox_head(aug_x)
            combine_cls_scores.append(aug_cls_score)
            combine_bbox_preds.append(aug_bbox_pred)

        teacher_cls_scores, teacher_bbox_preds = [], []

        for combine_cls_score in zip(*combine_cls_scores):
            tmp = sum(combine_cls_score) / len(combine_cls_score)
            teacher_cls_scores.append(tmp.detach())
        for combine_bbox_pred in zip(*combine_bbox_preds):
            tmp = sum(combine_bbox_pred) / len(combine_bbox_pred)
            teacher_bbox_preds.append(tmp.detach())   
         
        # for i in range(len(combine_cls_scores)):
        #     tmp = combine_cls_scores[i]
        #     teacher_cls_scores.append([t.detach() for t in tmp])
        # for i in range(len(combine_bbox_preds)):
        #     tmp = combine_bbox_preds[i]
        #     teacher_bbox_preds.append([t.detach() for t in tmp])

        losses = self.bbox_head.loss(cls_score, bbox_pred, gt_bboxes, gt_labels, img_metas, 
                                    class_splits=class_splits, img_splits=img_splits,
                                    cur_iter=cur_iter, max_iters=max_iters, 
                                    gt_bboxes_ignore=gt_bboxes_ignore)

        for i in range(len(combine_cls_scores)):
            cr_losses = self.bbox_head.cr_loss(combine_cls_scores[i], combine_bbox_preds[i], gt_bboxes, gt_labels, img_metas, 
                                            teacher_cls_scores, teacher_bbox_preds,
                                            class_splits=class_splits, img_splits=img_splits, 
                                            cur_iter=cur_iter, max_iters=max_iters, n_aug=2., 
                                            gt_bboxes_ignore=gt_bboxes_ignore)
            losses['cr_loss_cls_%d'%(i)] = cr_losses.pop('cr_loss_cls')
            losses['cr_loss_bbox_%d'%(i)] = cr_losses.pop('cr_loss_bbox')

        # just for debug
        # loss_inputs = (cls_score, bbox_pred, gt_bboxes, gt_labels, img_metas,
        #             combine_cls_scores[1], combine_bbox_preds[1], None, None, None, epoch)
        # losses2 = self.bbox_head.loss2(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        return losses
