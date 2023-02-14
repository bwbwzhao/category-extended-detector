import torch.nn as nn
import torch

from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from ..builder import HEADS
from .anchor_head import AnchorHead

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        force_fp32, images_to_levels, multi_apply,
                        multiclass_nms, unmap)
from ..builder import build_loss
import math


@HEADS.register_module()
class RetinaHead_CaEx(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 aux_thre=0.5,
                 norm_thre=0.5,
                 stage=None,
                 cls_strategy=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.aux_thre = aux_thre
        self.norm_thre = norm_thre
        self.stage = stage
        self.cls_strategy = cls_strategy
        self.strict_assigner = build_assigner(
            dict(type='MaxIoUAssigner',
                pos_iou_thr=0.9,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1)
        )
        super(RetinaHead_CaEx, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.dropout = nn.Dropout(p=0.2)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single_multidrop(self, x, test_runs):
        cls_feat = x
        reg_feat = x  
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # HEAD Dropout
        cls_shape, reg_shape = cls_feat.shape, reg_feat.shape
        cls_feat = cls_feat.expand(cls_shape[0]*test_runs, cls_shape[1], cls_shape[2], cls_shape[3])
        reg_feat = reg_feat.expand(reg_shape[0]*test_runs, reg_shape[1], reg_shape[2], reg_shape[3])
        cls_score = self.retina_cls(self.dropout(cls_feat))
        bbox_pred = self.retina_reg(self.dropout(reg_feat))
        
        return cls_score, bbox_pred

    def forward_multidrop(self, feats, test_runs):
        return multi_apply(self.forward_single_multidrop, feats, test_runs=test_runs)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, labels_aux, label_weights,
                    bbox_targets, bbox_weights, max_overlaps, avg_factor, img_splits=None, class_splits=None):
        # classification loss
        loss_cls = 0.
        for img_split, class_split in zip(img_splits, class_splits):
            if len(img_split) == 0:
                continue

            labels_i = labels[img_split].reshape(-1)
            labels_aux_i = labels_aux[img_split].reshape(-1)
            max_overlaps_i = max_overlaps[img_split].reshape(-1)
            label_weights_i = label_weights[img_split].reshape(-1)

            cls_score_i = cls_score[img_split].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

            tmp = self.loss_cls(cls_score_i, labels_i, label_weights_i, avg_factor=1.)
            
            if self.stage=='one':
                if self.cls_strategy=='plain':
                    pass
                elif self.cls_strategy=='dataset_aware':
                    bg_index = (labels_i==self.cls_out_channels)
                    tmp[bg_index, :class_split[0]] = 0.
                    tmp[bg_index, class_split[1]:] = 0.
                elif self.cls_strategy=='conflict_free':
                    bg_index_strict = (labels_aux_i==self.cls_out_channels)
                    tmp[bg_index_strict, :class_split[0]] = 0.
                    tmp[bg_index_strict, class_split[1]:] = 0.
            elif self.stage=='two':
                if self.cls_strategy=='as_full':
                    pass
                elif self.cls_strategy=='pos_only':
                    bg_index_strict = (labels_aux_i==self.cls_out_channels)
                    tmp[bg_index_strict, :class_split[0]] = 0.
                    tmp[bg_index_strict, class_split[1]:] = 0.
                elif self.cls_strategy=='pos_safeneg':
                    bg_index_hp = (labels_i==self.cls_out_channels) # under high precision pseudo annotations
                    bg_index_hr = (labels_aux_i==self.cls_out_channels) # under high recall pseudo annotations
                    bg_index_unsafe = (bg_index_hp!=bg_index_hr) # unsafe background
                    tmp[bg_index_unsafe, :class_split[0]] = 0.
                    tmp[bg_index_unsafe, class_split[1]:] = 0.
                elif self.cls_strategy=='pos_weightedneg':
                    a, b, c = 1.0, 10000., 25.
                    iou_weight = a * torch.exp(-b * torch.exp(-c * max_overlaps_i)) # overlap based weighting schedule
                    bg_index_strict = (labels_aux_i==self.cls_out_channels)
                    tmp[bg_index_strict, :class_split[0]] = tmp[bg_index_strict, :class_split[0]] * (iou_weight[bg_index_strict].reshape(-1, 1))
                    tmp[bg_index_strict, class_split[1]:] = tmp[bg_index_strict, class_split[1]:] * (iou_weight[bg_index_strict].reshape(-1, 1))

            tmp = tmp.sum() / avg_factor
            loss_cls += tmp
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=avg_factor)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             gt_confidences,
             img_metas,
             gt_bboxes_ignore=None,
             cur_iter=None, 
             max_iters=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # aux bboxes        
        gt_labels_aux, gt_bboxes_aux = [], []
        for i, gt_confidence in enumerate(gt_confidences):
            aux_index = gt_confidence > self.aux_thre 
            gt_labels_aux.append(gt_labels[i][aux_index])
            gt_bboxes_aux.append(gt_bboxes[i][aux_index])
        aux_assigner = self.assigner \
            if self.stage=='two' and self.cls_strategy=='pos_safeneg' \
            else self.strict_assigner 
        cls_reg_targets_aux = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes_aux,
            img_metas,
            aux_assigner,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels_aux,
            label_channels=label_channels)
        if cls_reg_targets_aux is None:
            return None
        (labels_list_aux, label_weights_list_aux, bbox_targets_list_aux, bbox_weights_list_aux,
         num_total_pos_aux, num_total_neg_aux, max_overlaps_list_aux,) = cls_reg_targets_aux
        # normal bboxes
        gt_labels_norm, gt_bboxes_norm = [], []
        for i, gt_confidence in enumerate(gt_confidences):
            norm_index = gt_confidence > self.norm_thre
            gt_labels_norm.append(gt_labels[i][norm_index])
            gt_bboxes_norm.append(gt_bboxes[i][norm_index])
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes_norm,
            img_metas,
            self.assigner,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels_norm,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, max_overlaps_list,) = cls_reg_targets # bbox_uncers_list no effect

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos) # in focal loss, self.sampling=False

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        self.loss_normalizer = (self.loss_normalizer_momentum * self.loss_normalizer + 
                                (1. - self.loss_normalizer_momentum) * num_total_samples)

        img_splits = [[] for set_s in self.set_splits]
        for i, img_meta in enumerate(img_metas):
            ind = self.set_splits.index(img_meta['class_set'])
            img_splits[ind].append(i)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            labels_list_aux,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            max_overlaps_list,
            avg_factor=max(self.loss_normalizer, 1),
            img_splits=img_splits,
            class_splits=self.class_splits)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True,
                            assigner=None):
        """Compute regression and classification targets for anchors in
            a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 6
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        # assign_result.max_overlaps
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            max_overlaps = unmap(assign_result.max_overlaps, num_total_anchors, inside_flags)

            return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                    neg_inds, sampling_result, max_overlaps)
        
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
        neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    assigner,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
            multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end

        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
            assigner=assigner)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)
