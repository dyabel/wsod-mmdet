import torch

from ..builder import HEADS, build_head, build_roi_extractor
from .wsod_head import WsodHead
from mmdet.models.losses import accuracy
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
import wandb

@HEADS.register_module()
class WsodContrastHead(WsodHead):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 contrast_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WsodContrastHead,self).__init__(
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg)
        self.init_contrast_head(contrast_head)

    def init_contrast_head(self,contrast_head):
        self.contrast_head = build_head(contrast_head)

    def _bbox_forward_train_branch2_pass(self, x, sampling_results, gt_bboxes, gt_labels,
                                         img_metas,gt_bboxes_ignore=None,oam_confidence=3):
        """Run forward function and calculate loss for box head in training."""

        x_strong = tuple([torch.unsqueeze(xx[0],0) for xx in x])
        x_weak = tuple([torch.unsqueeze(xx[1],0) for xx in x])

        rois_strong = bbox2roi([res.bboxes for res in [sampling_results[0]]])
        rois_weak = bbox2roi([res.bboxes for res in [sampling_results[1]]])

        bbox_feats_strong = self.bbox_roi_extractor(
            x_strong[:self.bbox_roi_extractor.num_inputs], rois_strong)
        bbox_feats_strong = self.bbox_head.double_fc_forward(bbox_feats_strong)

        bbox_feats_weak = self.bbox_roi_extractor(
            x_weak[:self.bbox_roi_extractor.num_inputs], rois_weak)
        bbox_feats_weak = self.bbox_head.double_fc_forward(bbox_feats_weak)

        if self.with_shared_head:
            bbox_feats_strong = self.shared_head(bbox_feats_strong)
            bbox_feats_weak = self.shared_head(bbox_feats_weak)

        bbox_targets_strong_branch2 = self.bbox_head.get_targets([sampling_results[0]], [gt_bboxes[0]],
                                                                 [gt_labels[0]], self.train_cfg)


        #TODO contrastive_losses
        contrastive_losses = self.contrast_forward_train(x,gt_bboxes[0],gt_labels[0],gt_bboxes[1],
                                                         gt_labels[1],img_metas,gt_bboxes_ignore=gt_bboxes_ignore)
        #calculate loss_strong_branch2
        bbox_results_strong_branch2 = self._bbox_forward_strong_branch2(bbox_feats_strong)
        loss_bbox_strong_branch2 = self.bbox_head.loss_strong(bbox_results_strong_branch2['cls_score'],
                                                              bbox_results_strong_branch2['bbox_pred'],
                                                              rois_strong,
                                                              *bbox_targets_strong_branch2)
        loss_strong_branch2 = dict()
        loss_strong_branch2['loss_cls_strong_branch2'] = loss_bbox_strong_branch2['loss_cls_strong']
        loss_strong_branch2['acc_strong_branch2'] = loss_bbox_strong_branch2['acc_strong']
        loss_strong_branch2['loss_bbox_strong_branch2'] = loss_bbox_strong_branch2['loss_bbox_strong']
        bbox_results_strong_branch2.update(loss_bbox_strong_branch2=loss_strong_branch2)
        #calculate loss_weak_branch2
        bbox_results_weak_branch2 = self._bbox_forward_strong_branch2(bbox_feats_weak)

        bbox_targets_weak_branch2 = self.bbox_head.get_targets([sampling_results[1]],[gt_bboxes[1]],[gt_labels[1]],self.train_cfg)
        labels,label_weights,bbox_targets,bbox_weights = bbox_targets_weak_branch2
        label_weights = label_weights.new_ones(label_weights.size(0))
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        acc_weak = accuracy(bbox_results_weak_branch2['cls_score'],labels)
        loss_bbox_weak_branch2 = dict()
        if oam_confidence>self.ss_cf_thr:
            loss_bbox_weak_branch2['loss_cls_weak_branch2'] = self.bbox_head.loss_cls(bbox_results_weak_branch2['cls_score'],
                                                                                      labels,
                                                                                      label_weights,
                                                                                      avg_factor=avg_factor,
                                                                                      reduction_override=None)*0
        else:
            loss_bbox_weak_branch2['loss_cls_weak_branch2'] = self.bbox_head.loss_cls(bbox_results_weak_branch2['cls_score'],
                                                                                      labels,
                                                                                      label_weights,
                                                                                      avg_factor=avg_factor,
                                                                                      reduction_override=None)/(oam_confidence-2)/self.oam_discount
        loss_bbox_weak_branch2['acc_weak_branch2'] = acc_weak
        # bbox_results_weak_branch2.update(loss_bbox_weak_branch2=loss_weak_branch2)
        bbox_results_weak_branch2.update(loss_bbox_weak_branch2=loss_bbox_weak_branch2)

        return  bbox_results_weak_branch2,bbox_results_strong_branch2,contrastive_losses

    def forward_train_branch2(self,
                              x,
                              img_metas,
                              proposal_list,
                              gt_bboxes,
                              gt_labels,
                              oam_confidence=3,
                              gt_bboxes_ignore=None,
                              gt_masks=None):
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            assert num_imgs == 2
            #assign for strong image
            assign_result = self.bbox_assigner.assign(
                proposal_list[0], gt_bboxes[0], gt_bboxes_ignore[0],
                gt_labels[0])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[0],
                gt_bboxes[0],
                gt_labels[0],
                feats=[lvl_feat[0][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

            #assign for weak image
            assign_result = self.bbox_assigner.assign(
                proposal_list[1], gt_bboxes[1][:,:4], gt_bboxes_ignore[1],
                gt_labels=gt_labels[1])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[1],
                gt_bboxes[1],
                gt_labels=gt_labels[1],
                feats=[lvl_feat[1][None] for lvl_feat in x])

            sampling_results.append(sampling_result)
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results_weak_branch2, bbox_results_strong_branch2,contrastive_losses = \
                self._bbox_forward_train_branch2_pass(x,
                                                      sampling_results,
                                                      gt_bboxes,
                                                      gt_labels,
                                                      img_metas,
                                                      gt_bboxes_ignore=gt_bboxes_ignore,
                                                      oam_confidence=oam_confidence)

            losses.update(bbox_results_weak_branch2['loss_bbox_weak_branch2'])
            losses.update(bbox_results_strong_branch2['loss_bbox_strong_branch2'])
            losses.update(contrastive_losses)
            return losses
    #duyu
    def contrast_forward_train(self,
                               x,
                               strong_bboxes,
                               strong_labels,
                               oam_bboxes,
                               oam_labels,
                               img_metas,
                               gt_bboxes_ignore=None,
                               ):
        """
        Args:
        :param x:
        :param strong_bboxes:
        :param strong_labels:
        :param oam_bboxes:
        :param oam_labels:
        :param img_metas:
        :param gt_bboxes_ignore:
        :param gggg: 222
        :return:
        """
        if self.iters < wandb.config.warm_iter:
            losses = dict()
            losses['contrastive_loss'] = strong_labels.new_zeros(1).type(torch.float)
            return losses

        torch_device = strong_labels.get_device()
        oam_labels = oam_labels.to(torch_device)
        x_strong = tuple([torch.unsqueeze(xx[0], 0) for xx in x])
        x_weak = tuple([torch.unsqueeze(xx[1], 0) for xx in x])

        rois_strong = bbox2roi([strong_bboxes])
        rois_weak = bbox2roi([oam_bboxes])

        bbox_feats_strong = self.bbox_roi_extractor(
            x_strong[:self.bbox_roi_extractor.num_inputs], rois_strong)
        bbox_feats_strong = self.bbox_head.double_fc_forward(bbox_feats_strong)

        bbox_feats_weak = self.bbox_roi_extractor(
            x_weak[:self.bbox_roi_extractor.num_inputs], rois_weak)
        bbox_feats_weak = self.bbox_head.double_fc_forward(bbox_feats_weak)

        if self.with_shared_head:
            bbox_feats_strong = self.shared_head(bbox_feats_strong)
            bbox_feats_weak = self.shared_head(bbox_feats_weak)

        # print(bbox_feats_strong.size(),labels_strong.size())
        contrastive_losses = self.contrast_head.forward_train(bbox_feats_strong,bbox_feats_weak,strong_labels,oam_labels)
        # print(contrastive_losses)
        # losses = dict()
        # losses.update(contrastive_losses)
        # print(losses)
        return contrastive_losses
        # losses['contrastive_loss'] = torch.tensor([0.0])
