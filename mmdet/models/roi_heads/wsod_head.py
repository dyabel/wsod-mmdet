import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import multiclass_nms,bbox_select_per_class,first_pass_filter
from mmdet.models.losses import accuracy
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from mmdet.core.utils import convert_label
from mmdet.utils import visualize_oam_boxes,iou
import wandb
import time

@HEADS.register_module()
class WsodHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
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
        super(WsodHead,self).__init__(
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg)
        self.iters = 0
        self.ss_cf_thr = wandb.config.ss_cf_thr
        self.oam_max_num = wandb.config.oam_max_num
        self.score_thr2 = wandb.config.score_thr2
        self.oam_discount = 1
        self.nms = 0.5
        # self.init_contrast_head(contrast_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)
            self.second_pass_sampler = build_sampler(
                self.train_cfg.second_pass_sampler,context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_contrast_head(self,contrast_head):
        self.contrast_head = build_head(contrast_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = self.bbox_head.double_fc_forward(bbox_feats)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        if self.with_bbox:
            bbox_results = self._bbox_forward_strong_branch2(bbox_feats)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def match(self,bboxes1=None,bboxes2=None,labels1=None,labels2=None):
        labels1 = labels1.detach().cpu()
        labels2 = labels2.detach().cpu()
        if len(labels1)!=len(labels2):
            return False
        if (len(torch.bincount(labels1))!=len(torch.bincount(labels2))):
            return False
        if (torch.bincount(labels1)!=torch.bincount(labels2)).any():
            return False
        labels1,idx1 = torch.sort(labels1)
        labels2,idx2 = torch.sort(labels2)
        bboxes1 = bboxes1[idx1]
        bboxes2 = bboxes2[idx2]
        if (torch.abs(bboxes1 - bboxes2) > 5).any():
            return False
        # print(bboxes1[0].unsqueeze(0),bboxes2[0].unsqueeze(0))
        # print(labels1,labels2)
        # matched = []
        # for i,ii in enumerate(labels1):
        #     flag = False
        #     for j in torch.where(labels2==ii)[0]:
        #         if j in matched:
        #             continue
        #         if iou(bboxes1[i].unsqueeze(0),bboxes2[j].unsqueeze(0))[0][0]>0.6:
        #             matched.append(j)
        #             flag = True
        #             break
        #     if not flag: return False
        # if len(matched)!=len(labels2):
        #     return False
        return True


    @torch.no_grad()
    def OAM_Confidence(self,
                       x,
                       img_metas,
                       bboxes,
                       labels,
                       img_level_label,
                       max_iter=30,
                        ):
        oam_bboxes, oam_labels = bboxes,labels
        if bboxes.size(0) == 0:
            # print('empty oam')
            return wandb.config.empty_cf,[bboxes],[labels]

        # begin iter
        k = 0
        T = max_iter
        count = 0
        while k < max_iter:
            k += 1
            if oam_bboxes.size(0) == 0:
                return wandb.config.empty_cf,[oam_bboxes],[oam_labels]
            oam_bboxes_next,oam_labels_next=self.oam_forward(x, oam_bboxes, img_level_label, img_metas)
            if self.match(bboxes1=oam_bboxes_next, bboxes2=oam_bboxes, labels1=oam_labels_next,
                          labels2=oam_labels):
                count += 1
                if count == 3:
                    T = k
                    k = max_iter + 1
                    break
            else:
                count = 0
            oam_bboxes, oam_labels = oam_bboxes_next, oam_labels_next
        if T==max_iter:
            T = 10000
        return T,[oam_bboxes],[oam_labels]


    #duyu
    @torch.no_grad()
    def oam_forward(self,x,oam_bboxes,img_level_label,img_metas):
        rois = bbox2roi([oam_bboxes])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = self.bbox_head.double_fc_forward(bbox_feats)
        bbox_results = self._bbox_forward_strong_branch1(bbox_feats)
        img_shape = img_metas[0]['img_shape']
        scale_factors = img_metas[0]['scale_factor']
        oam_bboxes, oam_labels = self.bbox_head.compute_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_level_label,
            img_shape,
            scale_factors,
            rescale=False,
            cfg=self.test_cfg)
        return oam_bboxes[:,:4],oam_labels

    #duyu
    def forward_train(self,
                              x,
                              img,
                              img_metas,
                              proposal_list,
                              gt_bboxes,
                              gt_labels,
                              gt_bboxes_ignore=None,
                              gt_masks=None):
        img_level_label = gt_labels[1]
        #branch1
        losses_branch1,oam_bboxes,oam_labels = self.forward_train_branch1(x,img_metas,proposal_list,gt_bboxes,gt_labels,gt_bboxes_ignore,
                                                       gt_masks=None)

        self.iters += 1
        # if self.iters % 1000 == 0 and self.iters>wandb.config.warm_iter:
            # self.ss_cf_thr -= 1
            # self.ss_cf_thr = max(6,self.ss_cf_thr)
            # self.oam_max_num += 1
            # self.score_thr2 += 0.01
            # self.score_thr2 = min(self.score_thr2,0.7)
            # self.oam_discount -= 0.5
            # self.oam_discount = max(1,self.oam_discount)
            # self.oam_max_num = min(self.oam_max_num,20)
            # print('nms:',self.nms)
            # print('oam_discount:',self.oam_discount)
            # print('ss_cf_thr:',self.ss_cf_thr)
            # print('oam_max_num:', self.oam_max_num)
        if self.iters < wandb.config.warm_iter:
                oam_confidence = 100
        else:
            x_weak = tuple([torch.unsqueeze(xx[1], 0) for xx in x])
            oam_confidence,oam_bboxes,oam_labels = self.OAM_Confidence(x_weak,
                                                     img_metas,
                                                     oam_bboxes[0][:,:4],
                                                     oam_labels[0],
                                                     img_level_label,
                                                     max_iter=wandb.config.empty_cf,
                                                     )
        torch_device = gt_labels[0].get_device()
        gt_bboxes[1] = oam_bboxes[0].to(torch_device)
        gt_labels[1] = oam_labels[0].to(torch_device)
        # if oam_confidence>self.ss_cf_thr:
        #     oam_confidence = 100
        if oam_confidence<wandb.config.ss_cf_thr or self.iters % 50 == 0 :
                visualize_oam_boxes(oam_bboxes[0][:,:4],oam_labels[0],img[1],img_metas,
                                win_name='T= %d E = %d'%(oam_confidence,self.iters//566+5),show=False,
                                out_dir='../work_dirs/oam_bboxes3/',show_score_thr=0)
        losses_branch2 = self.forward_train_branch2(x,
                                                    img_metas,
                                                    proposal_list,
                                                    gt_bboxes,
                                                    gt_labels,
                                                    gt_bboxes_ignore=gt_bboxes_ignore,
                                                    oam_confidence=oam_confidence,
                                                    gt_masks=None)
        losses = dict()
        losses.update(losses_branch1)
        losses.update(losses_branch2)
        return losses,oam_bboxes[0],oam_labels[0],oam_confidence
    #duyu
    def forward_train_branch1(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
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
                proposal_list[1], gt_bboxes[1], gt_bboxes_ignore[1],
                gt_labels=None)
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[1],
                gt_bboxes[1],
                gt_labels=None,
                feats=[lvl_feat[1][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results_strong,bbox_results_weak,oam_bboxes,oam_labels = \
                                                        self._bbox_forward_train_branch1_fp(x,sampling_results,
                                                                                        gt_bboxes, gt_labels,
                                                                                        img_metas)

            losses.update(bbox_results_strong['loss_bbox_strong_fp'])
            losses.update(bbox_results_weak['loss_bbox_weak_fp'])

        proposal_list = oam_bboxes
        gt_bboxes[1] = oam_bboxes[1]

        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            assert num_imgs == 2
            # assign for strong image
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

            # assign for weak image
            assign_result = self.bbox_assigner.assign(
                proposal_list[1], gt_bboxes[1], gt_bboxes_ignore[1],
                gt_labels=None)
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[1],
                gt_bboxes[1],
                gt_labels=None,
                feats=[lvl_feat[1][None] for lvl_feat in x])

            sampling_results.append(sampling_result)

        if self.with_bbox:
            bbox_results_strong,bbox_results_weak,oam_bboxes,oam_labels = \
                self._bbox_forward_train_branch1_sp(x,sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results_strong['loss_bbox_strong_sp'])
            losses.update(bbox_results_weak['loss_bbox_weak_sp'])

        return losses,oam_bboxes,oam_labels


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
            bbox_results_weak_branch2, bbox_results_strong_branch2 = \
                                                                    self._bbox_forward_train_branch2_pass(x,
                                                                                                          sampling_results,
                                                                                                          gt_bboxes,
                                                                                                          gt_labels,
                                                                                                          img_metas,
                                                                                                          gt_bboxes_ignore=gt_bboxes_ignore,
                                                                                                          oam_confidence=oam_confidence)

            losses.update(bbox_results_weak_branch2['loss_bbox_weak_branch2'])
            losses.update(bbox_results_strong_branch2['loss_bbox_strong_branch2'])
            # losses.update(contrastive_losses)
        return losses
    #duyu
    def _bbox_forward_strong_branch1(self,bbox_feats):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        cls_score, bbox_pred = self.bbox_head.forward_strong_branch1(bbox_feats)
        bbox_results = dict(
            cls_score = cls_score, bbox_pred = bbox_pred, bbox_feats = bbox_feats)
        return bbox_results
    #duyu
    def _bbox_forward_strong_branch2(self,bbox_feats):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        cls_score, bbox_pred = self.bbox_head.forward_strong_branch2(bbox_feats)
        bbox_results = dict(
            cls_score = cls_score, bbox_pred = bbox_pred, bbox_feats = bbox_feats)
        return bbox_results
    #duyu
    def _bbox_forward_weak(self,bbox_feats):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        cls_proposal_mat = self.bbox_head.forward_weak(bbox_feats)
        bbox_results = dict(
            cls_proposal_mat = cls_proposal_mat, bbox_feats=bbox_feats)
        return bbox_results
    #duyu
    def _bbox_forward_train_branch1_fp(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""

        torch_device = gt_labels[0].get_device()

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
        #caculate loss_strong_branch1
        bbox_targets_strong = self.bbox_head.get_targets([sampling_results[0]], [gt_bboxes[0]],
                                                  [gt_labels[0]], self.train_cfg)
        bbox_results_strong = self._bbox_forward_strong_branch1(bbox_feats_strong)
        loss_bbox_strong = self.bbox_head.loss_strong(bbox_results_strong['cls_score'],
                                                      bbox_results_strong['bbox_pred'], rois_strong,
                                                      *bbox_targets_strong)
        loss_strong = dict()
        loss_strong['loss_cls_strong_branch1_fp'] = loss_bbox_strong['loss_cls_strong']
        loss_strong['acc_strong_branch1_fp'] = loss_bbox_strong['acc_strong']
        loss_strong['loss_bbox_strong_branch1_fp'] = loss_bbox_strong['loss_bbox_strong']
        bbox_results_strong.update(loss_bbox_strong_fp=loss_strong)
        img_level_label_for_strong,_ = convert_label(gt_labels[0],gt_labels[1].size(0))
        img_shape = img_metas[0]['img_shape']
        scale_factors = img_metas[0]['scale_factor']

        bboxes_strong, _ = self.bbox_head.compute_bboxes(
                rois_strong,
                bbox_results_strong['cls_score'],
                bbox_results_strong['bbox_pred'],
                img_level_label_for_strong,
                img_shape,
                scale_factors,
                rescale=False,
                cfg=None)
        assert len(bboxes_strong)==bbox_results_strong['cls_score'].size(0)

        oam_bboxes_strong,oam_labels_strong = first_pass_filter(bboxes_strong,bbox_results_strong['cls_score'],img_level_label_for_strong)

        bbox_results_weak = self._bbox_forward_weak(bbox_feats_weak)
        bbox_results_weak_pseudo = self._bbox_forward_strong_branch1(bbox_feats_weak)

        loss_bbox_weak = self.bbox_head.loss_weak(bbox_results_weak['cls_proposal_mat'],
                                                  gt_labels[1])
        loss_weak = dict()
        loss_weak['loss_img_level_fp'] = loss_bbox_weak['loss_img_level']
        bbox_results_weak.update(loss_bbox_weak_fp=loss_weak)
        bboxes_weak, _ = self.bbox_head.compute_bboxes(
            rois_weak,
            bbox_results_weak_pseudo['cls_score'],
            bbox_results_weak_pseudo['bbox_pred'],
            gt_labels[1],
            img_shape,
            scale_factors,
            rescale=False,
            cfg=None)
        assert len(bboxes_weak)==bbox_results_weak_pseudo['cls_score'].size(0)

        oam_bboxes_weak,oam_labels_weak = first_pass_filter(bboxes_weak,bbox_results_weak_pseudo['cls_score'],gt_labels[1])

        oam_bboxes = []
        oam_labels = []
        oam_bboxes.append(oam_bboxes_strong[:,:4])
        oam_bboxes.append(oam_bboxes_weak[:,:4])
        oam_labels.append(oam_labels_strong.to(torch_device))
        oam_labels.append(oam_labels_weak.to(torch_device))
        return bbox_results_strong,bbox_results_weak,oam_bboxes,oam_labels

    def _bbox_forward_train_branch1_sp(self, x, sampling_results, gt_bboxes, gt_labels,
                                       img_metas):
        """Run forward function and calculate loss for box head in training."""

        torch_device = gt_labels[0].get_device()

        x_strong = tuple([torch.unsqueeze(xx[0], 0) for xx in x])
        x_weak = tuple([torch.unsqueeze(xx[1], 0) for xx in x])

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
        # caculate loss_strong_branch1
        bbox_targets_strong = self.bbox_head.get_targets([sampling_results[0]], [gt_bboxes[0]],
                                                         [gt_labels[0]], self.train_cfg)
        bbox_results_strong = self._bbox_forward_strong_branch1(bbox_feats_strong)
        loss_bbox_strong = self.bbox_head.loss_strong(bbox_results_strong['cls_score'],
                                                      bbox_results_strong['bbox_pred'], rois_strong,
                                                      *bbox_targets_strong)
        loss_strong = dict()
        loss_strong['loss_cls_strong_branch1_sp'] = loss_bbox_strong['loss_cls_strong']
        loss_strong['acc_strong_branch1_sp'] = loss_bbox_strong['acc_strong']
        loss_strong['loss_bbox_strong_branch1_sp'] = loss_bbox_strong['loss_bbox_strong']
        bbox_results_strong.update(loss_bbox_strong_sp=loss_strong)

        bbox_results_weak = self._bbox_forward_weak(bbox_feats_weak)
        bbox_results_weak_pseudo = self._bbox_forward_strong_branch1(bbox_feats_weak)

        loss_bbox_weak = self.bbox_head.loss_weak(bbox_results_weak['cls_proposal_mat'],
                                                  gt_labels[1])
        loss_weak = dict()
        loss_weak['loss_img_level_sp'] = loss_bbox_weak['loss_img_level']
        bbox_results_weak.update(loss_bbox_weak_sp=loss_weak)

        img_shape = img_metas[0]['img_shape']
        scale_factors = img_metas[0]['scale_factor']
        assert len(rois_weak)==bbox_results_weak_pseudo['bbox_pred'].size(0)
        bboxes_weak, _ = self.bbox_head.compute_bboxes(
            rois_weak,
            bbox_results_weak_pseudo['cls_score'],
            bbox_results_weak_pseudo['bbox_pred'],
            gt_labels[1],
            img_shape,
            scale_factors,
            rescale=False,
            cfg=None)
        oam_bboxes_weak, oam_labels_weak = bbox_select_per_class(bboxes_weak,
                                                          bbox_results_weak_pseudo['cls_score'],
                                                          gt_labels[1],
                                                          score_thr=self.score_thr2,
                                                          nms_cfg={'iou_threshold': self.nms},
                                                          max_num=self.oam_max_num,
                                                          )

        oam_bboxes = []
        oam_labels = []
        oam_bboxes.append(oam_bboxes_weak)
        oam_labels.append(oam_labels_weak.to(torch_device))
        return bbox_results_strong, bbox_results_weak, oam_bboxes, oam_labels
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
        torch_device = strong_labels.get_device()
        oam_labels = oam_labels.to(torch_device)
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            assert num_imgs == 2
            # assign for strong image
            assign_result = self.bbox_assigner.assign(
                strong_bboxes, strong_bboxes, gt_bboxes_ignore[0],
                strong_labels)
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                strong_bboxes,
                strong_bboxes,
                strong_labels,
                feats=[lvl_feat[0][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

            # assign for weak image
            # print(strong_bboxes.size(),oam_bboxes.size())
            assign_result = self.bbox_assigner.assign(
                oam_bboxes, oam_bboxes, gt_bboxes_ignore[1],
                oam_labels)
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                oam_bboxes,
                oam_bboxes,
                oam_labels,
                feats=[lvl_feat[1][None] for lvl_feat in x])

            sampling_results.append(sampling_result)
        x_strong = tuple([torch.unsqueeze(xx[0], 0) for xx in x])
        x_weak = tuple([torch.unsqueeze(xx[1], 0) for xx in x])

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

        labels_strong,_,_,_ = self.bbox_head.get_targets([sampling_results[0]], [strong_bboxes],
                                                         [strong_labels], self.train_cfg)
        labels_weak,_,_,_ = self.bbox_head.get_targets([sampling_results[1]], [oam_bboxes],
                                                         [oam_labels], self.train_cfg)
        # print(bbox_feats_strong.size(),labels_strong.size())
        # contrastive_losses = self.contrast_head.forward_train(bbox_feats_strong,bbox_feats_weak,labels_strong,labels_weak)
        losses = dict()
        # losses['contrastive_loss'] = contrastive_losses
        losses['contrastive_loss'] = torch.tensor([0.0])
        return losses

    #duyu
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
        # contrastive_losses = self.contrast_forward_train(x,gt_bboxes[0],gt_labels[0],oam_bboxes_weak[:,:4],
        #                                                  oam_labels_weak,img_metas,gt_bboxes_ignore=gt_bboxes_ignore)
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
            # loss_bbox_weak_branch2 = self.bbox_head.loss_strong(bbox_results_weak_branch2['cls_score'],
            #                                                     bbox_results_weak_branch2['bbox_pred'],
            #                                                     rois_weak,
            #                                                     *bbox_targets_weak_branch2)
            # loss_weak_branch2 = dict()
            # loss_weak_branch2['loss_cls_weak_branch2'] = loss_bbox_weak_branch2['loss_cls_strong']*0
            # loss_weak_branch2['loss_bbox_weak_branch2'] = loss_bbox_weak_branch2['loss_bbox_strong']*0
        else:
            loss_bbox_weak_branch2['loss_cls_weak_branch2'] = self.bbox_head.loss_cls(bbox_results_weak_branch2['cls_score'],
                                                                                  labels,
                                                                                  label_weights,
                                                                                  avg_factor=avg_factor,
                                                                                  reduction_override=None)/(oam_confidence-2)/self.oam_discount
            # loss_bbox_weak_branch2 = self.bbox_head.loss_strong(bbox_results_weak_branch2['cls_score'],
            #                                                       bbox_results_weak_branch2['bbox_pred'],
            #                                                       rois_weak,
            #                                                       *bbox_targets_weak_branch2)
            # loss_weak_branch2 = dict()
            # loss_weak_branch2['loss_cls_weak_branch2'] = loss_bbox_weak_branch2['loss_cls_strong']
            # loss_weak_branch2['loss_bbox_weak_branch2'] = loss_bbox_weak_branch2['loss_bbox_strong']
        loss_bbox_weak_branch2['acc_weak_branch2'] = acc_weak
        # bbox_results_weak_branch2.update(loss_bbox_weak_branch2=loss_weak_branch2)
        bbox_results_weak_branch2.update(loss_bbox_weak_branch2=loss_bbox_weak_branch2)

        return  bbox_results_weak_branch2,bbox_results_strong_branch2
               # contrastive_losses

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        print('async_simple_test')
        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = self.bbox_head.double_fc_forward(bbox_feats)


        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        bbox_results = self._bbox_forward_strong_branch2(bbox_feats)

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        # print(det_labels)
        return det_bboxes, det_labels

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]


    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            bbox_feats = self.bbox_head.double_fc_forward(bbox_feats)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            bbox_results = self._bbox_forward_strong_branch2(bbox_feats)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels