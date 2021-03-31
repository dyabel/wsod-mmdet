import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import multiclass_nms,bbox_select_per_class
from mmdet.core.evaluation import bbox_overlaps
from mmdet.models.losses import accuracy
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
import time
@HEADS.register_module()
class WsodEmbedHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
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
        super(WsodEmbedHead,self).__init__(
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg)
        # self.init_contrast_head(contrast_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    # def init_embedding_head(self,embedding_head):
    #     self.embedding_head = build_head(embedding_head)

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
        flag = False
        for i,box1 in enumerate(bboxes1):
            if labels1[i] != labels2[i]:
                return False
            for box2 in bboxes2:
                if bbox_overlaps(box1.unsqueeze(0).cpu().numpy(),box2.unsqueeze(0).cpu().numpy())[0][0] > 0.5:
                    flag = True
                    break
            if not flag:  return False
        return True



    #duyu
    #TODO online augmentation
    @torch.no_grad()
    def OAM_Confidence(self,
                    x,
                    img_metas,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    max_iter=30,
                    gt_bboxes_ignore=None,
                    gt_masks=None):
        if not self.with_bbox:
            raise Exception
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        bbox_results, oam_bboxes, oam_labels = self._bbox_forward_train_strong(x, sampling_results,
                                                                               gt_bboxes, gt_labels,
                                                                               img_metas)
        oam_bboxes = [oam_bboxes[:, :4]]
        oam_labels = [oam_labels]
        #begin iter
        k = 0
        T = max_iter
        count = 0
        while k < max_iter:
            k += 1
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    oam_bboxes[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    oam_bboxes[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            bbox_results, oam_bboxes_next, oam_labels_next = self._bbox_forward_train_strong(x, sampling_results,
                                                                                   gt_bboxes, gt_labels,
                                                                                   img_metas)
            oam_bboxes_next = [oam_bboxes_next[:,:4]]
            oam_labels_next = [oam_labels_next]
            if self.match(bboxes1=oam_bboxes_next[0],bboxes2=oam_bboxes[0],labels1=oam_labels_next[0],labels2=oam_labels[0]):
                count += 1
                if count == 3:
                    T = k
                    k = max_iter + 1
                    break
            else:
                count = 0
            oam_bboxes,oam_labels = oam_bboxes_next,oam_labels_next
        return T
    #duyu
    #TODO Double pass seems not work well. Here the first pass is removed
    def forward_train(self,
                              x,
                              img_metas,
                              proposal_list,
                              gt_bboxes,
                              gt_labels,
                              gt_bboxes_ignore=None,
                              gt_masks=None):
        # losses_first_pass,oam_bboxes,oam_labels = self.forward_train_first_pass(x,img_metas,proposal_list,gt_bboxes,gt_labels,gt_bboxes_ignore,
        #                                                gt_masks=None)
        losses_second_pass = self.forward_train_second_pass(x,img_metas,proposal_list,gt_bboxes,gt_labels,gt_bboxes_ignore,
                                                                             gt_masks=None)
        losses = dict()
        # losses.update(losses_first_pass)
        losses.update(losses_second_pass)
        return losses
    #duyu
    def forward_train_first_pass(self,
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
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results_strong,bbox_results_weak,oam_bboxes,oam_labels = \
                                                        self._bbox_forward_train_first_pass(x,sampling_results,
                                                                                        gt_bboxes, gt_labels,
                                                                                        img_metas)

            losses.update(bbox_results_strong['loss_bbox_strong_fp'])
            losses.update(bbox_results_weak['loss_bbox_weak_fp'])

        # mask head forward and loss
        #TODO
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results_strong['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses,oam_bboxes,oam_labels


    def forward_train_second_pass(self,
                              x,
                              img_metas,
                              proposal_list,
                              gt_bboxes,
                              gt_labels,
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
                proposal_list[1], gt_bboxes[1], gt_bboxes_ignore[1],
                gt_labels=None)
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[1],
                gt_bboxes[1],
                gt_labels=None,
                feats=[lvl_feat[1][None] for lvl_feat in x])

            sampling_results.append(sampling_result)
            # print('#'*100)
            # print(sampling_results)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results_weak_branch1, bbox_results_strong_branch1, bbox_results_weak_branch2, bbox_results_strong_branch2 = \
                                                                    self._bbox_forward_train_second_pass(x, sampling_results,
                                                                                                     gt_bboxes, gt_labels,
                                                                                                     img_metas,gt_bboxes_ignore=gt_bboxes_ignore)

            losses.update(bbox_results_weak_branch1['loss_bbox_weak_branch1_sp'])
            losses.update(bbox_results_strong_branch1['loss_bbox_strong_branch1_sp'])
            losses.update(bbox_results_weak_branch2['loss_bbox_weak_branch2'])
            losses.update(bbox_results_strong_branch2['loss_bbox_strong_branch2'])
            # losses.update(contrastive_losses)

        # mask head forward and loss
        # TODO
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results_strong_branch1['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])


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
    def _bbox_forward_train_first_pass(self, x, sampling_results, gt_bboxes, gt_labels,
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
        oam_bboxes_strong,oam_labels_strong = bbox_select_per_class(bbox_results_strong['bbox_pred'],
                                                                    bbox_results_strong['cls_score'],
                                                                    gt_labels[1],
                                                                    score_thr=0,
                                                                    nms_cfg={'iou_threshold':0.5},
                                                                    max_num=-1
                                                                    )

        #calculate loss_weak_branch1
        bbox_targets_weak = self.bbox_head.get_targets([sampling_results[1]], [gt_bboxes[1]],
                                                         [gt_labels[1]], self.train_cfg)

        bbox_results_weak = self._bbox_forward_weak(bbox_feats_weak)
        bbox_results_weak_pseudo = self._bbox_forward_strong_branch1(bbox_feats_weak)

        loss_bbox_weak = self.bbox_head.loss_weak(bbox_results_weak['cls_proposal_mat'],
                                                      rois_weak,
                                                      *bbox_targets_weak)
        loss_weak = dict()
        loss_weak['loss_img_level_fp'] = loss_bbox_weak['loss_img_level']
        bbox_results_weak.update(loss_bbox_weak_fp=loss_weak)
        oam_bboxes_weak,oam_labels_weak = bbox_select_per_class(bbox_results_weak_pseudo['bbox_pred'],
                                                                    bbox_results_weak_pseudo['cls_score'],
                                                                    gt_labels[1],
                                                                    score_thr=0,
                                                                    nms_cfg={'iou_threshold':0.5},
                                                                    max_num=-1
                                                                    )
        # print('oam_labels_first_pass: ',oam_labels_weak)


        oam_bboxes = []
        oam_labels = []
        oam_bboxes.append(oam_bboxes_strong[:,:4])
        oam_bboxes.append(oam_bboxes_weak[:,:4])
        oam_labels.append(oam_labels_strong.to(torch_device))
        oam_labels.append(oam_labels_weak.to(torch_device))
        return bbox_results_strong,bbox_results_weak,oam_bboxes,oam_labels

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

        contrastive_losses = self.contrast_head.forward_train(bbox_feats_strong,bbox_feats_weak,strong_labels,oam_labels)
        losses = dict()
        losses.update(contrastive_losses)
        # losses['contrastive_loss'] = torch.tensor([0.0])
        return losses

        # duyu
    def _bbox_forward_embedding_branch2(self, bbox_feats,hard_neg_roi_id=None,pos_roi_id=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        # time_start = time.time()
        cls_score, bbox_pred,min_pos_pos_dist, min_neg_neg_dist = self.bbox_head.forward_embedding(bbox_feats,hard_neg_roi_id=hard_neg_roi_id,pos_roi_id=pos_roi_id)
        # print(time.time()-time_start)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats,min_pos_pos_dist=min_pos_pos_dist,min_neg_neg_dist=min_neg_neg_dist)
        return bbox_results
    #yangyk
    def get_hard_neg_target(self,rois,sampling_results):
        #print([res.hard_neg_bboxes for res in sampling_results])
        #print(sampling_results)
        flag = True
        for res in sampling_results:
            if res.hard_neg_bboxes is None:
                flag = False
        if flag is False:
            hard_neg_labels = None
            hard_neg_roi_id = None

            return hard_neg_labels,hard_neg_roi_id

        # time_start = time.time()
        hard_neg_labels = sampling_results[0].hard_neg_labels
        hard_neg_roi_id = sampling_results[0].hard_neg_id
        # hard_neg_roi_id = torch.cat(hard_neg_roi_id, 0)
        # hard_neg_labels = torch.cat(hard_neg_labels_list, 0)
        # print(time.time()-time_start)

        return hard_neg_labels,hard_neg_roi_id
    #duyu
    def _bbox_forward_train_second_pass(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas,gt_bboxes_ignore=None):
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

        #calculate loss_strong_branch1
        bbox_targets_strong = self.bbox_head.get_targets([sampling_results[0]], [gt_bboxes[0]],
                                                  [gt_labels[0]], self.train_cfg)
        #yanyk
        bbox_results_strong_branch1 = self._bbox_forward_strong_branch1(bbox_feats_strong)
        loss_bbox_strong_branch1 = self.bbox_head.loss_strong_branch1(bbox_results_strong_branch1['cls_score'],
                                                              bbox_results_strong_branch1['bbox_pred'], rois_strong,
                                                              *bbox_targets_strong)


        loss_strong_branch1 = dict()
        loss_strong_branch1['loss_cls_strong_branch1'] = loss_bbox_strong_branch1['loss_cls_strong']
        loss_strong_branch1['loss_bbox_strong_branch1'] = loss_bbox_strong_branch1['loss_bbox_strong']
        loss_strong_branch1['acc_strong_branch1'] = loss_bbox_strong_branch1['acc_strong']
        bbox_results_strong_branch1.update(loss_bbox_strong_branch1_sp=loss_strong_branch1)

        #calculate loss_weak_branch1
        bbox_results_weak_pseudo = self._bbox_forward_strong_branch1(bbox_feats_weak)
        bbox_results_weak_branch1 = self._bbox_forward_weak(bbox_feats_weak)

        loss_bbox_weak_branch1 = self.bbox_head.loss_weak_branch1(bbox_results_weak_branch1['cls_proposal_mat'],
                                                          gt_labels[1]
                                                          )
        loss_weak_branch1 = dict()
        loss_weak_branch1['loss_img_level'] = loss_bbox_weak_branch1['loss_img_level']
        bbox_results_weak_branch1.update(loss_bbox_weak_branch1_sp=loss_weak_branch1)
        #generate oam labels for weak image
        oam_bboxes_weak, oam_labels_weak = bbox_select_per_class(bbox_results_weak_pseudo['bbox_pred'],
                                                                 bbox_results_weak_pseudo['cls_score'],
                                                                 gt_labels[1],
                                                                 score_thr=0,
                                                                 nms_cfg={'iou_threshold': 0.5},
                                                                 max_num=-1
                                                                 )

        #contrastive_losses
        # contrastive_losses = self.contrast_forward_train(x,gt_bboxes[0],gt_labels[0],oam_bboxes_weak[:,:4],
        #                                                  oam_labels_weak,img_metas,gt_bboxes_ignore=gt_bboxes_ignore)

        #calculate loss_strong_branch2
        hard_neg_roi_labels, hard_neg_roi_id = self.get_hard_neg_target(rois_strong, [sampling_results[0]])
        all_bbox_labels = bbox_targets_strong[0]
        pos_roi_labels, pos_roi_id = all_bbox_labels[(all_bbox_labels != self.bbox_head.num_classes).nonzero(as_tuple=False)], \
                                            (all_bbox_labels != self.bbox_head.num_classes).nonzero(as_tuple=False)
        pos_roi_id = pos_roi_id.squeeze(1)
        pos_roi_labels = pos_roi_labels.squeeze(1)
        bbox_results_strong_branch2 = self._bbox_forward_embedding_branch2(bbox_feats_strong,
                                                                           hard_neg_roi_id=hard_neg_roi_id,
                                                                           pos_roi_id=pos_roi_id)

        loss_bbox_strong_branch2 = self.bbox_head.loss_strong_branch2(bbox_results_strong_branch2['cls_score'],
                                                       bbox_results_strong_branch2['bbox_pred'],
                                                       rois_strong,
                                                       *bbox_targets_strong,
                                                       min_pos_pos_dist=bbox_results_strong_branch2['min_pos_pos_dist'],
                                                       min_neg_neg_dist=bbox_results_strong_branch2['min_neg_neg_dist'],
                                                       pos_roi_labels=pos_roi_labels,
                                                       hard_neg_roi_labels=hard_neg_roi_labels)
        loss_strong_branch2 = dict()
        loss_strong_branch2['loss_cls_strong_branch2'] = loss_bbox_strong_branch2['loss_cls_strong']
        loss_strong_branch2['acc_strong_branch2'] = loss_bbox_strong_branch2['acc_strong']
        loss_strong_branch2['loss_bbox_strong_branch2'] = loss_bbox_strong_branch2['loss_bbox_strong']
        loss_strong_branch2['loss_embedding_strong'] = loss_bbox_strong_branch2['loss_embed_strong']
        bbox_results_strong_branch2.update(loss_bbox_strong_branch2=loss_strong_branch2)

        #calculate loss_weak_branch2
        bbox_targets_weak_branch2 = self.bbox_head.get_targets([sampling_results[1]],oam_bboxes_weak,oam_labels_weak,self.train_cfg)
        hard_neg_roi_labels, hard_neg_roi_id = self.get_hard_neg_target(rois_weak, [sampling_results[1]])
        all_bbox_labels = bbox_targets_weak_branch2[0]
        pos_roi_labels, pos_roi_id = all_bbox_labels[
                                         (all_bbox_labels != self.bbox_head.num_classes).nonzero(as_tuple=False)], (
                                             all_bbox_labels != self.bbox_head.num_classes).nonzero(as_tuple=False)
        pos_roi_id = pos_roi_id.squeeze(1)
        pos_roi_labels = pos_roi_labels.squeeze(1)

        bbox_results_weak_branch2 = self._bbox_forward_embedding_branch2(bbox_feats_weak,
                                                                           hard_neg_roi_id=hard_neg_roi_id,
                                                                           pos_roi_id=pos_roi_id)
        labels,label_weights,bbox_targets,bbox_weights = bbox_targets_weak_branch2
        # avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        # acc_weak = accuracy(bbox_results_weak_branch2['cls_score'],labels)
        loss_bbox_weak_branch2 = self.bbox_head.loss_weak_branch2(bbox_results_weak_branch2['cls_score'],
                                                                  labels,
                                                                  label_weights,
                                                                  min_pos_pos_dist=bbox_results_weak_branch2['min_pos_pos_dist'],
                                                                  min_neg_neg_dist=bbox_results_weak_branch2['min_neg_neg_dist'],
                                                                  pos_roi_labels=pos_roi_labels,
                                                                  hard_neg_roi_labels=hard_neg_roi_labels)

        loss_weak_branch2 = dict()
        loss_weak_branch2['loss_cls_weak_branch2'] = loss_bbox_weak_branch2['loss_cls_weak']
        loss_weak_branch2['acc_weak_branch2'] = loss_bbox_weak_branch2['acc_weak']
        # loss_weak_branch2['loss_embedding_weak'] = loss_bbox_weak_branch2['loss_embed_weak']
        bbox_results_weak_branch2.update(loss_bbox_weak_branch2=loss_weak_branch2)

        return bbox_results_weak_branch1,bbox_results_strong_branch1,bbox_results_weak_branch2,bbox_results_strong_branch2
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

        bbox_results = self._bbox_forward_embedding_branch2(bbox_feats)

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

            bbox_results = self._bbox_forward_embedding_branch2(bbox_feats)
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