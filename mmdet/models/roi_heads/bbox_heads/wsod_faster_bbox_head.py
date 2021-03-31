import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.losses import accuracy,img_level_accuracy
from mmdet.models.builder import build_loss
from mmdet.core.utils import convert_label


@HEADS.register_module()
class ConvFCWSODHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                 loss_cls_weak=dict(
                     type='MyCrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 *args,
                 **kwargs):
        super(ConvFCWSODHead, self).__init__(*args, **kwargs)
        self.loss_cls = build_loss(loss_cls)
        self.loss_cls_weak = build_loss(loss_cls_weak)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls_weak_branch1 = nn.Linear(self.cls_last_dim, self.num_classes)
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
            self.fc_cls_branch2 = nn.Linear(self.cls_last_dim, self.num_classes + 1)

        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg_weak_branch1 = nn.Linear(self.reg_last_dim, self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
            self.fc_reg_branch2 = nn.Linear(self.reg_last_dim, out_dim_reg)
            # print(out_dim_reg)

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
            nn.init.normal_(self.fc_cls_weak_branch1.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls_weak_branch1.bias, 0)
            nn.init.normal_(self.fc_cls_branch2.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls_branch2.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
            nn.init.normal_(self.fc_reg_weak_branch1.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg_weak_branch1.bias, 0)
            nn.init.normal_(self.fc_reg_branch2.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg_branch2.bias, 0)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
                # print(self.fc_out_channels)
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCWSODHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    #duyu
    def forward_weak(self, x):
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls_weak_branch1(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg_weak_branch1(x_reg) if self.with_reg else None
        cls_score = F.softmax(cls_score,dim=0)
        bbox_pred = F.softmax(bbox_pred,dim=1)
        # cls_proposal_mat = cls_score*bbox_pred

        return cls_score*bbox_pred
    #duyu
    def forward_strong_branch1(self, x):
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
    def double_fc_forward(self,x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        return x
    #duyu
    def forward_strong_branch2(self, x):
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls_branch2(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg_branch2(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
    #yangyk
    def forward_embedding(self, x, hard_neg_roi_id=None, pos_roi_id=None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # yangyk
        pos_feat = self.pos_cls_fcs[0](x_cls)
        neg_feat = self.neg_cls_fcs[0](x_cls)

        pos_one = torch.ones(1, device=pos_feat.device)
        neg_one = torch.ones(1, device=neg_feat.device)

        pos_reps = self.pos_rep_fc[0](pos_one)
        neg_reps = self.pos_rep_fc[0](neg_one)

        pos_reps = torch.reshape(pos_reps, [self.base_classes, self.pos_vec_length, self.reps_per_class])
        neg_reps = torch.reshape(neg_reps, [self.base_classes, self.neg_vec_length, self.reps_per_class])

        norm_pos_reps = torch.nn.functional.normalize(pos_reps, p=2, dim=1)
        norm_neg_reps = torch.nn.functional.normalize(neg_reps, p=2, dim=1)

        norm_pos_feat = torch.nn.functional.normalize(pos_feat, p=2, dim=1)
        norm_neg_feat = torch.nn.functional.normalize(neg_feat, p=2, dim=1)

        # norm_pos_feat = torch.reshape(norm_pos_feat,[norm_pos_feat.shape[0],norm_pos_feat.shape[1],1])
        # norm_neg_feat = torch.reshape(norm_neg_feat,[norm_neg_feat.shape[0],norm_neg_feat.shape[1],1])

        # print('Computing cosine distance!!!')
        # cos distance
        pos_cos_dist = self.cos_distance(norm_pos_feat, norm_pos_reps)
        neg_cos_dist = self.cos_distance(norm_neg_feat, norm_neg_reps)

        # Eculid distance
        # print('Computing Eculid distance!!!')
        # pos_cos_dist = self.euclid_distance(norm_pos_feat,norm_pos_reps)
        # neg_cos_dist = self.euclid_distance(norm_neg_feat,norm_neg_reps)

        min_pos_dist_cls, arg_min_pos_dist_cls = pos_cos_dist.min(dim=2)
        min_neg_dist_cls, arg_min_neg_dist_cls = neg_cos_dist.min(dim=2)

        beta = 0.2
        gamma = 0.4
        min_np_dist_cls = min_pos_dist_cls - beta * min_neg_dist_cls + gamma

        min_np_probs_cls = torch.exp(-2 * min_np_dist_cls)
        max_fore_prob_cls, _ = torch.max(min_np_probs_cls, dim=1)
        # cos scalor first experiment
        # bg_scalor = 0.5
        bg_scalor = 0.2
        bg_prob = 1. - bg_scalor * max_fore_prob_cls

        cls_score = torch.cat((min_np_probs_cls, bg_prob.unsqueeze(1)), dim=1)

        # cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        '''
        final_pos_reps = norm_pos_reps
        final_neg_reps = norm_neg_reps


        #pos_feats
        if pos_roi_id is not None:
            pos_feats_embed = norm_pos_feat[pos_roi_id]
        else:
            pos_feats_embed = None

        #neg_feats
        if hard_neg_roi_id is not None:
            hard_neg_feats_embed = norm_neg_feat[hard_neg_roi_id]
        else:
            hard_neg_feats_embed = None

        min_pos_pos_dist = min_pos_dist_cls[pos_roi_id]
        min_neg_neg_dist = min_neg_dist_cls[hard_neg_roi_id]
        '''

        min_pos_pos_dist = min_pos_dist_cls[pos_roi_id]
        min_neg_neg_dist = min_neg_dist_cls[hard_neg_roi_id]

        return cls_score, bbox_pred, min_pos_pos_dist, min_neg_neg_dist

    @force_fp32(apply_to='cls_proposal_mat')
    def loss_weak(self,
             cls_proposal_mat,
             label_img_level,
             reduction_override=None):
        losses = dict()
        phi = torch.sum(cls_proposal_mat,dim=0)
        torch_device = label_img_level.get_device()
        label_weights = torch.ones(label_img_level.size(0)).to(torch_device)
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        # torch.set_printoptions(profile="full")
        # num_cls = cls_proposal_mat.size(1)
        # label_img_level,label_weights = convert_label(labels,num_cls)
        losses['loss_img_level'] = self.loss_cls_weak(phi,
                                                 label_img_level,
                                                 label_weights,
                                                 avg_factor=avg_factor,
                                                 reduction_override=reduction_override)
        return losses

    #duyu
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss_strong(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls_strong'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc_strong'] = accuracy(cls_score, labels)
                # losses['acc'] = 1
                # print(losses['acc'])
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox_strong'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox_strong'] = bbox_pred[pos_inds].sum()
        return losses

    #yangyk
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss_strong1(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             min_pos_pos_dist=None,
             min_neg_neg_dist=None,
             pos_roi_labels=None,
             hard_neg_roi_labels=None
             ):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls_strong'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc_strong'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox_strong'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox_strong'] = bbox_pred[pos_inds].sum()


            if pos_roi_labels is not None:
                min_pos_pos_correct_cls = min_pos_pos_dist.new_full((pos_roi_labels.shape[0], ), -1 ,dtype=min_pos_pos_dist.dtype)

                for i, _ in enumerate(min_pos_pos_correct_cls):
                    #print(i)
                    min_pos_pos_correct_cls[i] = min_pos_pos_dist[i,pos_roi_labels[i]]

                min_pos_pos_avg_dist = min_pos_pos_correct_cls.mean()

            else:
                min_pos_pos_avg_dist = 0




            if hard_neg_roi_labels is not None:
                min_neg_neg_correct_cls = min_neg_neg_dist.new_full((hard_neg_roi_labels.shape[0], ), -1 ,dtype=min_neg_neg_dist.dtype)

                for i, _ in enumerate(min_neg_neg_correct_cls):
                    min_neg_neg_correct_cls[i] = min_neg_neg_dist[i,hard_neg_roi_labels[i]]

                min_neg_neg_avg_dist = min_neg_neg_correct_cls.mean()

            else:
                min_neg_neg_avg_dist = 0


            losses['loss_embed_strong'] = min_pos_pos_avg_dist + min_neg_neg_avg_dist

        return losses

@HEADS.register_module()
class Shared2FCWSODHead(ConvFCWSODHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCWSODHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


