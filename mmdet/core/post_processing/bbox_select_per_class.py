
import torch
from mmcv.ops.nms import batched_nms
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import torch.nn.functional as F
@torch.no_grad()
def bbox_select_per_class_fixnum(multi_bboxes,
                   multi_scores,
                   img_level_label,
                   score_thr,
                   nms_cfg,
                   num=-1,
                   ):
    """ bbox selection per class for first pass of BBA

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        img_level_label: shape (n,class) indicating which classes in the image
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """

    # print('22222222222:',len(multi_bboxes),multi_scores.size())
    num_classes = multi_scores.size(1) - 1
    multi_scores = torch.softmax(multi_scores,dim=1)
    # exclude background category
    # print(multi_bboxes.shape)
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    # print('#'*100)
    # print('img_label',img_level_label.nonzero())
    # remove low scoring boxes
    img_level_label = img_level_label.view(1,-1).expand_as(scores).reshape(-1)
    gt_class_inds = img_level_label > 0
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    bboxes, scores, labels = bboxes[gt_class_inds], scores[gt_class_inds], labels[gt_class_inds]
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    _,idx = torch.sort(scores)
    # print(bboxes.size(),scores.size(),labels)
    # if inds.numel() == 0:
    #     if torch.onnx.is_in_onnx_export():
    #         raise RuntimeError('[ONNX Error] Can not record NMS '
    #                            'as it has not been executed this time')
    #     return bboxes, labels

    # TODO: add size check before feed into batched_nms
    # dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    # print(labels,gt_class_inds)

    # if max_num > 0:
    #     dets = dets[:max_num]
    #     keep = keep[:max_num]

    return bboxes[idx[-num::]],labels[idx[-num::]]

@torch.no_grad()
def bbox_select_per_class(multi_bboxes,
                          multi_scores,
                          img_level_label,
                          score_thr,
                          nms_cfg,
                          max_num=-1,
                          ):
    """ bbox selection per class for first pass of BBA

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        img_level_label: shape (n,class) indicating which classes in the image
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """
    # multi_scores = torch.sigmoid(multi_scores)
    num_classes = multi_scores.size(1) - 1
    multi_scores = torch.softmax(multi_scores,dim=1)
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)



    # remove low scoring boxes
    img_level_label = img_level_label.view(1,-1).expand_as(scores).reshape(-1)
    gt_class_inds = img_level_label > 0
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    bboxes, scores, labels = bboxes[gt_class_inds], scores[gt_class_inds], labels[gt_class_inds]
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    if inds.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    # TODO: add size check before feed into batched_nms
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    # print(labels,gt_class_inds)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]
