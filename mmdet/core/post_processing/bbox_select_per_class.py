
import torch
from mmcv.ops.nms import batched_nms
from mmdet.utils import kmeans
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
    if multi_bboxes.size(0) == 0:
        return multi_bboxes,multi_bboxes.new_empty(0)

    num_classes = multi_scores.size(1) - 1
    scores = torch.softmax(multi_scores,dim=1)[:, :-1]
    gt_class_ids = (img_level_label>0).expand(scores.size(0),-1)
    scores = scores[gt_class_ids].view(multi_scores.size(0),-1)
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[gt_class_ids]
        # bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), int(torch.sum(img_level_label).item()), 4)
    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels[img_level_label>0]
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    keep = bboxes.new_ones(bboxes.size(0)).type(torch.bool)
    for i,box in enumerate(bboxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        if (y2-y1)*(x2-x1) < 5:
            keep[i] = False
        if (y2-y1)/(x2-x1) > 10 or (y2-y1)/(x2-x1)<0.1:
            keep[i] = False
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if bboxes.size(0) == 0:
        return bboxes,labels
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    labels = labels[keep]
    if max_num > 0:
        dets_raw = dets[:max_num]
        labels_raw = labels[:max_num]
    return dets_raw, labels_raw

@torch.no_grad()
def first_pass_filter(multi_bboxes,
                          multi_scores,
                          img_level_label,
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

    num_classes = multi_scores.size(1) - 1
    scores = multi_scores[:, :-1]
    gt_class_ids = (img_level_label>0).expand(scores.size(0),-1)
    scores = scores[gt_class_ids].view(multi_scores.size(0),-1)

    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[gt_class_ids]
        # bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), int(torch.sum(img_level_label).item()), 4)

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels[img_level_label>0]
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    keep = bboxes.new_ones(bboxes.size(0)).type(torch.bool)
    for i,box in enumerate(bboxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        if (y2-y1)*(x2-x1) < 5:
            # print('too small')
            keep[i] = False
        if (y2-y1)/(x2-x1) > 10 or (y2-y1)/(x2-x1)<0.1:
            keep[i] = False
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if bboxes.size(0) == 0:
        return bboxes,labels
    img_cats = labels.unique()
    cat_bbox_dict = {}
    cat_label_dict = {}
    cat_scores_dict = {}
    for cat in img_cats:
        indexes = torch.where(labels == cat)[0]
        cat_scores_dict[cat.item()],idx = torch.sort(scores[indexes])
        cat_bbox_dict[cat.item()] = bboxes[indexes][idx[len(idx)//2:]]
        cat_label_dict[cat.item()] = labels[indexes][idx[len(idx)//2:]]
    bboxes = cat_bbox_dict[img_cats[0].item()]
    labels = cat_label_dict[img_cats[0].item()]
    for cat in img_cats[1:]:
        bboxes = torch.cat((bboxes,cat_bbox_dict[cat.item()]),dim=0)
        labels = torch.cat((labels,cat_label_dict[cat.item()]),dim=0)
    return bboxes, labels
@torch.no_grad()
def bbox_select_per_class_fixnum(multi_bboxes,
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
    # multi_scores = torch.softmax(multi_scores,dim=1)
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
