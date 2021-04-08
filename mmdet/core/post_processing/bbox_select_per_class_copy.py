
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

    num_classes = multi_scores.size(1) - 1
    # multi_scores = torch.sigmoid(multi_scores)
    scores = multi_scores[:, :-1]
    gt_class_ids = (img_level_label>0).expand(scores.size(0),-1)
    scores = scores[gt_class_ids].view(multi_scores.size(0),-1)
    scores = torch.sigmoid(scores)

    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), int(torch.sum(img_level_label).item()), 4)


    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels[img_level_label>0]

    labels = labels.view(1, -1).expand_as(scores)

    # img_level_label = img_level_label.view(1,-1).expand_as(scores).reshape(-1)

    # print(bboxes.size())
    # print(gt_class_inds.size())
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    # print(bboxes.size(),scores.size(),labels)
    # if inds.numel() == 0:
    #     if torch.onnx.is_in_onnx_export():
    #         raise RuntimeError('[ONNX Error] Can not record NMS '
    #                            'as it has not been executed this time')
    #     return bboxes, labels

    # TODO: add size check before feed into batched_nms
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


    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    labels = labels[keep]
    # img_cats = labels.unique()
    # cat_bbox_dict = dict()
    # cat_label_dict = dict()
    # for cat in img_cats:
    #     indexes = torch.where(labels == cat)[0]
    #     cat_bbox_dict[cat] = kmeans(bboxes[indexes],n_clusters=len(bboxes//2))
    #     cat_label_dict[cat] = labels[indexes][:len(bboxes)//2]
    # dets = torch.empty(1,4)
    # labels = torch.empty(1)
    # for cat, bbox in cat_bbox_dict.items():
    #     dets = torch.cat((dets, torch.tensor(bbox)), dim=0)
    #     labels = torch.cat((labels, torch.tensor(cat_label_dict[cat])), dim=0)
    # dets = dets[1:,:]
    # labels = labels[1:,:]

    if max_num > 0:
        dets_raw = dets[:max_num]
        labels_raw = labels[:max_num]
    # img_cats = img_level_label.nonzero()
    # for cat in img_cats:
    #     if cat not in labels_raw:
    #         index = torch.where(labels == cat)[0][0]
    #         labels_cat = torch.cat((labels_raw,cat),dim=0)
    #         dets = torch.cat((dets_raw,dets[index]),dim=0)
    #         labels_raw = labels_cat
    #         dets_raw = dets
    # return bboxes[idx[-fix_num::]],labels[idx[-fix_num::]]
    return dets_raw, labels_raw

@torch.no_grad()
def bbox_select_per_class1(multi_bboxes,
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

    num_classes = multi_scores.size(1) - 1
    # multi_scores = torch.sigmoid(multi_scores)
    scores = multi_scores[:, :-1]
    gt_class_ids = (img_level_label>0).expand(scores.size(0),-1)
    scores = scores[gt_class_ids].view(multi_scores.size(0),-1)
    scores = torch.sigmoid(scores)

    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), int(torch.sum(img_level_label).item()), 4)


    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels[img_level_label>0]

    labels = labels.view(1, -1).expand_as(scores)

    # img_level_label = img_level_label.view(1,-1).expand_as(scores).reshape(-1)

    # print(bboxes.size())
    # print(gt_class_inds.size())
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    # print(bboxes.size(),scores.size(),labels)
    # if inds.numel() == 0:
    #     if torch.onnx.is_in_onnx_export():
    #         raise RuntimeError('[ONNX Error] Can not record NMS '
    #                            'as it has not been executed this time')
    #     return bboxes, labels

    # TODO: add size check before feed into batched_nms
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
    # dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    # print(labels,gt_class_inds)
    # max_num = fix_num
    # labels = labels[keep]
    img_cats = labels.unique()
    cat_bbox_dict = dict()
    cat_label_dict = dict()
    cat_scores_dict = dict()
    for cat in img_cats:
        indexes = torch.where(labels == cat)[0]
        cat_bbox_dict[cat] = bboxes[indexes]
        cat_label_dict[cat] = labels[indexes]
        cat_scores_dict[cat] = scores[indexes]
    for cat in img_cats:
        if cat == 14 :
            # print(cat_labe_dict)
            cat_bbox_dict[cat], keep = batched_nms(cat_bbox_dict[cat],cat_scores_dict[cat], cat_label_dict[cat], {'iou_threshold':0.5})
            cat_label_dict[cat] = cat_label_dict[keep]
        else:
            # cat_bbox_dict[cat] = cat_bbox_dict[:4]
            cat_bbox_dict[cat], keep = batched_nms(cat_bbox_dict[cat],cat_scores_dict[cat], cat_label_dict[cat], {'iou_threshold':0.3})
            cat_label_dict[cat] = cat_label_dict[keep]
    dets = cat_bbox_dict[img_cats[0]]
    labels = cat_label_dict[img_cats[0]]
    for cat,bbox in cat_bbox_dict.items():
        if cat == img_cats[0]:
            continue
        dets = torch.cat((dets,bbox),dim=0)
        labels = torch.cat((labels,cat_label_dict[cat]),dim=0)

    # if max_num > 0:
    #     dets = dets[:max_num]
    #     keep = keep[:max_num]

    # return bboxes[idx[-fix_num::]],labels[idx[-fix_num::]]
    return dets[:max_num], labels[:max_num]
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
