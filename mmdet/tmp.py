# -*- coding: utf-8 -*-
# @Time    : 2021/3/1 22:48
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : tmp.py
# @Software: PyCharm
loss_bbox['loss_cls'] = strong_label * self.bbox_head.loss_cls_strong(bbox_results['cls_score'], labels, label_weights,
                                                                      avg_factor=avg_factor, reduction_override=None)
bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                          gt_labels, self.train_cfg)
loss_bbox_strong = self.bbox_head.loss_strong(bbox_results['cls_score'],
                                              bbox_results['bbox_pred'], rois,
                                              *bbox_targets)
loss_bbox['loss_cls'] = ~strong_label * loss_bbox['loss_cls'] + strong_label * loss_bbox_strong['loss_cls']
loss_bbox['loss_bbox'] = strong_label * loss_bbox_strong['loss_bbox']
bbox_results.update(loss_bbox=loss_bbox)