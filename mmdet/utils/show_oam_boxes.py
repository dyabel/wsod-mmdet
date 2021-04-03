import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
from mmcv.image import tensor2imgs
from mmdet.core.visualization import imshow_det_bboxes
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train','tvmonitor'
           )
def visualize_oam_boxes(bboxes,
                        labels,
                        img_tensor,
                        img_metas,
                        show=True,
                        out_dir=None,
                        win_name='',
                        show_score_thr=0.3):
    img_tensor = img_tensor.unsqueeze(0)
    img = tensor2imgs(img_tensor,**img_metas[1]['img_norm_cfg'])[0]
    h, w, _ = img_metas[0]['img_shape']
    img_show = img[:h, :w, :]
    ori_h, ori_w = img_metas[1]['ori_shape'][:-1]
    img_show = mmcv.imresize(img_show, (ori_w, ori_h))
    if out_dir:
        out_file = osp.join(out_dir, img_metas[1]['ori_filename'])
    else:
        out_file = None
    bboxes = bboxes.detach().cpu().numpy()
    labeles = labels.detach().cpu().numpy()
    imshow_det_bboxes(
        img,
        bboxes,
        labels,
        # segms,
        class_names=CLASSES,
        # score_thr=score_thr,
        # bbox_color=bbox_color,
        # text_color=text_color,
        # mask_color=mask_color,
        # thickness=thickness,
        # font_scale=font_scale,
        # font_size=font_size,
        win_name=win_name,
        # fig_size=fig_size,
        show=show,
        # wait_time=wait_time,
        out_file=out_file)
        # model.module.show_result(
            #     img_show,
            #     result[i],
            #     show=show,
            #     out_file=out_file,
            #     score_thr=show_score_thr)

    # encode mask results
    # # if isinstance(result[0], tuple):
    # #     result = [(bbox_results, encode_mask_results(mask_results))
    # #               for bbox_results, mask_results in result]
    # # results.extend(result)
    # #
    # # for _ in range(batch_size):
    # #     prog_bar.update()
    # return results
