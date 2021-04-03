import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.utils import visualize_oam_boxes,iou


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)
    # output_dir = args.output_dir

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    for iitem in dataset:
        # print(item['img_metas'].data)
        try:
            item = iitem['img_metas'].data
        except:
            print(iitem)

        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None

        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)
        # print(iitem['img'].data)
        # print(iitem['img_metas'])
        visualize_oam_boxes(iitem['gt_bboxes'].data, iitem['gt_labels'].data, iitem['img'].data, [1,iitem['img_metas'].data],
                            win_name=str(1), show=False,
                            out_dir=args.output_dir, show_score_thr=0)
        # imshow_det_bboxes(
        #     iitem['img'],
        #     iitem['gt_bboxes'].data,
        #     iitem['gt_labels'].data,
        #     gt_masks,
        #     class_names=dataset.CLASSES,
        #     show=not args.not_show,
        #     wait_time=args.show_interval,
        #     out_file=filename,
        #     bbox_color=(255, 102, 61),
        #     text_color=(255, 102, 61))

        progress_bar.update()


if __name__ == '__main__':
    main()
