from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
import numpy as np
import torch


@DATASETS.register_module()
class MyVOCDataset(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        super(MyVOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
        weak_ann_frac = 5
        self.id_idx = {}
        for idx, id in enumerate(self.img_ids):
            self.id_idx[id] = idx
        self.strong_image_ids = []
        self.weak_image_ids = []
        self.id_labelattr = {}
        self.cat_strong_ids = {}
        self.cat_weak_ids = {}
        self.num_classes = len(self.CLASSES)
        for i in self.img_ids:
            self.id_labelattr[i] = -1
        for i in self.coco.catToImgs.keys():
            self.coco.catToImgs[i] = list(set(self.coco.catToImgs[i]))
            cat_strong_image_ids = self.coco.catToImgs[i][0:(
                        len(self.coco.catToImgs[i]) // weak_ann_frac + len(self.coco.catToImgs[i]) % weak_ann_frac)]
            self.cat_weak_ids[i] = []
            self.cat_strong_ids[i] = []
            for j in self.coco.catToImgs[i]:
                if j not in self.img_ids:
                    continue
                if self.id_labelattr[j] != -1:
                    continue
                if j in cat_strong_image_ids:
                    self.cat_strong_ids[i].append(j)
                    self.id_labelattr[j] = True
                else:
                    self.cat_weak_ids[i].append(j)
                    self.id_labelattr[j] = False
        assert len(self.id_labelattr) == len(self.img_ids)
        print('allocating completed')
        indices = []
        for i in self.cat_strong_ids.keys():
            num_strong = len(self.cat_strong_ids[i])
            if num_strong == 0:
                continue
            for j in range(len(self.cat_weak_ids[i])):
                indices.append([self.id_idx[self.cat_strong_ids[i][j % num_strong]],
                                self.id_idx[self.cat_weak_ids[i][j]]])
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        for i in range(len(self.img_ids)):
            if i not in indices:
                indices.append(i)
                print('append', i)
        random_indices = torch.tensor(indices)
        # random_indices = random_indices[0:4200]

        # random_indice = torch.randperm(len(self.img_ids))
        # random_indice = []
        # for i in range(len(self.img_ids)):
        #     random_indice.append([1,i])
        # np.random.shuffle(random_indice)
        # random_indices = np.concatenate(random_indice)

        # random_indice = torch.cat((random_indice,torch.randperm(1860)))
        # random_indices=torch.unique(random_indices)
        # assert  (np.unique(random_indices.numpy()) == np.unique(random_indice.numpy())).all()
        # random_indices = random_indice
        # print(len(self.img_ids))
        # print(len(random_indices))
        self.data_infos = [self.data_infos[i] for i in valid_inds]
        self.data_infos = [self.data_infos[i] for i in random_indices]
        if self.proposals is not None:
            self.proposals = [self.proposals[i] for i in valid_inds]
            self.proposals = [self.proposals[i] for i in random_indices]
        self.img_ids = [self.img_ids[i] for i in random_indices]
        # assert len(self.indices) == len(self.img_ids)
        for i, j in enumerate(self.img_ids):
            assert j == self.data_infos[i]['id']
        self._set_group_flag()

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        half_len = len(self) // 2
        if half_len & 1 != 0:
            half_len += 1
        for i in range(half_len):
            self.flag[i] = 1

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
