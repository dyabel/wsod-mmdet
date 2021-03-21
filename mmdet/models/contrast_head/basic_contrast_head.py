from abc import ABCMeta,abstractmethod
import torch.nn as nn
from ..builder import HEADS, build_head, build_roi_extractor

@HEADS.register_module()
class BaseContrastHead(nn.Module, metaclass=ABCMeta):
    "Base class for ContrastHead"
    def __init__(self):
        super(BaseContrastHead,self).__init__()

    def forwad_train(self,
                     bbox_feats_strong,
                     bbox_feats_weak,
                     strong_bboxes,
                     strong_labels,
                     oam_bboxes,
                     oam_labels):
        torch_device = strong_labels[0].get_device()


        pass