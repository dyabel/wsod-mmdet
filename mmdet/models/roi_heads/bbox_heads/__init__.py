from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
# from .oicr_bbox_head_copy import (ConvFCOICRHead,Shared2FCOICRHead)
from .oicr_bbox_head import (ConvFCOICRHead,Shared2FCOICRHead)
from .oicr_bbox_head_branch1 import (ConvFCOICRHeadBranch1,Shared2FCOICRHeadBranch1)
from .oicr_bbox_head_branch2 import (ConvFCOICRHeadBranch2,Shared2FCOICRHeadBranch2)
from .double_bbox_head import DoubleConvFCBBoxHead
from .sabl_head import SABLHead
from .wsod_bbox_head import ConvFCWSODHead,Shared2FCWSODHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead','ConvFCOICRHead','Shared2FCOICRHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead','ConvFCOICRHeadBranch1','Shared2FCOICRHeadBranch1',
    'ConvFCOICRHeadBranch2','Shared2FCOICRHeadBranch2','ConvFCWSODHead','Shared2FCWSODHead'
]
