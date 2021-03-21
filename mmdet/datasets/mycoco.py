from pycocotools.coco import COCO
class COCO(COCO):
    def __init__(self,annatation_file=None):
        super(COCO,self).__init__(annatation_file)

