from mmdet.core.evaluation import bbox_overlaps
import torch
a = torch.tensor([[1,2,3,4]])
b = torch.tensor([[4,3,2,1]])
a = a.numpy()
b = b.numpy()
print(bbox_overlaps(a,b))
