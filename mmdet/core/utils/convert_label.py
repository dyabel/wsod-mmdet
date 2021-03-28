import torch
def convert_label(labels,num_cls):
    label_img_level = labels.new_zeros(num_cls,dtype=torch.float)
    for i in labels:
        if i == num_cls:
            pass
        else:
            label_img_level[i] = 1.0
    label_weights = labels.new_ones(num_cls,dtype=torch.float)
    return label_img_level,label_weights
