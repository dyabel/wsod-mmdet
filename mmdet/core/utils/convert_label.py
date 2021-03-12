import torch
def convert_label(labels,num_cls):
    torch_device = labels.get_device()
    label_img_level = torch.zeros(num_cls).to(torch_device)
    for i in labels:
        if i == num_cls:
            pass
        else:
            label_img_level[i] = 1.0
    label_weights = torch.ones(num_cls).to(torch_device)
    return label_img_level,label_weights
