from abc import ABCMeta,abstractmethod
import torch.nn as nn
from ..builder import HEADS, build_head, build_roi_extractor,build_loss
from functools import partial
import torch
from tqdm import tqdm
import logging
import os
from torchvision import transforms, datasets
from PIL import ImageFilter
import random
import torch
from pytorch_metric_learning import losses, miners
import record_keeper
memory_size = 3072
embedding_size = 128
paramK_momentum = 0.99
# loss_fn = losses.CrossBatchMemory(loss = losses.NTXentLoss(temperature = 0.1),
#                                   embedding_size = embedding_size,
#                                   memory_size = memory_size)

device = torch.device("cuda")

@HEADS.register_module()
class BaseContrastHead(nn.Module, metaclass=ABCMeta):
    "Base class for ContrastHead"
    def __init__(self,
                 encoder_k=None,
                 encoder_q=None,
                 loss=None):
        super(BaseContrastHead,self).__init__()
        self.init_encoder(encoder_k,encoder_q)
        self.queue = torch.tensor([])
        self.batch_size = 0
        self.max_size = 1024
        # self.loss_fn = losses.CrossBatchMemory(loss = losses.NTXentLoss(temperature = 0.1),
        #                           embedding_size = embedding_size,
        #                           memory_size = memory_size)
        self.loss_fn = build_loss(loss)
    def init_encoder(self,encoder_k,encoder_q):
        if encoder_k is not None:
            self.encoder_k = build_head(encoder_k)
        if encoder_q is not None:
            self.encoder_q = build_head(encoder_q)




    def forward_train(self,
                     feats_strong,
                     feats_weak,
                     strong_labels,
                     weak_labels):
        torch_device = strong_labels.get_device()
        q_batch = torch.tensor([]).to(torch_device)
        k_batch = torch.tensor([]).to(torch_device)
        for i,weak_label in enumerate(weak_labels):
            if weak_label in strong_labels:
                num_strong = len(torch.where(strong_labels==weak_label)[0])
                index_strong = torch.where(strong_labels==weak_label)[0][torch.where(torch.where(weak_labels==weak_label)[0]==i)[0][0]%num_strong]
                # print('index_strong:',index_strong)
                q = self.encoder_q(feats_weak[i])
                q_batch = torch.cat((q_batch,q.view(1,-1)),0)
                # print('q:',q.size())
                k = self.encoder_k(feats_strong[index_strong])
                k = k.detach()
                k_batch = torch.cat((k_batch,k.view(1,-1)), 0)
                # print('k:',k.size())
        # print(q_batch.size(),weak_labels.size())
        # print(k_batch.size(),q_batch.size())
        # print(q_batch.view(-1,1,embedding_size).size())
        # print(q_batch.view(-1,embedding_size).size(),k_batch.view(-1,embedding_size))
        l_pos = q_batch.view(-1,1,embedding_size).bmm(k_batch.view(-1,embedding_size,1)).view(-1,1)
        # print(q_batch.view(-1,embedding_size).size(),self.queue.view(embedding_size,-1))
        l_neg = q_batch.view(-1,embedding_size).mm(self.queue.view(embedding_size,-1))
        copy_params(self.encoder_q,self.encoder_k, m = paramK_momentum)
        self.dequeue()
        self.enqueue(k_batch)
        # print(weak_labels.size())


        # previous_max_label = torch.max(self.loss_fn.label_memory)

        # compute output
        # imgK, idx_unshuffle = batch_shuffle_single_gpu(imgK)
        # encK_out = encK(imgK)
        # print(l_pos.size(),l_neg.size())

        all_enc = torch.cat([l_pos, l_neg], dim=1)
        # print(all_enc.size())
        # labels, enqueue_idx = create_labels(q_batch.size(0), previous_max_label)
        # loss = self.loss_fn(all_enc, labels, enqueue_idx = enqueue_idx)
        loss = self.loss_fn(all_enc,torch.zeros(all_enc.size(0)).long().to(torch_device))
        if torch.isnan(loss):
            loss = torch.tensor([0.0]).to(torch_device)
            print(strong_labels,weak_labels)
        # print(loss)
        losses = dict()
        losses['contrastive_loss'] = loss
        return losses
    def enqueue(self,k_batch):
        self.batch_size = k_batch.size(0)
        # print(self.batch_size,self.queue.size())
        torch_device = k_batch.get_device()
        self.queue = self.queue.to(torch_device)
        self.queue = torch.cat((self.queue, k_batch), 0)

    def dequeue(self):
        if self.queue.size(0) > self.max_size:
            self.queue = self.queue[self.batch_size::]


    # def forward(self):
    #     pass


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

######################
### from MoCo repo ###
######################
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


######################
### from MoCo repo ###
######################
def copy_params(encQ, encK, m=None):
    if m is None:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    else:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)





######################
### from MoCo repo ###
######################
# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


"""# Training and logging functions"""


def update_records(loss, loss_fn, optimizer, record_keeper, global_iteration):
    def optimizer_custom_attr_func(opt):
        return {"lr": opt.param_groups[0]["lr"]}

    record_these = [[{"loss": loss.item()}, {"input_group_name_for_non_objects": "loss_histories"}],
                    [{"loss_function": loss_fn}, {"recursive_types": [torch.nn.Module]}],
                    [{"optimizer": optimizer}, {"custom_attr_func": optimizer_custom_attr_func}]]
    for record, kwargs in record_these:
        record_keeper.update_records(record, global_iteration, **kwargs)


def save_model(encQ):
    model_folder = "example_saved_models"
    if not os.path.exists(model_folder): os.makedirs(model_folder)
    torch.save(encQ.state_dict(), "{}/encQ_best.pth".format(model_folder))


######################
### from MoCo repo ###
######################
def batch_shuffle_single_gpu(x):
    """
    Batch shuffle, for making use of BatchNorm.
    """
    # random shuffle index
    idx_shuffle = torch.randperm(x.shape[0]).cuda()

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    return x[idx_shuffle], idx_unshuffle


######################
### from MoCo repo ###
######################
def batch_unshuffle_single_gpu(x, idx_unshuffle):
    """
    Undo batch shuffle.
    """
    return x[idx_unshuffle]


def create_labels(num_pos_pairs, previous_max_label):
    # create labels that indicate what the positive pairs are
    labels = torch.arange(0, num_pos_pairs)
    labels = torch.cat((labels, labels)).to(device)
    # add an offset so that the labels do not overlap with any labels in the memory queue
    labels += previous_max_label + 1
    # we want to enqueue the output of encK, which is the 2nd half of the batch
    enqueue_idx = torch.arange(num_pos_pairs, num_pos_pairs * 2)
    return labels, enqueue_idx


