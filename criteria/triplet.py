from tokenize import Triple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import pdb
"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

### Standard Triplet Loss, finds triplets in Mini-batches.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.margin     = opt.loss_triplet_margin
        self.batchminer = batchminer
        self.name           = 'triplet'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    # def triplet_distance(self, anchor, positive, negative):
        # return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)
    def triplet_distance(self, positive_dist, negative_dist):
        return torch.max(positive_dist-negative_dist+self.margin,  torch.tensor([0.]).to("cuda"))

    def forward(self, batch, labels, distance=None, **kwargs):
        
        # pdb.set_trace()
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        sampled_triplets = self.batchminer(batch, labels, distances=distance.detach())
        # loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])
        loss = torch.stack([self.triplet_distance(distance[triplet[0],triplet[1]], distance[triplet[0],triplet[2]]) for triplet in sampled_triplets])
        return torch.mean(loss)
