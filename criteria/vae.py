import imp
from cv2 import log
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einops, rearrange, repeat, reduce
import pdb
# def reconstruction_loss(real_embeding, recon_embedding):
#     pdb.set_trace()
#     re_loss = nn.(real_embeding, recon_embedding)
#     return re_loss

def log_dist(prob_embed, mu, log_var):
    std = torch.exp(0.5 * log_var)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_qzx = q.log_prob(prob_embed)
    log_pz = p.log_prob(prob_embed)

    # kl_loss = F.kl_div(log_qzx, log_pz, log_target=True, reduction)
    # kl_loss = torch.mean(kl_loss.sum(-1).sum(-1))
    return log_qzx, log_pz
