import imp
from cv2 import log
import torch
import torch.nn as nn

from einops import einops, rearrange, repeat, reduce

def reconstruction_loss(real_embeding, recon_embedding):
    re_loss = nn.MSELoss(real_embeding, recon_embedding)
    return re_loss

def kl_div(prob_embed, mu, log_var):
    std = torch.exp(0.5 * log_var)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_qzx = q.log_prob(prob_embed)
    log_pz = p.log_prob(prob_embed)

    kl_loss = log_qzx-log_pz
    kl_loss = torch.mean(kl_loss.sum(-1).sum(-1))
    return kl_loss
