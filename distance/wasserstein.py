import imp
from cv2 import log
import torch
import torch.nn as nn

from einops import einops, rearrange, repeat, reduce

    
def wasserstein_dist(mu, log_var):
    bs, head, dim = mu.shape
    mu_dist = repeat(mu, 'b h d -> b repeat (h d)', repeat=bs) - rearrange(mu, 'b h d -> b (h d)')
    dist = reduce(mu_dist**2, 'b n d -> b n', reduction='sum')
    
    avg_var = reduce(log_var, 'b h d -> b h', reduction='mean')

    co_var = torch.diag_embed(avg_var**2)
    co_dist = einops.repeat(co_var, 'b h1 h2 -> b repeat h1 h2', repeat = bs)**0.5 - co_var**0.5
    dist += torch.norm(co_dist, p='fro', dim=(-2, -1))
    dist = torch.sqrt(dist)
    return dist

