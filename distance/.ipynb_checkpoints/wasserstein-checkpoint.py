import imp
from cv2 import log
import torch
import torch.nn as nn

from einops import einops, rearrange, repeat, reduce

    
def wasserstein_dist(mu, log_var):
    bs, head, dim = mu.shape
    mu_dist = repeat(mu, 'b h d -> b repeat (h d)', repeat=bs) - rearrange(mu, 'b h d -> b (h d)')
    # mu_dist.register_hook(lambda x: print(mu_dist))
    dist = reduce(mu_dist**2, 'b n d -> b n', reduction='sum')
    
    std = torch.exp(0.5 * log_var)
    avg_std = reduce(std, 'b h d -> b h', reduction='mean')
    
    co_var = torch.matmul(rearrange(avg_std, 'b h -> b h 1'),rearrange(avg_std, 'b h -> b 1 h'))
    co_var = torch.diagonal(co_var, dim1=-2, dim2=-1)
    # co_var = torch.diag_embed(torch.abs(avg_std))
    # co_var.register_hook(lambda x: print(co_var))
    co_dist = einops.repeat(co_var**0.5, 'b h  -> b repeat h ', repeat=bs) - co_var**0.5
    # co_dist.register_hook(lambda x: print(co_dist))
    dist += reduce(co_dist**2, 'b n d -> b n', reduction='sum')
    
    # dist = torch.sqrt(dist)
    return dist

