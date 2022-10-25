import imp
from cv2 import log
import torch
import torch.nn as nn

from einops import reduce

def co_var(log_var):
    avg_var = reduce(log_var, 'b h d -> b h', reduction='mean')
    co_var = 