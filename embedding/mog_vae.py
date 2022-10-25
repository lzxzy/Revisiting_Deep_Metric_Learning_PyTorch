from operator import ipow
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from einops import einops, rearrange, reduce, repeat
import pdb


class Encoder(nn.Module):
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 1024, latent_dim: int = 2048, head_num: int = 32):
        super(Encoder, self).__init__()
        
        self.head_number = head_num
        
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim//head_num, latent_dim//head_num)
        self.FC_var = nn.Linear(hidden_dim//head_num, latent_dim//head_num)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True
        
    def forward(self, x):
        # pdb.set_trace()
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        h_ = rearrange(h_, 'b (h d) -> b h d', h=self.head_number)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        
        return mean, log_var

class Decoder(nn.Module):
    
    def __init__(self, latent_dim: int = 2048, hidden_dim: int = 1024, output_dim=128):
        super(Decoder, self).__init__()
        
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        pdb.set_trace()
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        
        prob_embd = self.FC_output(h)
        return prob_embd
    
class MoG_VAE(nn.Module):

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, latent_dim: int = 2048, head_num: int = 32, output_dim: int = 2048):
        super(MoG_VAE, self).__init__()
        
        self.encoder = Encoder(input_dim = input_dim,
                               hidden_dim = hidden_dim,
                               latent_dim = latent_dim,
                               head_num = head_num)
        
        self.decoder = Decoder(latent_dim = latent_dim,
                               hidden_dim = hidden_dim,
                               output_dim = output_dim)
        
    def reparameterization(self, mean, log_var):
        pdb.set_trace()
        epsilon = torch.randn_like(log_var).to(log_var.device)
        z = mean + log_var * epsilon
        return z
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        pdb.set_trace()
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        z = rearrange(z, 'b h d -> b (h d)')
        prob_embd = self.decoder(z)
        return prob_embd, mean, log_var