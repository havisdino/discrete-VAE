import torch
from torch import dropout, nn

from utils import init_weights


class ReZeroFFNBlock(nn.Module):
    def __init__(self, dff, dropout):
        super().__init__()
        self.linear = nn.Linear(dff, dff)
        self.gelu = nn.GELU()
        self.coef = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        y = self.linear(x)
        y = self.gelu(x)
        y = x + self.dropout(y * self.coef)
        return y


class DecoderFFN(nn.Sequential):
    def __init__(self, d_model, dff, nlayers, dropout):
        super().__init__()
        self.append(nn.Linear(d_model, dff))
        self.append(nn.GELU())
        
        for _ in range(nlayers - 1):
            self.append(ReZeroFFNBlock(dff, dropout))
        

class Decoder(nn.Module):
    def __init__(self, d_model, dff, nlayers, d_latent, dropout=0.1, nbits=8):
        super().__init__()
        
        self.latent_dim = d_latent
        self.nclasses = 2 ** nbits
        
        self.zproj = nn.Linear(d_latent, d_model)
        self.ffn = DecoderFFN(d_model, dff, nlayers, dropout)
        self.outporj = nn.Linear(dff, self.nclasses)
        
        self.apply(init_weights)
    
    def forward(self, z, pe):
        assert z.ndim == 2 and z.size(-1) == self.latent_dim
        B = z.size(0)
        
        z = self.zproj(z)
        x = z.unsqueeze(1) + pe
        x = self.ffn(x)
        logits = self.outporj(x)
        return logits
    
