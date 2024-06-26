from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.dim_head = d_model // n_heads
        self.d_model = d_model
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        self.register_buffer('scale', torch.FloatTensor([self.dim_head]).sqrt())
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, attn_mask=None):
        B, L, _ = inputs.size()
        
        qkv = self.qkv(inputs)
        qkv = qkv.view(B, L, self.n_heads, -1)
        qkv = qkv.permute(0, 2, 1, 3)
        Q, K, V = qkv.split(self.dim_head, dim=-1)
        scores = Q.matmul(K.permute(0, 1, 3, 2)) / self.scale
        if attn_mask is not None:
            scores += attn_mask
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        outputs = scores.matmul(V)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        outputs = outputs.view(B, L, -1)
        
        return outputs
    

class FFN(nn.Sequential):
    def __init__(self, d_model, dff):
        super().__init__()
        self.append(nn.Linear(d_model, dff))
        self.append(nn.GELU())
        self.append(nn.Linear(dff, d_model))
        

class ReZeroTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout=0.1):
        super().__init__()
        self.register_parameter(
            'alpha',
            nn.Parameter(torch.zeros(1), requires_grad=True)
        )
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, attn_mask=None):
        x = self.self_attn(inputs, attn_mask) * self.alpha + inputs
        x = self.dropout(x)
        x = self.ffn(x) * self.alpha + x
        x = self.dropout(x)
        return x
        

class Transformer(nn.Module, ABC):
    def __init__(self, d_model, dff, nheads, nblocks, d_latent, dropout=0.1, nbits=8):
        super().__init__()
        self.n_blocks = nblocks
        self.te = nn.Embedding(2 ** nbits, d_model)
        
        self.mean = nn.Linear(d_model, d_latent)
        self.logstd = nn.Linear(d_model, d_latent)
        
        self.blocks = nn.ModuleList()
        self._build_transformer_blocks(nblocks, d_model, nheads, dff, dropout)
    
    @abstractmethod
    def _build_transformer_blocks(self, n_blocks, d_model, n_heads, dff, dropout):
        pass
        
    def forward(self, input_ids, pemat):
        te = self.te(input_ids)
        
        x = te + pemat
        
        for block in self.blocks:
            x = block(x)
            
        x = x.mean(1)
        mean = self.mean(x)
        logstd = self.logstd(x)
        
        return mean, logstd
    

class Encoder(Transformer):
    def _build_transformer_blocks(self, n_blocks, d_model, n_heads, dff, dropout):
        for _ in range(n_blocks):
            self.blocks.append(ReZeroTransformerBlock(d_model, n_heads, dff, dropout))