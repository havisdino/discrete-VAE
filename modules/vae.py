from dataclasses import dataclass
import torch
from torch import nn

from modules.decoder import Decoder
from modules.encoder import Encoder


@dataclass(eq=False)
class VAE(nn.Module):
    '''
    Decoder config: d_model, dff, nlayers, d_latent, dropout=0.1, nbits=8
    Encoder config: d_model, dff, nheads, nblocks, d_latent, dropout=0.1, nbits=8
    '''
    
    chw: int
    d_model: int
    dff: int
    d_latent: int
    nlayers_decoder: int
    nheads_encoder: int
    nblocks_encoder: int
    dropout: float = 0.1
    nbits: int = 8
    
    def __post_init__(self):
        super().__init__()
        
        self.pe = nn.Parameter(torch.randn(1, self.chw, self.d_model))
        
        self.encoder = Encoder(
            self.d_model, self.dff, self.nheads_encoder,
            self.nblocks_encoder, self.d_latent, self.dropout, self.nbits
        )
        self.decoder = Decoder(
            self.d_model, self.dff, self.nlayers_decoder, self.d_latent,
            self.dropout, self.nbits
        )
    
    def forward(self, x):
        mean, logstd = self.forward_encoder(x)
        eps = torch.randn_like(mean)
        z = mean + eps * logstd.exp()
        decoder_logits = self.forward_decoder(z)
        return mean, logstd, z, decoder_logits

    def forward_decoder(self, z):
        return self.decoder(z, self.pe)
    
    def forward_encoder(self, x):
        return self.encoder(x, self.pe)
    
    @torch.no_grad()
    def generate_z(self, nsamples, device):
        return torch.randn(nsamples, self.d_latent, device=device)
    