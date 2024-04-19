import torch
from dataclasses import dataclass
from typing import Any

from modules.vae import VAE


@dataclass
class Sampler:
    model: VAE
    device: Any
    use_amp: bool = True
    
    def __post_init__(self):
        self.model.to(self.device)
        self.model.eval()
        
        if self.device == 'cuda':
            self.dtype = torch.float16
        elif self.device == 'cpu':
            self.dtype = torch.bfloat16
    
    @torch.no_grad()
    def sample(self, nsamples, temparature=1.0):
        with torch.autocast(self.device, self.dtype, self.use_amp):
            z = self.model.generate_z(nsamples, self.device)
            logits = self.model.forward_decoder(z)
            logits = (logits - logits.mean(-1, keepdim=True)) * temparature
            probs = logits.softmax(-1)
            B, L, D = probs.shape
            probs = probs.view(-1, D)
        outputs = torch.multinomial(probs, 1)
        return outputs.view(B, L)
