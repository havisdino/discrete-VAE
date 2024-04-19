import os
from torch import nn
import torch

from utils import get_model_config


class Saver:
    def __init__(self, last_k, checkpoint_interval):
        self.last_k = last_k * checkpoint_interval
    
    def _build_checkpoint(self, model, optimizer, grad_scaler, lr_scheduler, global_step):
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        elif isinstance(model, nn.Module):
            model_state_dict = model.state_dict()
        
        self.last_checkpoint = dict(
            model=model_state_dict,
            optimizer=optimizer.state_dict(),
            grad_scaler=grad_scaler.state_dict(),
            lr_scheduler=lr_scheduler.state_dict(),
            config=get_model_config(),
            global_step=global_step
        )
    
    def save(self, model, optimizer, grad_scaler, lr_scheduler, global_step):
        if not os.path.exists('./checkpoints'):
            os.makedirs('checkpoints')

        path = f'vae-{global_step}.pt'
        last_kth = f'vae-{global_step - self.last_k}.pt'

        if os.path.exists(last_kth):
            os.remove(last_kth)
            
        self._build_checkpoint(model, optimizer, grad_scaler, lr_scheduler, global_step)   
        torch.save(self.last_checkpoint, path)
        