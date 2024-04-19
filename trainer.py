from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler
from dataclasses import dataclass

from tqdm import tqdm

from logger import Logger
from modules.vae import VAE
from sampler import Sampler
from saver import Saver
from utils import save_images, unflatten_imgs
import config as C


@dataclass
class Trainer:
    model: VAE
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    grad_scaler: GradScaler
    device: Any
    grad_accum_interval: int
    checkpoint_interval: int
    use_amp: bool
    global_step: int = 0
    save_last_k_checkpoints: int = 5
    
    def __post_init__(self):
        self.substep = 0
        self.info = dict()
        self.model.to(self.device)
        self.saver = Saver(self.save_last_k_checkpoints, self.checkpoint_interval)
    
    @staticmethod
    def get_reconstruction_loss(logits, x):
        B = x.size(0)
        x = x.view(B, -1)
        logits = logits.view(B, -1)
        return F.cross_entropy(logits, x)
    
    @staticmethod
    def get_kl_loss(mean, logstd, z):
        t1 = 0.5 * z.square().sum()
        t2 = -logstd - 0.5 * ((z - mean) / logstd).square()
        t2 = t2.sum()
        kl_loss = t1 + t2
        return kl_loss
    
    @staticmethod
    def get_vae_loss(mean, logstd, z, logits, x):
        rec_loss = Trainer.get_reconstruction_loss(logits, x)
        kl_loss = Trainer.get_kl_loss(mean, logstd, z)
        loss = rec_loss + kl_loss
        return loss, rec_loss, kl_loss
    
    def train_step(self, x):
        self.model.train()
        with torch.autocast(self.device, torch.float16, self.use_amp):
            mean, logstd, z, logits = self.model(x)
            self.loss, self.rec_loss, self.kl_loss = Trainer.get_vae_loss(mean, logstd, z, logits, x)
            self.loss /= self.grad_accum_interval
        self.grad_scaler.scale(self.loss).backward()
        self.substep += 1
        self._detach_component_losses()
        
    def _detach_component_losses(self):
        self.rec_loss = self.rec_loss.detach().item()
        self.kl_loss = self.kl_loss.detach().item()
        
    def accumulate_gradients(self):
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.global_step += 1
        
        self.info.update(
            dict(
                step=self.global_step,
                loss=self.global_batch_loss(),
                rec_loss=self.rec_loss,
                kl_loss=self.kl_loss
            )
        )
    
    def global_batch_loss(self):
        return self.loss.detach().item() * self.grad_accum_interval
            
    def fit(self, trainloader, valloader, nsteps, logger: Logger | None = None):
        nsteps += self.global_step
        
        print(f'Accumulating gradients after {self.grad_accum_interval} substeps')
        
        sampler = Sampler(self.model, self.device, self.use_amp)
        
        data_iter = iter(trainloader)
        self.optimizer.zero_grad()
        
        bar = tqdm()
        
        while self.substep < nsteps:
            try:
                x = next(data_iter)
            except StopIteration:
                x = iter(trainloader)
                continue
            x.to(self.device)
            print(x.device)
            self.train_step(x)
            
            if self.substep % self.grad_accum_interval == 0:
                self.accumulate_gradients()
                
                if self.global_step % self.checkpoint_interval == 0:
                    # Validate and save model here
                    imgs = sampler.sample(nsamples=36)
                    imgs = unflatten_imgs(imgs, C.img_size)
                    save_images(imgs, './results')
                    
                    self.saver.save(
                        self.model, self.optimizer, self.grad_scaler,
                        self.lr_scheduler, self.global_step
                    )
                
                bar.set_postfix(self.info)
                if logger is not None:
                    logger.log(**self.info)
        bar.close()
        if logger is not None:
            logger.close()
 