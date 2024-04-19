import os
from time import time
from typing import Literal, overload
from torch import nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.utils
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.utils.data

import config as C
from dataset import ImageDataset
from modules.vae import VAE


def init_weights(m):
    for p in m.parameters():
        nn.init.normal_(p, std=0.04)
        
        
def count_params(model):
    if isinstance(model, nn.DataParallel):
        n_params = sum(p.numel() for p in model.module.parameters())
    elif isinstance(model, nn.Module):
        n_params = sum(p.numel() for p in model.parameters())
    return n_params


def vae_summary(vae: VAE):
    decoder_nparams = count_params(vae.decoder)
    encoder_nparams = count_params(vae.encoder)
    print(f'Decoder parameters: {decoder_nparams:,}')
    print(f'Encoder parameters: {encoder_nparams:,}')
    print(f'Total parameters: {decoder_nparams + encoder_nparams}')


def flatten_imgs(imgs):
    if imgs.ndim == 3:
        return imgs.view(-1)
    elif imgs.ndim == 4:
        B = imgs.size(0)
        return imgs.view(B, -1)
    raise ValueError()


def unflatten_imgs(imgs, img_size):
    if imgs.ndim == 1:
        return imgs.view(img_size)
    elif imgs.ndim == 2:
        B = imgs.size(0)
        return imgs.view(B, *img_size)
    

def generate_file_name():
    return f'{int(time() * 1e9)}.png'


def save_images(imgs, dir, make_grid=True):
    if make_grid:
        fp = os.path.join(dir, generate_file_name())
        torchvision.utils.save_image(imgs / 255., fp)
    else:
        for img in imgs:
            fp = os.path.join(dir, generate_file_name())
            torchvision.utils.save_image(img / 255., fp)


def get_model_config(include_imgsize=True):
    config = dict(
        chw=C.chw,
        d_model=C.d_model,
        dff=C.dff,
        d_latent=C.d_latent,
        nlayers_decoder=C.nlayers_decoder,
        nheads_encoder=C.nheads_encoder,
        nblocks_encoder=C.nblocks_encoder,
        dropout=C.dropout,
        nbits=C.nbits
    )
    if include_imgsize:
        config.update(img_size=C.img_size)
    return config
        

def modify_config(config, **kwargs):
    for key, item in kwargs.items():
        setattr(config, key.upper(), item)


def create_model_from_checkpoint(checkpoint):
    config = checkpoint['config']
    modify_config(C, **config)
    print('Checkpoint loaded, default configurations might be ignored')
    return VAE(**config)


def create_model_from_default_config():
    config = get_model_config(include_imgsize=False)
    return VAE(**config)


def create_modules_from_checkpoint(checkpoint):
    model = create_model_from_checkpoint(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 10, eta_min=5e-5, T_mult=2)
    grad_scaler = torch.cuda.amp.GradScaler()

    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    grad_scaler.load_state_dict(checkpoint['grad_scaler'])

    return model, optimizer, lr_scheduler, grad_scaler


def create_modules_from_default_config():
    model = create_model_from_default_config()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 10, eta_min=5e-5, T_mult=2)
    grad_scaler = torch.cuda.amp.GradScaler()

    return model, optimizer, lr_scheduler, grad_scaler


# @overload
# def create_data_loader(*, batch_size: int, path: str) -> DataLoader:
#     ...


# @overload
# def create_data_loader(*, batch_size: int, download_data: Literal['cifar10'] | None, train: bool) -> DataLoader:
#     ...


def create_data_loader(*, batch_size, path=None, download_data=None, train=True) -> DataLoader:
    def remove_label_collate_fn(batch):
        imgs = [pair[0] for pair in batch]
        return torch.stack(imgs)
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([32, 32]),
        transforms.Lambda(lambda x: (x * 255).to(torch.long)),
        transforms.Lambda(lambda x: flatten_imgs(x))
    ])
    
    loader_config = dict(batch_size=batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
    
    if download_data == 'cifar10':
        data = torchvision.datasets.CIFAR10(root='data/', train=train, download=True, transform=transform)
        loader = DataLoader(data, collate_fn=remove_label_collate_fn, **loader_config)
    elif download_data is None:
        data = ImageDataset(path, transform)
        loader = DataLoader(data, **loader_config)
    
    return loader
