from argparse import ArgumentParser
import os
from time import time

import torch

from sampler import Sampler
from utils import create_model_from_checkpoint, save_images, unflatten_imgs


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-amp', type=bool, default=True)
    parser.add_argument('--temparature', type=float, default=1.0)
    parser.add_argument('--grid', type=bool, default=True)

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, args.device)
    img_size = checkpoint['config']['img_size']
    model = create_model_from_checkpoint(checkpoint)

    sampler = Sampler(model, args.device, args.use_amp)
    imgs = sampler.sample(args.nsamples, args.temparature)

    imgs = unflatten_imgs(imgs, img_size)
    
    dir = './result'
    if os.path.exists(dir):
        os.makedirs(dir)
    
    save_images(imgs, dir, args.grid)
    