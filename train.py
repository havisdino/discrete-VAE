from argparse import ArgumentParser
import torch

from trainer import Trainer
from utils import (create_data_loader, create_modules_from_checkpoint,
                   create_modules_from_default_config, init_weights, vae_summary)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--traindata', type=str, required=True)
    parser.add_argument('--valdata', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--from-checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--grad-accum-interval', type=int, default=1)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--save_last_k_checkpoints', type=int, default=5)
    parser.add_argument('--use-amp', type=bool, default=True)
    parser.add_argument('--nsteps', type=int, default=1000)

    args = parser.parse_args()

    if args.from_checkpoint is not None:
        checkpoint = torch.load(args.from_checkpoint, args.device)

        model, optimizer, lr_scheduler, grad_scaler = create_modules_from_checkpoint(
            checkpoint)
        init_step = checkpoint['global_step']

    else:
        model, optimizer, lr_scheduler, grad_scaler = create_modules_from_default_config()
        model.apply(init_weights)
        init_step = 0
    
    vae_summary(model)

    available_data = ['cifar10']
    if args.traindata in available_data and args.valdata in available_data:
        trainloader = create_data_loader(batch_size=args.batch_size, download_data=args.traindata, train=True)
        valloader = create_data_loader(batch_size=args.batch_size, download_data=args.valdata, train=False)
    else:
        trainloader = create_data_loader(path=args.traindata, batch_size=args.batch_size)
        valloader = create_data_loader(path=args.valdata, batch_size=args.batch_size)

    trainer = Trainer(
        model, optimizer, lr_scheduler, grad_scaler, args.device, args.grad_accum_interval,
        args.checkpoint_interval, args.use_amp, init_step, args.save_last_k_checkpoints
    )

    trainer.fit(trainloader, valloader, args.nsteps)
