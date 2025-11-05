import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.sn7_dataset import get_datasets
from data.util import ensure_or_create_synth_dataset
from models.unet import UNet
from models.ddpm import LinearNoiseScheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='./sample_data')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--out', type=str, default='./checkpoints/ddpm_unet.pt')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensure_or_create_synth_dataset(args.data_dir, size=args.image_size)
    train_ds, _ = get_datasets(args.data_dir, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = UNet(in_channels=3, base=64, time_dim=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    noise_pred_loss = nn.MSELoss()

    scheduler = LinearNoiseScheduler(timesteps=args.timesteps, device=device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            # scale to [-1,1] for diffusion model
            x = x * 2 - 1
            t = torch.randint(0, scheduler.timesteps, (x.size(0),), device=device).long()
            noise = torch.randn_like(x)
            x_t = scheduler.add_noise(x, noise, t)
            noise_pred = model(x_t, t)
            loss = noise_pred_loss(noise_pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

    torch.save({'model': model.state_dict(), 'args': vars(args)}, args.out)
    print(f"Saved DDPM UNet to {args.out}")


if __name__ == '__main__':
    main()
