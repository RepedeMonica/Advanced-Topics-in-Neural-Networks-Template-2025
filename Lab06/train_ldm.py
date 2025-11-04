import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.sn7_dataset import get_datasets
from data.util import ensure_or_create_synth_dataset
from models.ddpm import LinearNoiseScheduler
from ldm.latent_autoencoder import LatentAutoencoder
from ldm.ldm import LatentDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='./sample_data')
    p.add_argument('--vae_ckpt', type=str, required=True, help='Path to trained VAE checkpoint (from train_vae.py)')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--out', type=str, default='./checkpoints/ldm.pt')
    p.add_argument('--latent_base', type=int, default=64, help='Base channels inside latent UNet')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Load datasets
    ensure_or_create_synth_dataset(args.data_dir, size=args.image_size)
    train_ds, _ = get_datasets(args.data_dir, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 2) Load frozen VAE encoder/decoder
    vae = LatentAutoencoder(args.vae_ckpt, device=device)
    latent_channels = vae.latent_channels

    # 3) Initialize latent diffusion model
    ldm = LatentDiffusion(latent_channels=latent_channels, time_dim=256, base=args.latent_base)
    ldm = ldm.to(device)

    opt = torch.optim.AdamW(ldm.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    scheduler = LinearNoiseScheduler(timesteps=args.timesteps, device=device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 4) Training loop over latents
    for epoch in range(args.epochs):
        ldm.train()
        pbar = tqdm(train_loader, desc=f'LDM Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)  # [0,1]
            with torch.no_grad():
                z0 = vae.encode(x)  # B, C, H/8, W/8 (approx)

            t = torch.randint(0, scheduler.timesteps, (z0.size(0),), device=device).long()
            noise = torch.randn_like(z0)

            z_t = scheduler.add_noise(z0, noise, t)
            pred = ldm(z_t, t)
            loss = loss_fn(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss))

    # 5) Save checkpoint
    torch.save({'model': ldm.state_dict(), 'args': vars(args), 'latent_channels': latent_channels}, args.out)
    print(f"Saved LDM to {args.out}")


if __name__ == '__main__':
    main()
