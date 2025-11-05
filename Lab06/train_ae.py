import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.sn7_dataset import get_datasets
from data.util import ensure_or_create_synth_dataset
from models.ae import AutoEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='./sample_data')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--latent_channels', type=int, default=128)
    p.add_argument('--out', type=str, default='./checkpoints/ae.pt')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensure_or_create_synth_dataset(args.data_dir, size=args.image_size)
    train_ds, val_ds = get_datasets(args.data_dir, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = AutoEncoder(in_channels=3, base=64, latent_channels=args.latent_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.L1Loss()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        total = 0.0
        for batch in pbar:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            x_hat, _ = model(x)
            loss = crit(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n = 0
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                x_hat, _ = model(x)
                l = crit(x_hat, x)
                val_loss += l.item() * x.size(0)
                n += x.size(0)
        print(f"Val L1: {val_loss / max(1,n):.4f}")

    torch.save({'model': model.state_dict(), 'args': vars(args)}, args.out)
    print(f"Saved AE to {args.out}")


if __name__ == '__main__':
    main()
