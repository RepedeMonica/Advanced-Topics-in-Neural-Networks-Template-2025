"""
Generate a tiny dummy dataset (train/val PNGs) to sanity-check the pipeline.

Usage:
  python Lab06/scripts/make_dummy_data.py --out Lab06/sample_data --n 200
"""

import os
import argparse
import random
from PIL import Image, ImageDraw


def make_img(sz=256, seed=None):
    rng = random.Random(seed)
    img = Image.new('RGB', (sz, sz), (rng.randint(0, 20), rng.randint(0, 20), rng.randint(0, 20)))
    d = ImageDraw.Draw(img)
    # draw random rectangles and circles
    for _ in range(5):
        x0, y0 = rng.randint(0, sz-64), rng.randint(0, sz-64)
        x1, y1 = x0 + rng.randint(16, 128), y0 + rng.randint(16, 128)
        color = (rng.randint(50,255), rng.randint(50,255), rng.randint(50,255))
        if rng.random() < 0.5:
            d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        else:
            d.ellipse([x0, y0, x1, y1], outline=color, width=3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--size', type=int, default=256)
    args = ap.parse_args()

    train_dir = os.path.join(args.out, 'train')
    val_dir = os.path.join(args.out, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    n_val = int(args.n * args.val_ratio)
    n_train = args.n - n_val
    for i in range(n_train):
        img = make_img(args.size, seed=i)
        img.save(os.path.join(train_dir, f'{i:05d}.png'))
    for i in range(n_val):
        img = make_img(args.size, seed=10_000 + i)
        img.save(os.path.join(val_dir, f'{i:05d}.png'))
    print(f"Wrote {n_train} train and {n_val} val images to {args.out}")


if __name__ == '__main__':
    main()

