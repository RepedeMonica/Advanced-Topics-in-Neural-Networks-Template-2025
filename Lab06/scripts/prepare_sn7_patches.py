"""
Optional sketch for extracting 256x256 RGB patches from raw SpaceNet-7 GeoTIFFs.
Requires `rasterio` and `numpy`. This is a simplified example and may need
adjustments for the exact SN7 folder structure.

Usage:
  python Lab06/scripts/prepare_sn7_patches.py \
    --sn7_root /path/to/SN7_root \
    --out_dir /path/to/sn7_patches \
    --size 256 --stride 256 --limit_per_image 200

Expected output:
  out_dir/train/*.png and out_dir/val/*.png
"""

import os
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sn7_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--limit_per_image', type=int, default=0, help='0 = no limit')
    args = parser.parse_args()

    try:
        import rasterio
        import numpy as np
        from PIL import Image
    except Exception as e:
        raise SystemExit("This script requires rasterio, numpy, pillow. Install them first.")

    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'val'), exist_ok=True)

    # naive search for GeoTIFFs
    tifs = []
    for dp, _, files in os.walk(args.sn7_root):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                tifs.append(os.path.join(dp, f))

    rng = random.Random(123)
    for tif_path in tifs:
        try:
            with rasterio.open(tif_path) as src:
                # Read first 3 bands as RGB (adjust if needed)
                bands = src.read([1, 2, 3])  # (3, H, W)
                H, W = bands.shape[1], bands.shape[2]
                count = 0
                for y in range(0, H - args.size + 1, args.stride):
                    for x in range(0, W - args.size + 1, args.stride):
                        patch = bands[:, y:y+args.size, x:x+args.size]
                        # simple contrast stretch to 0-255
                        p = np.clip((patch - patch.min()) / (patch.ptp() + 1e-5) * 255, 0, 255).astype(np.uint8)
                        img = np.transpose(p, (1, 2, 0))  # HWC
                        split = 'val' if rng.random() < args.val_ratio else 'train'
                        name = f"{os.path.basename(tif_path).split('.')[0]}_{y}_{x}.png"
                        out_path = os.path.join(args.out_dir, split, name)
                        Image.fromarray(img).save(out_path)
                        count += 1
                        if args.limit_per_image and count >= args.limit_per_image:
                            break
                    if args.limit_per_image and count >= args.limit_per_image:
                        break
        except Exception:
            # ignore files rasterio can't open
            continue

    print(f"Saved patches under {args.out_dir}")


if __name__ == '__main__':
    main()

