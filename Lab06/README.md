## Lab 6
***
## Self-Supervised Learning, Autoencoders and Diffusion

This lab uses the SpaceNet-7 Multi-Temporal Urban Development imagery to explore representation learning and generative modeling:

- Train convolutional Autoencoders (AE) and Variational Autoencoders (VAE) for satellite imagery reconstruction.
- Train a basic unconditional diffusion model (DDPM) directly in pixel space.
- Exercise: implement and train a Latent Diffusion Model (LDM) that learns a diffusion process in a VAE latent space.

Resources:
- Self-Supervised Learning Overview + Medical Applications ([slides](HealthTech_Connect___Self_Supervised_Deep_Learning_Approaches_for_Medical_Image_Analysis_Presentation.pdf))
- Self-Supervised Learning for CV ([slides](https://docs.google.com/presentation/d/1UYCZFrGMcfcNQX3us-Jl5p82Hk4bRDZmWaS5XSh-qyA/edit?usp=share_link))
- Variational Autoencoders ([slides](https://docs.google.com/presentation/d/1WAaW-uY10OL9qVQx5FPsvF7ppyiBawgfr3Kd_Qj9vh8/edit?usp=sharing))
- Diffusion ([slides](https://docs.google.com/presentation/d/1RrmrAuFi2fk2OIWr8EsG9CmeoI3qQ6OFR29EzpdlYJ4/edit?usp=sharing))

---

Setup

- Python 3.9+, PyTorch 2.x, CUDA optional.
- Install requirements: `python3 -m venv .venv && source .venv/bin/activate && pip install -r Lab06/requirements.txt`

Data (SpaceNet-7)

- SpaceNet-7 is large. For this lab, use pre-extracted RGB patches (e.g., 256x256 PNGs) sampled from the monthly Planet imagery. Expected layout:
  - `<SN7_PATCHES_DIR>/train/*.png` and `<SN7_PATCHES_DIR>/val/*.png`
  - Set `SN7_PATCHES_DIR` environment variable or pass `--data_dir` to training scripts.
- If you have raw SpaceNet-7 data, see `Lab06/scripts/prepare_sn7_patches.py` for an optional patch extraction sketch (requires `rasterio`).
- Sanity check (optional): generate tiny dummy images to verify the pipeline: `python Lab06/scripts/make_dummy_data.py --out Lab06/sample_data`.

Quickstarts

- From inside the `Lab06` folder and using local paths:
- AE: `python train_ae.py --data_dir ./sample_data --epochs 5`
- VAE: `python train_vae.py --data_dir ./sample_data --epochs 5`
- DDPM: `python train_ddpm.py --data_dir ./sample_data --epochs 5`

Live Demo Notebook

- Open `Lab06/Lab06_LDM_Demo.ipynb` to run a compact, end-to-end demo of AE, VAE, DDPM, and Latent Diffusion. It auto-generates a dummy dataset if SN7 patches are not available.

Exercise: Latent Diffusion Model (LDM)

- Goal: Train diffusion in a learned latent space. Steps:
  1) Train a VAE to compress images to small latents (default 32x32x4).
  2) Freeze the VAE encoder/decoder.
  3) Train a UNet-based diffusion model on the VAE latents.
- Where to implement: see `ldm/ldm.py` and `train_ldm.py` for TODOs.
- Run (after filling TODOs): `python train_ldm.py --data_dir ./sample_data --vae_ckpt ./checkpoints/vae.pt`

What to implement (TODOs)

- In `Lab06/ldm/ldm.py`:
  - Initialize a latent-space UNet with `in_channels=latent_channels`.
  - Forward pass: predict noise given `(z_t, t)`.
- In `Lab06/train_ldm.py`:
  - Add forward diffusion on latents: `z_t = scheduler.add_noise(z0, noise, t)`.
  - Train loop steps: compute loss `MSE(pred, noise)`, optimize, log.

Notes and Tips

- Images are normalized to [0,1] and center-cropped/resized (default 256x256).
- Use small batch sizes (e.g., 16) if training on a laptop GPU; start with small epochs.
- For a multi-temporal twist, you can condition the diffusion model on month index or AOI ID by extending the dataset to return conditioning labels.
