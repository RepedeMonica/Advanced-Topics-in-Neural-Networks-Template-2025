import torch
import torch.nn as nn

from models.unet import UNet


class LatentDiffusion(nn.Module):
    """
    Latent Diffusion Model (unconditional):
    - Operates in VAE latent space with `latent_channels`.
    - Uses a UNet backbone to predict noise for DDPM objective.
    """

    def __init__(self, latent_channels: int, time_dim: int = 256, base: int = 64):
        super().__init__()
        self.unet = UNet(in_channels=latent_channels, base=base, time_dim=time_dim)

    def forward(self, z_t, t):
        return self.unet(z_t, t)
