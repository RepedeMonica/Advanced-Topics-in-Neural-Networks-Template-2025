import torch
import torch.nn as nn

from models.vae import VAE


class LatentAutoencoder(nn.Module):
    """
    Thin wrapper to load a trained VAE and expose encode/decode.
    - Infers base/latent_channels from checkpoint if not present in args
    - Freezes VAE parameters
    """

    def __init__(self, ckpt_path: str, device: str = 'cpu'):
        super().__init__()
        sd = torch.load(ckpt_path, map_location=device)
        args = sd.get('args', {}) or {}
        model_sd = sd.get('model', sd)

        # Infer latent_channels if missing
        latent_channels = args.get('latent_channels')
        if latent_channels is None:
            # try encoder.mu or logvar conv out_channels
            for k in ['encoder.mu.weight', 'encoder.logvar.weight']:
                if k in model_sd:
                    latent_channels = int(model_sd[k].shape[0])
                    break
        if latent_channels is None:
            latent_channels = 128
        latent_channels = int(latent_channels)

        # Infer base if missing
        base = args.get('base')
        if base is None:
            if 'decoder.out.weight' in model_sd:
                base = int(model_sd['decoder.out.weight'].shape[1])
            elif 'encoder.stem.bn.weight' in model_sd:
                base = int(model_sd['encoder.stem.bn.weight'].shape[0])
            else:
                base = 64
        base = int(base)

        self.vae = VAE(in_channels=3, base=base, latent_channels=latent_channels)
        # Load strictly now that shapes match base/latents
        self.vae.load_state_dict(model_sd, strict=True)
        self.vae.to(device)
        self.device = device
        self.latent_channels = latent_channels
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, x):
        # x in [0,1]
        self.vae.eval()
        mu, logvar = self.vae.encoder(x)
        z = VAE.reparameterize(mu, logvar)
        return z

    @torch.no_grad()
    def decode(self, z):
        self.vae.eval()
        x_hat = self.vae.decoder(z)
        return x_hat
