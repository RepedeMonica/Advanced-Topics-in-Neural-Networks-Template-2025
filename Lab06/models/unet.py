import math
import torch
import torch.nn as nn


def sinusoidal_time_embedding(timesteps, dim, max_period=10000):
    # timesteps: (b,)
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim, out_c)
        ) if time_dim is not None else None
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_c), nn.SiLU(), nn.Conv2d(in_c, out_c, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_c), nn.SiLU(), nn.Conv2d(out_c, out_c, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t=None):
        h = self.block1(x)
        if self.time_mlp is not None and t is not None:
            temb = self.time_mlp(t)
            h = h + temb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_c, out_c, time_dim)
        self.res2 = ResidualBlock(out_c, out_c, time_dim)
        self.down = nn.Conv2d(out_c, out_c, 4, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        skip = x
        x = self.down(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)
        # after upsample, we concatenate with skip: channels = out_c + skip_c
        self.res1 = ResidualBlock(out_c + skip_c, out_c, time_dim)
        self.res2 = ResidualBlock(out_c, out_c, time_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, base=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        # encoder
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)
        self.down1 = Down(base, base * 2, time_dim)   # -> skip1: C=base*2
        self.down2 = Down(base * 2, base * 4, time_dim)  # -> skip2: C=base*4
        self.mid1 = ResidualBlock(base * 4, base * 4, time_dim)
        self.mid2 = ResidualBlock(base * 4, base * 4, time_dim)
        # decoder: choose out_c to match skip dims for concatenation
        # up2: in base*4 -> up to base*4, concat with skip2 (base*4), then reduce to base*4
        self.up2 = Up(in_c=base * 4, skip_c=base * 4, out_c=base * 4, time_dim=time_dim)
        # up1: in base*4 -> up to base*2, concat with skip1 (base*2), then reduce to base*2
        self.up1 = Up(in_c=base * 4, skip_c=base * 2, out_c=base * 2, time_dim=time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base * 2), nn.SiLU(), nn.Conv2d(base * 2, in_channels, 3, padding=1)
        )
        self.time_dim = time_dim

    def forward(self, x, timesteps):
        t = sinusoidal_time_embedding(timesteps, self.time_dim)
        t = self.time_mlp(t)
        x0 = self.in_conv(x)
        x1, s1 = self.down1(x0, t)
        x2, s2 = self.down2(x1, t)
        m = self.mid1(x2, t)
        m = self.mid2(m, t)
        u2 = self.up2(m, s2, t)
        u1 = self.up1(u2, s1, t)
        out = self.out(u1)
        return out
