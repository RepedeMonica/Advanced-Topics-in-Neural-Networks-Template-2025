import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base=64, latent_channels=128):
        super().__init__()
        self.stem = ConvBlock(in_channels, base)
        self.down1 = ConvBlock(base, base * 2, s=2)
        self.down2 = ConvBlock(base * 2, base * 4, s=2)
        self.down3 = ConvBlock(base * 4, base * 4)
        self.down4 = ConvBlock(base * 4, latent_channels, s=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        z = self.down4(x)
        return z


class Decoder(nn.Module):
    def __init__(self, out_channels=3, base=64, latent_channels=128):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(latent_channels, base * 4),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base * 4, base * 2),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base * 2, base),
        )
        self.out = nn.Conv2d(base, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, z):
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return self.out_act(x)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, base=64, latent_channels=128):
        super().__init__()
        self.encoder = Encoder(in_channels, base, latent_channels)
        self.decoder = Decoder(in_channels, base, latent_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
