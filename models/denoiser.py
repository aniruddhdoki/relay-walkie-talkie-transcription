"""Small U-Net denoiser for spectrogram enhancement."""

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """Encoder block: Conv2d -> BN -> ReLU -> MaxPool; returns pre-pool features for skip."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        return self.pool(features), features


class DecoderBlock(nn.Module):
    """Decoder block: Upsample -> concat skip -> Conv2d -> BN -> ReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from skip connections
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Denoiser(nn.Module):
    """Small U-Net for spectrogram denoising. Input/output: [B, 1, F, T]."""

    def __init__(
        self,
        base_channels: int = 32,
        n_levels: int = 4,
    ):
        super().__init__()
        self.n_levels = n_levels
        chs = [1] + [base_channels * (2**i) for i in range(n_levels)]  # 1, 32, 64, 128, 256

        self.encoder_blocks = nn.ModuleList()
        for i in range(n_levels):
            self.encoder_blocks.append(EncoderBlock(chs[i], chs[i + 1]))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(chs[-1], chs[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(chs[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(chs[-1] * 2, chs[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(chs[-1]),
            nn.ReLU(inplace=True),
        )

        self.decoder_blocks = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            in_ch = chs[i + 1] if i < n_levels - 1 else chs[-1]
            skip_ch = chs[i + 1]
            out_ch = chs[i]
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))

        self.final = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        h = x
        for block in self.encoder_blocks:
            h, skip = block(h)
            skips.append(skip)

        h = self.bottleneck(h)

        for i, block in enumerate(self.decoder_blocks):
            h = block(h, skips[-(i + 1)])

        return nn.functional.relu(self.final(h))
