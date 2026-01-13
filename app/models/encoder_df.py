import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
    )


class DFEncoder(nn.Module):
    """
    DF encoder (本家 SAEHD の DF を参考にした最小構成)
    入力: (N, 3, model_size, model_size)
    出力: latent tensor (N, ae_dims, H/16, W/16)
    """

    def __init__(self, model_size=128, e_dims=64, ae_dims=256):
        super().__init__()

        ch = e_dims

        self.down1 = conv_block(3, ch)
        self.down2 = conv_block(ch, ch * 2)
        self.down3 = conv_block(ch * 2, ch * 4)
        self.down4 = conv_block(ch * 4, ch * 8)

        self.pool = nn.AvgPool2d(2)

        # bottleneck
        self.to_latent = nn.Conv2d(ch * 8, ae_dims, 3, padding=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.pool(x)

        x = self.down2(x)
        x = self.pool(x)

        x = self.down3(x)
        x = self.pool(x)

        x = self.down4(x)
        x = self.pool(x)

        z = self.to_latent(x)
        return z
