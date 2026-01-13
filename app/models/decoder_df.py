import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
    )


class DFDecoder(nn.Module):
    """
    DF decoder (本家 SAEHD の DF を参考にした最小構成)
    latent → 画像再構成
    """

    def __init__(self, model_size=128, d_dims=64, d_mask_dims=32):
        super().__init__()

        ch = d_dims

        # upsampling blocks
        self.up1 = conv_block(256, ch * 8)
        self.up2 = conv_block(ch * 8, ch * 4)
        self.up3 = conv_block(ch * 4, ch * 2)
        self.up4 = conv_block(ch * 2, ch)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # final RGB output
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.up1(z)
        x = self.upsample(x)

        x = self.up2(x)
        x = self.upsample(x)

        x = self.up3(x)
        x = self.upsample(x)

        x = self.up4(x)
        x = self.upsample(x)

        out = self.to_rgb(x)
        return out
