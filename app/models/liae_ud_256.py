import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 基本ブロック
# ============================================================

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.block(x), 0.1, inplace=True)


# ============================================================
# DF-style encoder（residual + sharpen 強化）
# ============================================================

class DFEncoder(nn.Module):
    def __init__(self, e_dims=256, ae_dims=768):
        super().__init__()
        ch = e_dims

        self.down1 = conv_block(3, ch)
        self.pool = nn.AvgPool2d(2)

        self.down2 = conv_block(ch, ch * 2)
        self.down3 = conv_block(ch * 2, ch * 4)
        self.down4 = conv_block(ch * 4, ch * 8)

        self.res2 = ResidualBlock(ch * 2)
        self.res3 = ResidualBlock(ch * 4)
        self.res4 = ResidualBlock(ch * 8)

        self.reduce = nn.Conv2d(ch * 8, ch * 8, 1)

        # sharpen 強化（2層）
        self.sharpen = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch * 8, ch * 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.to_latent = nn.Conv2d(ch * 8, ae_dims, 3, padding=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.pool(x)

        x = self.down2(x)
        x = self.res2(x)
        x = self.pool(x)

        x = self.down3(x)
        x = self.res3(x)
        x = self.pool(x)

        x = self.down4(x)
        x = self.res4(x)
        x = self.pool(x)

        x = self.reduce(x)
        x = self.sharpen(x)
        z = self.to_latent(x)
        return z  # (N, ae_dims, 16,16)


# ============================================================
# UD
# ============================================================

class UniformDistribution(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + self.eps
        return (x - mean) / std


# ============================================================
# PixelShuffle-based UpBlock
# ============================================================

class PSUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # in_ch -> 4*out_ch → PixelShuffle(2) → out_ch
        self.conv_ps = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.block = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(out_ch),
        )

    def forward(self, x):
        x = self.conv_ps(x)
        x = self.ps(x)
        x = self.block(x)
        return x


# ============================================================
# Decoder / MaskDecoder（PixelShuffle + Residual）
# ============================================================

class Decoder(nn.Module):
    def __init__(self, d_dims=768):
        super().__init__()
        ch = 64

        self.up1 = PSUpBlock(d_dims, ch * 8)
        self.up2 = PSUpBlock(ch * 8, ch * 4)
        self.up3 = PSUpBlock(ch * 4, ch * 2)
        self.up4 = PSUpBlock(ch * 2, ch)

        # 出力側のシャープさ強化
        self.out_block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out_block(x)


class MaskDecoder(nn.Module):
    def __init__(self, d_mask_dims=768):
        super().__init__()
        ch = 64

        self.up1 = PSUpBlock(d_mask_dims, ch * 8)
        self.up2 = PSUpBlock(ch * 8, ch * 4)
        self.up3 = PSUpBlock(ch * 4, ch * 2)
        self.up4 = PSUpBlock(ch * 2, ch)

        self.out_block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out_block(x)


# ============================================================
# Landmark head
# ============================================================

class LandmarkHead(nn.Module):
    def __init__(self, ae_dims=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ae_dims, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 136, 1),
        )

    def forward(self, z):
        out = self.net(z)
        N = out.size(0)
        out = out.mean(dim=[2, 3])
        return out.view(N, 68, 2)


# ============================================================
# LIAE_UD_256（軽量強化版）
# ============================================================

class LIAE_UD_256(nn.Module):
    def __init__(self, e_dims=256, ae_dims=768, d_dims=256, d_mask_dims=256):
        super().__init__()

        self.encoder = DFEncoder(e_dims=e_dims, ae_dims=ae_dims)
        self.post_bn = nn.Sequential(
            nn.Conv2d(ae_dims, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.ud = UniformDistribution(eps=1e-6)

        self.decoder_A = Decoder(d_dims=ae_dims)
        self.decoder_B = Decoder(d_dims=ae_dims)
        self.mask_decoder = MaskDecoder(d_mask_dims=ae_dims)

        self.lm_head = LandmarkHead(ae_dims=ae_dims)

    def encode(self, x):
        z = self.encoder(x)
        z = self.post_bn(z)
        return z

    def forward(self, img_a, img_b, lm_a=None, lm_b=None,
                warp_prob=0.0, hsv_power=0.0, noise_power=0.0, shell_power=0.0):

        za = self.encode(img_a)
        zb = self.encode(img_b)

        # decoder は生 latent（sharp）
        aa = self.decoder_A(za)
        ab = self.decoder_B(za)
        bb = self.decoder_B(zb)
        ba = self.decoder_A(zb)

        # mask だけ UD
        za_ud = self.ud(za)
        zb_ud = self.ud(zb)
        mask_a_pred = self.mask_decoder(za_ud)
        mask_b_pred = self.mask_decoder(zb_ud)

        lm_a_pred = self.lm_head(za)
        lm_b_pred = self.lm_head(zb)

        return aa, bb, ab, ba, mask_a_pred, mask_b_pred, lm_a_pred, lm_b_pred
