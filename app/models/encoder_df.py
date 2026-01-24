# liae_ud_256.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DF-style encoder（UD 用に最適化）
# ============================================================

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
    )


class DFEncoder(nn.Module):
    """
    入力: (N, 3, 256, 256)
    出力: (N, ae_dims, 16, 16)
    """

    def __init__(self, model_size=256, e_dims=256, ae_dims=768):
        super().__init__()

        ch = e_dims

        self.down1 = conv_block(3, ch)          # 256 -> 256
        self.down2 = conv_block(ch, ch * 2)     # 256 -> 128
        self.down3 = conv_block(ch * 2, ch * 4) # 128 -> 64
        self.down4 = conv_block(ch * 4, ch * 8) # 64  -> 32

        self.pool = nn.AvgPool2d(2)             # 4 回で 256 -> 16

        # bottleneck（圧縮を滑らかに）
        self.reduce = nn.Conv2d(ch * 8, ch * 8, 1)
        self.to_latent = nn.Conv2d(ch * 8, ae_dims, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.down1(x)
        x = self.pool(x)  # 256 -> 128

        x = self.down2(x)
        x = self.pool(x)  # 128 -> 64

        x = self.down3(x)
        x = self.pool(x)  # 64 -> 32

        x = self.down4(x)
        x = self.pool(x)  # 32 -> 16

        x = self.reduce(x)
        z = self.to_latent(x)
        return z  # (N, ae_dims, 16, 16)


# ============================================================
# UD（Uniform Distribution）層（少し弱め）
# ============================================================

class UniformDistribution(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + self.eps
        return (x - mean) / std


# ============================================================
# Decoder / MaskDecoder
# ============================================================

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, d_dims=768):
        super().__init__()
        ch = 64

        self.up1 = UpBlock(d_dims, ch * 8)   # 16 -> 32
        self.up2 = UpBlock(ch * 8, ch * 4)   # 32 -> 64
        self.up3 = UpBlock(ch * 4, ch * 2)   # 64 -> 128
        self.up4 = UpBlock(ch * 2, ch)       # 128 -> 256

        self.out_block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
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

        self.up1 = UpBlock(d_mask_dims, ch * 8)
        self.up2 = UpBlock(ch * 8, ch * 4)
        self.up3 = UpBlock(ch * 4, ch * 2)
        self.up4 = UpBlock(ch * 2, ch)

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
# LIAE_UD_256 本体（TrainerLIAE 互換）
# ============================================================

class LIAE_UD_256(nn.Module):
    def __init__(self, e_dims=256, ae_dims=768, d_dims=256, d_mask_dims=256):
        super().__init__()

        # encoder は DFEncoder ベース
        self.encoder = DFEncoder(model_size=256, e_dims=e_dims, ae_dims=ae_dims)

        # bottleneck 後に UD をかける（encoder を固めすぎない）
        self.ud = UniformDistribution(eps=1e-3)

        # decoder 入力側の bottleneck（軽く整形）
        self.post_bn = nn.Sequential(
            nn.Conv2d(ae_dims, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.decoder_A = Decoder(d_dims=ae_dims)
        self.decoder_B = Decoder(d_dims=ae_dims)
        self.mask_decoder = MaskDecoder(d_mask_dims=ae_dims)

        # Landmark head（LIAEModel と同じ 68×2）
        self.lm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ae_dims, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 68 * 2),
        )

    # --------------------------------------------------------
    # encode
    # --------------------------------------------------------
    def encode(self, x):
        # DFEncoder で latent を作る
        z = self.encoder(x)          # (N, ae_dims, 16, 16)
        z = self.ud(z)               # 分布を正規化（弱め）
        z = self.post_bn(z)          # 軽く整形
        return z

    # --------------------------------------------------------
    # forward（TrainerLIAE と互換）
    # --------------------------------------------------------
    def forward(
        self,
        img_a,
        img_b,
        lm_a=None,
        lm_b=None,
        warp_prob=0.0,
        hsv_power=0.0,
        noise_power=0.0,
        shell_power=0.0,
    ):
        za = self.encode(img_a)
        zb = self.encode(img_b)

        # decode
        aa = self.decoder_A(za)
        ab = self.decoder_B(za)
        bb = self.decoder_B(zb)
        ba = self.decoder_A(zb)

        # mask
        mask_a_pred = self.mask_decoder(za)
        mask_b_pred = self.mask_decoder(zb)

        # landmark prediction
        N = img_a.size(0)
        lm_a_pred = self.lm_head(za).view(N, 68, 2)
        lm_b_pred = self.lm_head(zb).view(N, 68, 2)

        return (
            aa, bb, ab, ba,
            mask_a_pred, mask_b_pred,
            lm_a_pred, lm_b_pred,
        )
