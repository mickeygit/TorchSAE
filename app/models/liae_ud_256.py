import torch
import torch.nn as nn
import torch.nn.functional as F
from app.utils.preview_utils import to_image_tensor, prepare_mask
from app.utils.model_output import ModelOutput

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
# PixelShuffle UpBlock
# ============================================================

class PSUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
# Decoder（EXP 強化版）
# ============================================================

class Decoder(nn.Module):
    def __init__(self, d_dims=768):
        super().__init__()
        ch = 64

        self.up1 = PSUpBlock(d_dims, ch * 8)
        self.up2 = PSUpBlock(ch * 8, ch * 4)
        self.up3 = PSUpBlock(ch * 4, ch * 2)
        self.up4 = PSUpBlock(ch * 2, ch)

        # EXP ゲート（強化）
        self.exp_gate1 = nn.Sequential(
            nn.Conv2d(d_dims, d_dims, 1),
            nn.Sigmoid()
        )
        self.exp_gate2 = nn.Sequential(
            nn.Conv2d(d_dims, ch * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, z_exp):

        if z_exp.shape[2:] != x.shape[2:]:
            z_exp = F.interpolate(z_exp, size=x.shape[2:], mode="nearest")

        # ★ ゲート強度 2.0
        gate1 = self.exp_gate1(z_exp)
        x = x * (1.0 + gate1 * 2.0)

        # ★ EXP 注入 3.0
        x = x + z_exp * 3.0

        x = self.up1(x)

        gate2 = self.exp_gate2(z_exp)
        if gate2.shape[2:] != x.shape[2:]:
            gate2 = F.interpolate(gate2, size=x.shape[2:], mode="nearest")

        x = x * (1.0 + gate2 * 2.0)

        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return x

# ============================================================
# MaskDecoder（同じ強化）
# ============================================================

class MaskDecoder(nn.Module):
    def __init__(self, d_mask_dims=768):
        super().__init__()
        ch = 64

        self.up1 = PSUpBlock(d_mask_dims, ch * 8)
        self.up2 = PSUpBlock(ch * 8, ch * 4)
        self.up3 = PSUpBlock(ch * 4, ch * 2)
        self.up4 = PSUpBlock(ch * 2, ch)

        self.exp_gate1 = nn.Sequential(
            nn.Conv2d(d_mask_dims, d_mask_dims, 1),
            nn.Sigmoid()
        )
        self.exp_gate2 = nn.Sequential(
            nn.Conv2d(d_mask_dims, ch * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, z_exp):

        if z_exp.shape[2:] != x.shape[2:]:
            z_exp = F.interpolate(z_exp, size=x.shape[2:], mode="nearest")

        x = x * (1.0 + self.exp_gate1(z_exp) * 2.0)
        x = x + z_exp * 3.0

        x = self.up1(x)

        gate2 = self.exp_gate2(z_exp)
        if gate2.shape[2:] != x.shape[2:]:
            gate2 = F.interpolate(gate2, size=x.shape[2:], mode="nearest")

        x = x * (1.0 + gate2 * 2.0)

        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return x

# ============================================================
# DFEncoder（浅層 EXP 分岐）
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

        self.sharpen = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch * 8, ch * 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ★ EXP は浅層（down3）から
        self.to_exp_shallow = nn.Sequential(
            nn.Conv2d(ch * 4, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(ae_dims),
            nn.Conv2d(ae_dims, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.to_id = nn.Sequential(
            nn.Conv2d(ch * 8, ae_dims, 1),
        )

    def forward(self, x):
        x1 = self.down1(x)
        x1p = self.pool(x1)

        x2 = self.down2(x1p)
        x2r = self.res2(x2)
        x2p = self.pool(x2r)

        x3 = self.down3(x2p)
        x3r = self.res3(x3)
        x3p = self.pool(x3r)

        x4 = self.down4(x3p)
        x4r = self.res4(x4)
        x4p = self.pool(x4r)

        # ID（深層）
        x_id = self.reduce(x4p)
        x_id = self.sharpen(x_id)
        z_id = self.to_id(x_id)

        # EXP（浅層）
        z_exp = self.to_exp_shallow(x3r)

        return z_exp, z_id

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
        return (x - mean) / (std * 0.8 + self.eps)

# ============================================================
# LIAE_UD_256（EXP 強化微調整版）
# ============================================================

class LIAE_UD_256(nn.Module):
    def __init__(self, e_dims=256, ae_dims=768):
        super().__init__()

        self.id_encoder = DFEncoder(e_dims=e_dims, ae_dims=ae_dims)
        self.ud = UniformDistribution()

        self.decoder_A = Decoder(d_dims=ae_dims * 2)
        self.decoder_B = Decoder(d_dims=ae_dims * 2)
        self.mask_decoder = MaskDecoder(d_mask_dims=ae_dims * 2)

        # ★ EXP を強く（6.0）
        self.exp_gain = 6.0
        self.id_scale = 0.1
        self.id_inject_gain = 0.2

    def encode(self, x):
        z_exp_raw, z_id_raw = self.id_encoder(x)

        z_exp = F.interpolate(z_exp_raw, size=(16, 16), mode="area")

        z_id = z_id_raw * self.id_scale

        z_exp = self.ud(z_exp)
        z_id  = self.ud(z_id)

        # ★ EXP 強化
        z_exp = z_exp * self.exp_gain

        return z_exp, z_id

    def forward(self, img_a, img_b):

        zA_exp, zA_id = self.encode(img_a)
        zB_exp, zB_id = self.encode(img_b)

        zA_id = zA_id + zA_exp * self.id_inject_gain
        zB_id = zB_id + zB_exp * self.id_inject_gain

        zA_full = torch.cat([zA_id, zA_exp], dim=1)
        zB_full = torch.cat([zB_id, zB_exp], dim=1)

        zA_exp_full = torch.cat([zA_exp, zA_exp], dim=1)
        zB_exp_full = torch.cat([zB_exp, zB_exp], dim=1)

        aa = self.decoder_A(zA_full, zA_exp_full)
        bb = self.decoder_B(zB_full, zB_exp_full)

        ab_full = torch.cat([zB_id, zA_exp], dim=1)
        ba_full = torch.cat([zA_id, zB_exp], dim=1)

        ab = self.decoder_B(ab_full, zA_exp_full)
        ba = self.decoder_A(ba_full, zB_exp_full)

        mask_a_pred = self.mask_decoder(zA_full, zA_exp_full)
        mask_b_pred = self.mask_decoder(zB_full, zB_exp_full)

        return ModelOutput(
            aa=aa, bb=bb, ab=ab, ba=ba,
            mask_a_pred=mask_a_pred,
            mask_b_pred=mask_b_pred,
        )
