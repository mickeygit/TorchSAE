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

        self.exp_gate1 = nn.Sequential(
            nn.Conv2d(d_dims, d_dims, 1),
            nn.Sigmoid()
        )
        self.exp_gate2 = nn.Sequential(
            nn.Conv2d(d_dims, ch * 8, 1),
            nn.Sigmoid()
        )

        self.out_block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1, bias=False),
        )


    def forward(self, x, z_exp):

        if z_exp.shape[2:] != x.shape[2:]:
            z_exp = F.interpolate(z_exp, size=x.shape[2:], mode="nearest")

        # ★ EXP の強度を弱める（3.0 → 1.0）
        x = x + z_exp * 1.0

        # ★ ゲートの強度も弱める（2.0 → 1.0）
        gate1 = self.exp_gate1(z_exp)
        x = x * (1.0 + gate1 * 1.0)

        x = self.up1(x)

        gate2 = self.exp_gate2(z_exp)
        if gate2.shape[2:] != x.shape[2:]:
            gate2 = F.interpolate(gate2, size=x.shape[2:], mode="nearest")

        # ★ ここも 2.0 → 1.0
        x = x * (1.0 + gate2 * 1.0)

        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return self.out_block(x)


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

        self.out_block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, x, z_exp):

        if z_exp.shape[2:] != x.shape[2:]:
            z_exp = F.interpolate(z_exp, size=x.shape[2:], mode="nearest")

        # ★ EXP の強度を弱める（3.0 → 1.0）
        x = x + z_exp * 1.0

        # ★ ゲートの強度も弱める（2.0 → 1.0）
        x = x * (1.0 + self.exp_gate1(z_exp) * 1.0)

        x = self.up1(x)

        gate2 = self.exp_gate2(z_exp)
        if gate2.shape[2:] != x.shape[2:]:
            gate2 = F.interpolate(gate2, size=x.shape[2:], mode="nearest")

        # ★ ここも 2.0 → 1.0
        x = x * (1.0 + gate2 * 1.0)

        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return self.out_block(x)

# ============================================================
# DFEncoder（浅層 EXP 分岐）
# ============================================================

class DFEncoder(nn.Module):
    def __init__(self, e_dims=256, ae_dims=768):
        super().__init__()
        ch = e_dims

        # -------------------------
        # Down blocks
        # -------------------------
        self.down1 = conv_block(3, ch)            # 256
        self.pool = nn.AvgPool2d(2)

        self.down2 = conv_block(ch, ch * 2)       # 128
        self.down3 = conv_block(ch * 2, ch * 4)   # 64
        self.down4 = conv_block(ch * 4, ch * 8)   # 32

        self.res2 = ResidualBlock(ch * 2)
        self.res3 = ResidualBlock(ch * 4)
        self.res4 = ResidualBlock(ch * 8)

        # -------------------------
        # ID head（深層）
        # -------------------------
        self.reduce = nn.Conv2d(ch * 8, ch * 8, 1)
        self.sharpen = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch * 8, ch * 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.to_id = nn.Conv2d(ch * 8, ae_dims, 1)

        # -------------------------
        # EXP head（浅層＋高周波差分）
        # -------------------------
        self.to_exp_shallow = nn.Sequential(
            nn.Conv2d(ch * 4, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(ae_dims),
            nn.Conv2d(ae_dims, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        # -------------------------
        # Down path
        # -------------------------
        x1 = self.down1(x)
        x1p = self.pool(x1)

        x2 = self.down2(x1p)
        x2r = self.res2(x2)
        x2p = self.pool(x2r)

        x3 = self.down3(x2p)
        x3r = self.res3(x3)        # ← 浅層特徴（64×64）
        x3p = self.pool(x3r)

        x4 = self.down4(x3p)
        x4r = self.res4(x4)
        x4p = self.pool(x4r)       # 16×16

        # -------------------------
        # ID（深層）
        # -------------------------
        x_id = self.reduce(x4p)
        x_id = self.sharpen(x_id)
        z_id = self.to_id(x_id)    # [B,768,16,16]

        # -------------------------
        # EXP（浅層＋高周波差分）
        # -------------------------
        # 1) ぼかし（低周波）
        blur = F.avg_pool2d(x3r, kernel_size=5, stride=1, padding=2)

        # 2) 高周波差分（表情の局所変化だけ残す）
        high = x3r - blur

        # 3) EXP head に通す
        z_exp = self.to_exp_shallow(high)   # [B,768,64,64]

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
# Landmark head
# ============================================================

class LandmarkHead(nn.Module):
    def __init__(self, ae_dims=768):
        super().__init__()
        in_ch = ae_dims * 2

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 136, 1),
        )

    def forward(self, z):
        out = self.net(z)
        N = out.size(0)
        out = out.mean(dim=[2, 3])
        return out.view(N, 68, 2)

# ============================================================
# LIAE_UD_256（微調整＋lm_head 復活版）
# ============================================================

class LIAE_UD_256(nn.Module):
    def __init__(self, e_dims=256, ae_dims=768, d_dims=256, d_mask_dims=256):
        super().__init__()

        self.id_encoder = DFEncoder(e_dims=e_dims, ae_dims=ae_dims)
        self.ud = UniformDistribution()

        self.decoder_A = Decoder(d_dims=ae_dims * 2)
        self.decoder_B = Decoder(d_dims=ae_dims * 2)
        self.mask_decoder = MaskDecoder(d_mask_dims=ae_dims * 2)

        self.lm_head = LandmarkHead(ae_dims=ae_dims)

        self.exp_gain = 2.0
        self.id_scale = 0.02
        self.id_inject_gain = 0.2

        self.max_mode = False   # ← デフォルトは OFF


    def encode(self, x):
        z_exp_raw, z_id_raw = self.id_encoder(x)

        z_exp = F.interpolate(z_exp_raw, size=(16, 16), mode="area")

        z_id = z_id_raw * self.id_scale

        z_exp = self.ud(z_exp)
        z_id  = self.ud(z_id)

        z_exp = z_exp * self.exp_gain

        return z_exp, z_id

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
        zA_exp, zA_id = self.encode(img_a)
        zB_exp, zB_id = self.encode(img_b)

        zA_id = zA_id + zA_exp * self.id_inject_gain
        zB_id = zB_id + zB_exp * self.id_inject_gain

        zA_full = torch.cat([zA_exp, zA_id], dim=1)
        zB_full = torch.cat([zB_exp, zB_id], dim=1)




        zA_exp_full = torch.cat([zA_exp, zA_exp], dim=1)
        zB_exp_full = torch.cat([zB_exp, zB_exp], dim=1)

        aa = self.decoder_A(zA_full, zA_exp_full)
        bb = self.decoder_B(zB_full, zB_exp_full)

        # ============================
        # ★ SWAP 部分（通常 or MAX モード）
        # ============================

        if self.max_mode:
            # -----------------------------------------
            # MAX モード：ID を完全に殺して EXP だけで生成
            # -----------------------------------------
            zero_id_A = torch.zeros_like(zA_id)
            zero_id_B = torch.zeros_like(zB_id)

            # A→B：B の ID を殺して A の EXP だけで生成
            ab_full = torch.cat([zA_exp, zero_id_B], dim=1)
            ab = self.decoder_B(ab_full, zA_exp_full)

            # B→A：A の ID を殺して B の EXP だけで生成
            ba_full = torch.cat([zB_exp, zero_id_A], dim=1)
            ba = self.decoder_A(ba_full, zB_exp_full)


        else:
            # -----------------------------------------
            # 通常モード（今まで通り）
            # -----------------------------------------

            # 変更後（EXP 主導）
            ab_full = torch.cat([zA_exp, zB_id], dim=1)  # A の EXP を先頭に
            ba_full = torch.cat([zB_exp, zA_id], dim=1)  # B の EXP を先頭に


            ab = self.decoder_B(ab_full, zA_exp_full)
            ba = self.decoder_A(ba_full, zB_exp_full)


        mask_a_pred = self.mask_decoder(zA_full, zA_exp_full)
        mask_b_pred = self.mask_decoder(zB_full, zB_exp_full)

        lm_a_in = torch.cat([zA_id, zA_exp], dim=1)
        lm_b_in = torch.cat([zB_id, zB_exp], dim=1)
        lm_a_pred = self.lm_head(lm_a_in)
        lm_b_pred = self.lm_head(lm_b_in)

        # ★ A→B の表情制約用ランドマーク
        lm_ab_in = torch.cat([zB_id, zA_exp], dim=1)
        lm_ab_pred = self.lm_head(lm_ab_in)

        return ModelOutput(
            aa=aa,
            bb=bb,
            ab=ab,
            ba=ba,
            mask_a_pred=mask_a_pred,
            mask_b_pred=mask_b_pred,
            lm_a_pred=lm_a_pred,
            lm_b_pred=lm_b_pred,
            lm_ab_pred=lm_ab_pred,  # ★追加
        )

    @torch.no_grad()
    def make_preview_grid(self, preview_dict):

        a_orig = to_image_tensor(preview_dict["a_orig"]).unsqueeze(0)
        aa     = to_image_tensor(preview_dict["aa"]).unsqueeze(0)
        ab     = to_image_tensor(preview_dict["ab"]).unsqueeze(0)

        b_orig = to_image_tensor(preview_dict["b_orig"]).unsqueeze(0)
        bb     = to_image_tensor(preview_dict["bb"]).unsqueeze(0)
        ba     = to_image_tensor(preview_dict["ba"]).unsqueeze(0)

        mask_a = prepare_mask(preview_dict["mask_a"].unsqueeze(0))
        mask_b = prepare_mask(preview_dict["mask_b"].unsqueeze(0))

        row_a = torch.cat([a_orig, aa, ab, mask_a], dim=3)
        row_b = torch.cat([b_orig, bb, ba, mask_b], dim=3)

        return torch.cat([row_a, row_b], dim=2)
