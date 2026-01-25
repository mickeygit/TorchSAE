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

        # 出力側のシャープさ強化（白かすみ軽減版）
        self.out_block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1, bias=False),  # ★ bias を切る
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
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out_block(x)


# ============================================================
# DF-style encoder（dual latent, residual + sharpen 強化）
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

        # dual latent
        self.to_exp = nn.Sequential(
            nn.Conv2d(ch * 8, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ae_dims, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.to_id  = nn.Conv2d(ch * 8, ae_dims, 3, padding=1)

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

        z_exp = self.to_exp(x)
        z_id  = self.to_id(x)
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
        in_ch = ae_dims * 2  # ★ z_id + z_exp を想定

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
# LIAE_UD_256（dual latent 軽量強化版）
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

        # ★ ae_dims*2 を前提にした LandmarkHead
        self.lm_head = LandmarkHead(ae_dims=ae_dims)

    def encode(self, x):
        z_exp, z_id = self.encoder(x)

        # ★ z_exp は post_bn を通す（表情は揺れてOK）
        z_exp = self.post_bn(z_exp)

        # ★ z_id は post_bn を通さない（identity の安定化）
        # z_id = self.post_bn(z_id)
        z_id = z_id

        return z_exp, z_id


    def forward(self, img_a, img_b, lm_a=None, lm_b=None,
                warp_prob=0.0, hsv_power=0.0, noise_power=0.0, shell_power=0.0):

        zA_exp, zA_id = self.encode(img_a)
        zB_exp, zB_id = self.encode(img_b)

        # # ★ identity dropout（学習中だけ）
        # if self.training:
        #     zB_id = zB_id + torch.randn_like(zB_id) * 0.11

        # A→A / B→B
        aa = self.decoder_A(zA_exp)
        bb = self.decoder_B(zB_id)

        # A→B / B→A（SWAP）
        ab = self.decoder_B(zA_exp)
        ba = self.decoder_A(zB_exp)

        # mask は expression latent から
        zA_exp_ud = self.ud(zA_exp)
        zB_exp_ud = self.ud(zB_exp)
        mask_a_pred = self.mask_decoder(zA_exp_ud)
        mask_b_pred = self.mask_decoder(zB_exp_ud)

        # ★ landmarks は z_id + z_exp を concat して使う
        lm_a_in = torch.cat([zA_id, zA_exp], dim=1)
        lm_b_in = torch.cat([zB_id, zB_exp], dim=1)
        lm_a_pred = self.lm_head(lm_a_in)
        lm_b_pred = self.lm_head(lm_b_in)

        return aa, bb, ab, ba, mask_a_pred, mask_b_pred, lm_a_pred, lm_b_pred

    @torch.no_grad()
    def reset_decoder_A_out_block(self):
        print("=== Reset decoder_A.out_block parameters ===")
        for m in self.decoder_A.out_block.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()


    @torch.no_grad()
    def reset_decoder_B_out_block(self):
        print("=== Reset decoder_B.out_block parameters ===")
        for m in self.decoder_B.out_block.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        
    @torch.no_grad()
    def reset_encoder_id_block(self):
        print("=== Reset encoder identity block (down2/res2/down3/res3/down4/res4/reduce/sharpen/to_id/to_exp) ===")
        # ここは既存のまま
        for m in self.encoder.down2.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.res2.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.down3.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.res3.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.down4.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.res4.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.reduce.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.sharpen.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.to_id.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        for m in self.encoder.to_exp.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    @torch.no_grad()
    def reset_encoder_full(self):
        print("=== Reset FULL encoder (all Conv2d in DFEncoder) ===")
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    @torch.no_grad()
    def make_preview_grid(self, preview_dict):
        import torchvision.utils as vutils
        import torch

        # --- 値域補正 ---
        def to_float01(x):
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            else:
                x = x.float()
                if x.max() > 1.5:
                    x = x / 255.0
            return x.clamp(0.0, 1.0)

        a_orig = to_float01(preview_dict["a_orig"])
        b_orig = to_float01(preview_dict["b_orig"])
        aa     = to_float01(preview_dict["aa"])
        bb     = to_float01(preview_dict["bb"])
        ab     = to_float01(preview_dict["ab"])
        ba     = to_float01(preview_dict["ba"])

        mask_a = to_float01(preview_dict["mask_a"])
        mask_b = to_float01(preview_dict["mask_b"])

        if mask_a.ndim == 2:
            mask_a = mask_a.unsqueeze(0)
        if mask_b.ndim == 2:
            mask_b = mask_b.unsqueeze(0)

        mask_a_rgb = mask_a.repeat(3, 1, 1)
        mask_b_rgb = mask_b.repeat(3, 1, 1)

        # normalize=False が絶対に正しい
        grid = vutils.make_grid(
            [
                a_orig, aa, ab, mask_a_rgb,
                b_orig, bb, ba, mask_b_rgb,
            ],
            nrow=4,
            normalize=False,
        )

        return grid
