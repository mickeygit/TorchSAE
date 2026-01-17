import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 基本ブロック（既存）
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpscaleBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch * 4, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return F.pixel_shuffle(x, 2)


# ============================================================
# Encoder（RGB + 1ch landmarks heatmap）
# ============================================================

class LIAEEncoder(nn.Module):
    def __init__(self, in_ch=4, base_ch=128):
        super().__init__()
        ch = base_ch

        self.down1 = ConvBlock(in_ch, ch)
        self.down2 = ConvBlock(ch, ch * 2)
        self.down3 = ConvBlock(ch * 2, ch * 4)
        self.down4 = ConvBlock(ch * 4, ch * 8)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return self.flatten(x)


# ============================================================
# InterFlow（既存）
# ============================================================

class LIAEInter(nn.Module):
    def __init__(self, latent_dim, e_dims, res=8, ae_dim=512):
        super().__init__()

        self.res = res
        self.e_dims = e_dims
        self.enc_out_ch = e_dims * 8

        self.fc1 = nn.Linear(latent_dim, ae_dim)
        self.fc2 = nn.Linear(ae_dim, res * res * self.enc_out_ch)

        self.up = UpscaleBlock(self.enc_out_ch)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        x = self.fc2(x)
        x = x.view(-1, self.enc_out_ch, self.res, self.res)
        x = self.up(x)
        return x


# ============================================================
# Decoder（既存）
# ============================================================

class LIAEDecoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=128):
        super().__init__()
        ch = base_ch
        in_ch = ch * 8

        self.up1 = UpscaleBlock(in_ch)
        self.to1 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

        self.up2 = UpscaleBlock(in_ch)
        self.to2 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

        self.up3 = UpscaleBlock(in_ch)
        self.to3 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

    def forward(self, x):
        x1 = self.up1(x)
        o1 = torch.tanh(self.to1(x1))

        x2 = self.up2(x1)
        o2 = torch.tanh(self.to2(x2))

        x3 = self.up3(x2)
        o3 = torch.tanh(self.to3(x3))

        return o3


class LIAEMaskDecoder(nn.Module):
    def __init__(self, base_ch=128):
        super().__init__()
        self.dec = LIAEDecoder(out_ch=1, base_ch=base_ch)

    def forward(self, x):
        return self.dec(x)


# ============================================================
# LIAE Model（本家方式：RGB + 1ch landmarks heatmap）
# ============================================================

class LIAEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        res = cfg.model_size
        e_dims = cfg.e_dims
        ae_dims = cfg.ae_dims
        d_dims = cfg.d_dims
        d_mask_dims = cfg.d_mask_dims

        # Encoder（RGB 3ch + heatmap 1ch = 4ch）
        self.encoder = LIAEEncoder(in_ch=4, base_ch=e_dims)

        enc_spatial = res // 16
        enc_out_ch = e_dims * 8
        latent_dim = enc_spatial * enc_spatial * enc_out_ch

        self.inter_B = LIAEInter(latent_dim, e_dims, enc_spatial, ae_dims)
        self.inter_AB = LIAEInter(latent_dim, e_dims, enc_spatial, ae_dims)

        self.decoder = LIAEDecoder(out_ch=3, base_ch=d_dims)
        self.mask_decoder = LIAEMaskDecoder(base_ch=d_mask_dims)

    # ---------------------------------------------------------
    # landmarks → 1ch heatmap（本家と同じ思想）
    # ---------------------------------------------------------
    def _landmarks_to_heatmap(self, lm, H, W, sigma=2.0):
        """
        lm: (N, 68, 2)  pixel coords
        return: (N, 1, H, W)
        """
        device = lm.device
        N, K, _ = lm.shape

        ys = torch.arange(0, H, device=device).view(1, H, 1)
        xs = torch.arange(0, W, device=device).view(1, 1, W)

        lm_x = lm[:, :, 0].view(N, K, 1, 1)
        lm_y = lm[:, :, 1].view(N, K, 1, 1)

        dist2 = (xs - lm_x) ** 2 + (ys - lm_y) ** 2
        heatmaps = torch.exp(-dist2 / (2 * sigma * sigma))

        heatmap = heatmaps.max(dim=1, keepdim=True).values
        return heatmap

    # ---------------------------------------------------------
    def forward(self, a, b, lm_a, lm_b):
        N, C, H, W = a.shape

        # landmarks → heatmap
        hm_a = self._landmarks_to_heatmap(lm_a, H, W)
        hm_b = self._landmarks_to_heatmap(lm_b, H, W)

        # RGB + heatmap
        a_in = torch.cat([a, hm_a], dim=1)
        b_in = torch.cat([b, hm_b], dim=1)

        # Encode
        z_a = self.encoder(a_in)
        z_b = self.encoder(b_in)

        # InterFlow
        AB_a = self.inter_AB(z_a)
        AB_b = self.inter_AB(z_b)
        B_a = self.inter_B(z_a)
        B_b = self.inter_B(z_b)

        # Decode（本家 LIAE の identity/expression 分離）
        aa = self.decoder(AB_a)
        bb = self.decoder(AB_b)
        ab = self.decoder(B_b)
        ba = self.decoder(B_a)

        return aa, bb, ab, ba
