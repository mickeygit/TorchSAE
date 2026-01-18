import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 基本ブロック
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
# InterFlow（latent → feature map）
# ============================================================

class LIAEInter(nn.Module):
    def __init__(self, latent_dim, e_dims, res=8, ae_dim=512):
        super().__init__()

        self.res = res
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
# Decoder（RGB / Mask 共通）
# ============================================================

class LIAEDecoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=128, use_tanh=True):
        super().__init__()
        ch = base_ch
        in_ch = ch * 8

        self.use_tanh = use_tanh

        self.up1 = UpscaleBlock(in_ch)
        self.to1 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

        self.up2 = UpscaleBlock(in_ch)
        self.to2 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

        self.up3 = UpscaleBlock(in_ch)
        self.to3 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

    def forward(self, x):
        x1 = self.up1(x)
        o1 = self.to1(x1)

        x2 = self.up2(x1)
        o2 = self.to2(x2)

        x3 = self.up3(x2)
        o3 = self.to3(x3)

        if self.use_tanh:
            o3 = torch.tanh(o3)

        return o3


# ============================================================
# Mask Decoder（logits 出力）
# ============================================================

class LIAEMaskDecoder(nn.Module):
    def __init__(self, base_ch=128):
        super().__init__()
        self.dec = LIAEDecoder(out_ch=1, base_ch=base_ch, use_tanh=False)

    def forward(self, x):
        return self.dec(x)


# ============================================================
# LIAE Model（LIAE + SAEHD 風）
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

        # Encoder
        self.encoder = LIAEEncoder(in_ch=4, base_ch=e_dims)

        # latent dims
        enc_spatial = res // 16
        enc_out_ch = e_dims * 8
        latent_dim = enc_spatial * enc_spatial * enc_out_ch
        self.enc_out_ch = enc_out_ch

        # InterFlow
        self.inter_B = LIAEInter(latent_dim, e_dims, enc_spatial, ae_dims)
        self.inter_AB = LIAEInter(latent_dim, e_dims, enc_spatial, ae_dims)

        # Decoders
        self.decoder = LIAEDecoder(out_ch=3, base_ch=d_dims, use_tanh=True)
        self.mask_decoder = LIAEMaskDecoder(base_ch=d_mask_dims)

        # Landmark head
        self.lm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.enc_out_ch, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 68 * 2),
        )

    # ---------------------------------------------------------
    # landmarks → 1ch heatmap
    # ---------------------------------------------------------
    def _landmarks_to_heatmap(self, lm, H, W, sigma=2.0):
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

        # heatmap
        hm_a = self._landmarks_to_heatmap(lm_a, H, W)
        hm_b = self._landmarks_to_heatmap(lm_b, H, W)

        # concat
        a_in = torch.cat([a, hm_a], dim=1)
        b_in = torch.cat([b, hm_b], dim=1)

        # encode
        z_a = self.encoder(a_in)
        z_b = self.encoder(b_in)

        # interflow
        AB_a = self.inter_AB(z_a)
        AB_b = self.inter_AB(z_b)
        B_a = self.inter_B(z_a)
        B_b = self.inter_B(z_b)

        # decode
        aa = self.decoder(AB_a)
        bb = self.decoder(AB_b)
        ab = self.decoder(B_b)
        ba = self.decoder(B_a)

        # mask logits
        mask_a = self.mask_decoder(AB_a)
        mask_b = self.mask_decoder(AB_b)

        # landmark prediction
        lm_a_pred = self.lm_head(AB_a).view(N, 68, 2)
        lm_b_pred = self.lm_head(AB_b).view(N, 68, 2)

        return aa, bb, ab, ba, mask_a, mask_b, lm_a_pred, lm_b_pred
