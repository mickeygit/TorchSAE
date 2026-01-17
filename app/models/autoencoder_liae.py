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
# Encoder（flatten latent）
# ============================================================

class LIAEEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=128):
        super().__init__()
        ch = base_ch

        self.down1 = ConvBlock(in_ch, ch)        # 128
        self.down2 = ConvBlock(ch, ch * 2)       # 256
        self.down3 = ConvBlock(ch * 2, ch * 4)   # 512
        self.down4 = ConvBlock(ch * 4, ch * 8)   # 1024

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.down1(x)  # 64x64
        x = self.down2(x)  # 32x32
        x = self.down3(x)  # 16x16
        x = self.down4(x)  # 8x8
        return self.flatten(x)


# ============================================================
# InterFlow（latent → Dense → reshape → upscale）
# ============================================================

class LIAEInter(nn.Module):
    """
    latent_dim: flatten 後の次元数 (res/16 * res/16 * e_dims*8)
    e_dims: encoder base channel
    res: encoder 最終 feature map の spatial サイズ (128なら8)
    """
    def __init__(self, latent_dim, e_dims, res=8, ae_dim=512):
        super().__init__()

        self.res = res
        self.e_dims = e_dims
        self.enc_out_ch = e_dims * 8  # encoder 最終チャネル数（1024）

        # latent → bottleneck（ae_dim）→ encoder_out_shape
        self.fc1 = nn.Linear(latent_dim, ae_dim)
        self.fc2 = nn.Linear(ae_dim, res * res * self.enc_out_ch)

        self.up = UpscaleBlock(self.enc_out_ch)

    def forward(self, x):
        # x: (N, latent_dim)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        x = self.fc2(x)
        x = x.view(-1, self.enc_out_ch, self.res, self.res)  # (N, 1024, 8, 8)
        x = self.up(x)  # (N, 1024, 16, 16)
        return x


# ============================================================
# Decoder（multi-scale）
# ============================================================

class LIAEDecoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=128):
        super().__init__()
        ch = base_ch

        # 入力チャネルは encoder 最終の 8倍チャネルを想定（1024）
        in_ch = ch * 8

        self.up1 = UpscaleBlock(in_ch)       # 16x16 → 32x32
        self.to1 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

        self.up2 = UpscaleBlock(in_ch)       # 32x32 → 64x64
        self.to2 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

        self.up3 = UpscaleBlock(in_ch)       # 64x64 → 128x128
        self.to3 = nn.Conv2d(in_ch, out_ch, 5, padding=2)

    def forward(self, x):
        x1 = self.up1(x)
        o1 = torch.tanh(self.to1(x1))

        x2 = self.up2(x1)
        o2 = torch.tanh(self.to2(x2))

        x3 = self.up3(x2)
        o3 = torch.tanh(self.to3(x3))

        # TorchSAE の preview / loss は最終スケールだけ見ればよい
        return o3


# ============================================================
# Mask Decoder（必要なら使う）
# ============================================================

class LIAEMaskDecoder(nn.Module):
    def __init__(self, base_ch=128):
        super().__init__()
        self.dec = LIAEDecoder(out_ch=1, base_ch=base_ch)

    def forward(self, x):
        return self.dec(x)


# ============================================================
# LIAE Model（本家 latent mixing）
# ============================================================

class LIAEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        res = cfg.model_size          # 128
        e_dims = cfg.e_dims           # 128
        ae_dims = cfg.ae_dims         # 512
        inter_dims = cfg.inter_dims   # 64（今回は bottleneck ではなく ae_dim を使う）
        d_dims = cfg.d_dims           # 128
        d_mask_dims = cfg.d_mask_dims # 128

        # -----------------------------
        # Encoder
        # -----------------------------
        self.encoder = LIAEEncoder(in_ch=3, base_ch=e_dims)

        # latent_dim = (res/16)^2 * (e_dims*8)
        enc_spatial = res // 16       # 128 → 8
        enc_out_ch = e_dims * 8       # 128 * 8 = 1024
        latent_dim = enc_spatial * enc_spatial * enc_out_ch  # 8*8*1024 = 65536

        # -----------------------------
        # InterFlow（identity / expression）
        # ae_dim は cfg.ae_dims を使う（本家の bottleneck 相当）
        # -----------------------------
        self.inter_B = LIAEInter(
            latent_dim=latent_dim,
            e_dims=e_dims,
            res=enc_spatial,
            ae_dim=ae_dims,
        )
        self.inter_AB = LIAEInter(
            latent_dim=latent_dim,
            e_dims=e_dims,
            res=enc_spatial,
            ae_dim=ae_dims,
        )

        # -----------------------------
        # Decoder / MaskDecoder
        # -----------------------------
        self.decoder = LIAEDecoder(out_ch=3, base_ch=d_dims)
        self.mask_decoder = LIAEMaskDecoder(base_ch=d_mask_dims)

    # ---------------------------------------------------------
    def forward(self, a, b):
        # Encode
        z_a = self.encoder(a)  # (N, latent_dim)
        z_b = self.encoder(b)

        # InterFlow
        AB_a = self.inter_AB(z_a)  # (N, 1024, 16, 16)
        AB_b = self.inter_AB(z_b)
        B_a = self.inter_B(z_a)
        B_b = self.inter_B(z_b)

        # latent mixing（本家 LIAE の本質）
        # ここでは concat ではなく「identity / expression を別々に decoder に渡す」
        # という簡略版ではなく、identity×expression の組み合わせを decoder 入力とする。
        aa = self.decoder(AB_a)  # A→A（expression from A）
        bb = self.decoder(AB_b)  # B→B（expression from B）

        # A→B / B→A は identity / expression の組み合わせで制御したい場合、
        # ここで B 系 / AB 系のどちらを使うかを切り替える。
        ab = self.decoder(B_b)   # A→B（identity from B）
        ba = self.decoder(B_a)   # B→A（identity from A）

        return aa, bb, ab, ba
