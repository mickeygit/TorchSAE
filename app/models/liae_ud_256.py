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
        x = x + z_exp * 0.5

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
        # EXP head（浅層）
        # -------------------------
        self.to_exp_shallow = nn.Sequential(
            nn.Conv2d(ch * 4, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(ae_dims),
            nn.Conv2d(ae_dims, ae_dims, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # -------------------------
        # UD（EXP 用に弱める）
        # -------------------------
        self.ud = UniformDistributionEXP()


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
        x3r = self.res3(x3)        # ← EXP の主成分（64×64）
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
        # EXP（浅層：x3r を直接使う）
        # -------------------------
        z_exp_64 = self.to_exp_shallow(x3r)   # [B,768,64,64]

        # EXP 正規化（弱め）
        z_exp_64 = self.ud(z_exp_64)

        return z_exp_64, z_id

class ExpAutoNorm(nn.Module):
    """
    EXP の振幅を自動調整して decoder の白飛びを防ぐ。
    入力 EXP の標準偏差を測り、ターゲット値に合わせてスケールする。
    """
    def __init__(self, target_std=0.5, eps=1e-6):
        super().__init__()
        self.target_std = target_std
        self.eps = eps

    def forward(self, z):
        # 現在の EXP の標準偏差
        mean = z.mean(dim=[1,2,3], keepdim=True)
        z = z - mean                      # ★ 追加：EXP のバイアスを除去

        std = z.std(dim=[1,2,3], keepdim=True) + self.eps
        scale = self.target_std / std
        scale = scale.clamp(0.1, 10.0)
        return z * scale

# ============================================================
# EXP 用に弱めた UD（EXP-only の振幅を潰さない）
# ============================================================
class UniformDistributionEXP(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + self.eps
        return (x - mean) / (std * 1.5 + self.eps)


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

        self.max_mode = False   # ← デフォルトは OFF

        # ============================
        # ID / EXP バランス調整（ID 復活版）
        # ============================

        # ★ ID のスケールを EXP と同等レベルに戻す（0.02 → 1.0）
        #    → B の顔の特徴（骨格・輪郭・質感）が復活する
        self.id_scale = 1.0

        # ★ EXP に飲まれないように ID の注入ゲインを弱める（0.2 → 0.1）
        #    → EXP が表情、ID が見た目を担当する正しい分離になる
        self.id_inject_gain = 0.1

        # ★ EXP はすでに強いので 6.0 → 2.0 のままでOK
        #    → EXP が ID を破壊しない適正値
        self.exp_gain = 2.0

        self.exp_auto_norm = ExpAutoNorm(target_std=0.3)


    def encode(self, x):
        z_exp_raw, z_id_raw = self.id_encoder(x)

        # EXP 64×64（可視化用）
        z_exp_64 = z_exp_raw

        # EXP 16×16（decoder 用）
        z_exp_16 = F.interpolate(z_exp_raw, size=(16, 16), mode="area")

        # ★ EXP の自動スケール安定化（白飛び防止）
        z_exp_16 = self.exp_auto_norm(z_exp_16)

        # ID のスケール
        z_id = z_id_raw * self.id_scale
        z_id = self.ud(z_id)

        # EXP の強度（exp_gain は 1.0〜2.0 の範囲で OK）
        z_exp_16 = z_exp_16 * self.exp_gain

        return z_exp_64, z_exp_16, z_id


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
        # -------------------------
        # encode
        # -------------------------
        zA_exp_64, zA_exp, zA_id = self.encode(img_a)  # zA_exp: [B,768,16,16]
        zB_exp_64, zB_exp, zB_id = self.encode(img_b)

        # ID に少し EXP を注入
        zA_id = zA_id + zA_exp * self.id_inject_gain
        zB_id = zB_id + zB_exp * self.id_inject_gain

        # ID+EXP を concat（1536ch）
        zA_full = torch.cat([zA_exp, zA_id], dim=1)  # [B,1536,16,16]
        zB_full = torch.cat([zB_exp, zB_id], dim=1)

        # decoder 用の EXP も 1536ch に揃える
        zA_exp_full = torch.cat([zA_exp, zA_exp], dim=1)  # [B,1536,16,16]
        zB_exp_full = torch.cat([zB_exp, zB_exp], dim=1)

        # -------------------------
        # 自己再構成
        # -------------------------
        aa = self.decoder_A(zA_full, zA_exp_full)
        bb = self.decoder_B(zB_full, zB_exp_full)

        # ============================
        # EXP-only 再構成（EXP 育成用）
        # ============================
        # A 側 EXP-only（64→16 に縮小して decoder に渡す）
        zero_id_A = torch.zeros_like(zA_id)

        # A 側 EXP-only
        zA_exp_64_down = F.interpolate(zA_exp_64, size=(16,16), mode="area")
        zA_exp_64_down = self.exp_auto_norm(zA_exp_64_down)   # ★ 追加
        zA_exp_only_full = torch.cat([zA_exp_64_down, zA_exp_64_down], dim=1)
        aa_exp_only = self.decoder_A(zA_exp_only_full, zA_exp_only_full)




        # B 側 EXP-only（同じ処理）
        zero_id_B = torch.zeros_like(zB_id)
        # B 側 EXP-only
        zB_exp_64_down = F.interpolate(zB_exp_64, size=(16,16), mode="area")
        zB_exp_64_down = self.exp_auto_norm(zB_exp_64_down)   # ★ 追加
        zB_exp_only_full = torch.cat([zB_exp_64_down, zB_exp_64_down], dim=1)
        bb_exp_only = self.decoder_B(zB_exp_only_full, zB_exp_only_full)

        # ============================
        # SWAP 部分（通常 or MAX モード）
        # ============================
        if self.max_mode:
            # MAX モード：ID を完全に殺して EXP だけで生成
            zero_id_A = torch.zeros_like(zA_id)
            zero_id_B = torch.zeros_like(zB_id)

            # A→B：B の ID を殺して A の EXP だけで生成
            ab_full = torch.cat([zA_exp, zero_id_B], dim=1)          # [B,1536,16,16]
            ab = self.decoder_B(ab_full, zA_exp_full)

            # B→A：A の ID を殺して B の EXP だけで生成
            ba_full = torch.cat([zB_exp, zero_id_A], dim=1)
            ba = self.decoder_A(ba_full, zB_exp_full)

        else:
            # 通常モード（EXP 主導）
            ab_full = torch.cat([zA_exp, zB_id], dim=1)  # [B,1536,16,16]
            ba_full = torch.cat([zB_exp, zA_id], dim=1)

            ab = self.decoder_B(ab_full, zA_exp_full)
            ba = self.decoder_A(ba_full, zB_exp_full)

        # ============================
        # mask / landmark
        # ============================
        mask_a_pred = self.mask_decoder(zA_full, zA_exp_full)
        mask_b_pred = self.mask_decoder(zB_full, zB_exp_full)

        lm_a_in = torch.cat([zA_id, zA_exp], dim=1)  # [B,1536,16,16]
        lm_b_in = torch.cat([zB_id, zB_exp], dim=1)
        lm_a_pred = self.lm_head(lm_a_in)
        lm_b_pred = self.lm_head(lm_b_in)

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
            lm_ab_pred=lm_ab_pred,
            aa_exp_only=aa_exp_only,
            bb_exp_only=bb_exp_only,
        )


    @torch.no_grad()
    def make_preview_grid(self, preview_dict):

        # A 側
        a_orig       = to_image_tensor(preview_dict["a_orig"]).unsqueeze(0)
        aa           = to_image_tensor(preview_dict["aa"]).unsqueeze(0)
        ab           = to_image_tensor(preview_dict["ab"]).unsqueeze(0)
        aa_exp_only  = to_image_tensor(preview_dict["aa_exp_only"]).unsqueeze(0)

        # B 側
        b_orig       = to_image_tensor(preview_dict["b_orig"]).unsqueeze(0)
        bb           = to_image_tensor(preview_dict["bb"]).unsqueeze(0)
        ba           = to_image_tensor(preview_dict["ba"]).unsqueeze(0)
        bb_exp_only  = to_image_tensor(preview_dict["bb_exp_only"]).unsqueeze(0)

        # mask
        mask_a = prepare_mask(preview_dict["mask_a"].unsqueeze(0))
        mask_b = prepare_mask(preview_dict["mask_b"].unsqueeze(0))

        # A 行：A_orig / AA / AB / AA_EXP_ONLY / mask
        row_a = torch.cat([a_orig, aa, ab, aa_exp_only, mask_a], dim=3)

        # B 行：B_orig / BB / BA / BB_EXP_ONLY / mask
        row_b = torch.cat([b_orig, bb, ba, bb_exp_only, mask_b], dim=3)

        # 2 行を縦に結合
        return torch.cat([row_a, row_b], dim=2)
