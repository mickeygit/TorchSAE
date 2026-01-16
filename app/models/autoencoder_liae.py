import torch
import torch.nn as nn

from app.models.encoder_df import DFEncoder
from app.models.decoder_df import DFDecoder


class LIAEModel(nn.Module):
    """
    LIAE (DFL-style) autoencoder model.
    共通エンコーダ + A/B 専用 latent-branch + A/B デコーダ。
    A→A, B→B, A→B, B→A の4出力を返す。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ---------------------------------------------------------
        # Shared Encoder (DF と同じ)
        # ---------------------------------------------------------
        self.encoder_shared = DFEncoder(
            model_size=cfg.model_size,
            e_dims=cfg.e_dims,
            ae_dims=cfg.ae_dims,
        )

        # ---------------------------------------------------------
        # A / B 専用 latent-branch（1×1 conv）
        # DFL LIAE の "inter" に相当
        # ---------------------------------------------------------
        act = nn.ReLU6  # TensorRT 最適化に有利

        self.inter_A = nn.Sequential(
            nn.Conv2d(cfg.ae_dims, cfg.inter_dims, kernel_size=1),
            act(inplace=True),
            nn.BatchNorm2d(cfg.inter_dims),
            nn.Conv2d(cfg.inter_dims, cfg.ae_dims, kernel_size=1),
        )

        self.inter_B = nn.Sequential(
            nn.Conv2d(cfg.ae_dims, cfg.inter_dims, kernel_size=1),
            act(inplace=True),
            nn.BatchNorm2d(cfg.inter_dims),
            nn.Conv2d(cfg.inter_dims, cfg.ae_dims, kernel_size=1),
        )

        # ---------------------------------------------------------
        # Two Decoders (A / B)
        # ---------------------------------------------------------
        self.decoder_A = DFDecoder(
            d_dims=cfg.d_dims,
            d_mask_dims=cfg.d_mask_dims,
            ae_dims=cfg.ae_dims,
        )

        self.decoder_B = DFDecoder(
            d_dims=cfg.d_dims,
            d_mask_dims=cfg.d_mask_dims,
            ae_dims=cfg.ae_dims,
        )

        # ---------------------------------------------------------
        # Initialize inter layers (Xavier)
        # ---------------------------------------------------------
        self._init_weights()

    # ---------------------------------------------------------
    def _init_weights(self):
        for m in list(self.inter_A.modules()) + list(self.inter_B.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------
    # Forward: return 4 reconstructions
    # ---------------------------------------------------------
    def forward(self, batch_a, batch_b):
        """
        batch_a: (N, C, H, W)
        batch_b: (N, C, H, W)

        Returns:
            out_aa: A→A
            out_bb: B→B
            out_ab: A→B
            out_ba: B→A
        """

        # -----------------------------
        # Shared encode
        # -----------------------------
        z_a_shared = self.encoder_shared(batch_a)
        z_b_shared = self.encoder_shared(batch_b)

        # -----------------------------
        # A/B 専用 latent-branch
        # -----------------------------
        z_a = self.inter_A(z_a_shared)
        z_b = self.inter_B(z_b_shared)

        # -----------------------------
        # Decode
        # -----------------------------
        out_aa = self.decoder_A(z_a)
        out_bb = self.decoder_B(z_b)

        # Cross decode
        out_ab = self.decoder_B(z_a)
        out_ba = self.decoder_A(z_b)

        return out_aa, out_bb, out_ab, out_ba
