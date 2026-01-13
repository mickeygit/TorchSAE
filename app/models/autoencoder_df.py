import torch
import torch.nn as nn

from app.models.encoder_df import DFEncoder
from app.models.decoder_df import DFDecoder


class DFModel(nn.Module):
    """
    DF (DeepFaceLab standard) autoencoder model.
    本家 SAEHD の A→A, B→B, A→B, B→A の4出力を再現する最小構成。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ---------------------------------------------------------
        # Shared Encoder
        # ---------------------------------------------------------
        self.encoder = DFEncoder(
            model_size=cfg.model_size,
            e_dims=cfg.e_dims,
            ae_dims=cfg.ae_dims,
        )

        # ---------------------------------------------------------
        # Two Decoders (A / B)
        # ---------------------------------------------------------
        self.decoder_a = DFDecoder(
            model_size=cfg.model_size,
            d_dims=cfg.d_dims,
            d_mask_dims=cfg.d_mask_dims,
            ae_dims=cfg.ae_dims,
        )

        self.decoder_b = DFDecoder(
            model_size=cfg.model_size,
            d_dims=cfg.d_dims,
            d_mask_dims=cfg.d_mask_dims,
            ae_dims=cfg.ae_dims,
        )

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
        # Encode
        # -----------------------------
        z_a = self.encoder(batch_a)
        z_b = self.encoder(batch_b)

        # -----------------------------
        # Decode
        # -----------------------------
        out_aa = self.decoder_a(z_a)
        out_bb = self.decoder_b(z_b)

        out_ab = self.decoder_b(z_a)
        out_ba = self.decoder_a(z_b)

        return out_aa, out_bb, out_ab, out_ba
