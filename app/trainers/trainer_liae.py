# app/trainers/trainer_liae.py

import torch
import torch.nn as nn

from app.trainers.base_trainer import BaseTrainer
from app.models.autoencoder_liae import LIAEModel

# 追加：軽量 SAEHD loss
from app.losses.loss_saehd_light import SAEHDLightLoss

from utils.preview import make_saehd_style_preview

class TrainerLIAE(BaseTrainer):
    """
    Trainer for LIAE (DFL-style) autoencoder.
    Handles:
        - A→A
        - B→B
        - A→B
        - B→A
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------
        self.model = LIAEModel(cfg).to(self.device)

        # ---------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------
        if cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        # ---------------------------------------------------------
        # Loss (DSSIM + L1 + eyes-mouth-prio)
        # ---------------------------------------------------------
        self.loss_fn = SAEHDLightLoss(resolution=cfg.model_size)

    # ---------------------------------------------------------
    # One training step (landmarks 対応)
    # ---------------------------------------------------------
    def train_step(self, batch_a, batch_b):
        img_a, lm_a = batch_a
        img_b, lm_b = batch_b

        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        lm_a = lm_a.to(self.device)
        lm_b = lm_b.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            # LIAEModel は mask も返す
            aa, bb, ab, ba, mask_a, mask_b = self.model(img_a, img_b, lm_a, lm_b)

            loss_aa = self.loss_fn(aa, img_a, lm_a)
            loss_bb = self.loss_fn(bb, img_b, lm_b)
            loss_ab = self.loss_fn(ab, img_b, lm_b)
            loss_ba = self.loss_fn(ba, img_a, lm_a)

            loss = loss_aa + loss_bb + loss_ab + loss_ba

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()

        if self.cfg.clip_grad > 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)

        self.scaler.step(self.opt)
        self.scaler.update()

        # mask も返す
        return loss.item(), (aa, bb, ab, ba, mask_a, mask_b)

    # ---------------------------------------------------------
    # Preview hook
    # ---------------------------------------------------------
    def make_preview(self, outputs, batch_a, batch_b):
        aa, bb, ab, ba, mask_a, mask_b = outputs
        img_a, _ = batch_a
        img_b, _ = batch_b

        return {
            "aa": aa.detach().cpu(),
            "bb": bb.detach().cpu(),
            "ab": ab.detach().cpu(),
            "ba": ba.detach().cpu(),
            "mask_a": mask_a.detach().cpu(),
            "mask_b": mask_b.detach().cpu(),
            "a_orig": img_a.cpu(),
            "b_orig": img_b.cpu(),
        }

    @torch.no_grad()
    def get_preview_batch(self, batch_a, batch_b):
        """
        batch_a, batch_b:
          (img, landmarks, mask_gt, ...)
        """
        img_a = batch_a["img"]      # (B,3,H,W), 0-1
        img_b = batch_b["img"]

        # モデルの forward（あなたの実装に合わせて）
        out = self.model(img_a, img_b)
        # 例: out = (aa, bb, ab, mask_a_pred, mask_b_pred)
        aa, bb, ab, mask_a_pred, mask_b_pred = out

        preview = make_saehd_style_preview(
            img_a, img_b,
            aa, bb,
            ab,
            mask_b_pred,
        )

        # 1枚だけ取り出して numpy に
        p = preview[0].detach().cpu().permute(1,2,0).numpy()
        p = (p * 255).clip(0,255).astype("uint8")
        return p
