# app/trainers/trainer_liae.py

import torch
import torch.nn as nn

from app.trainers.base_trainer import BaseTrainer
from app.models.autoencoder_liae import LIAEModel


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

        # Loss
        self.l1 = nn.L1Loss()

    # ---------------------------------------------------------
    # One training step
    # ---------------------------------------------------------
    def train_step(self, batch_a, batch_b):
        """
        batch_a, batch_b: (N, C, H, W)
        """

        batch_a = batch_a.to(self.device)
        batch_b = batch_b.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            aa, bb, ab, ba = self.model(batch_a, batch_b)

            loss_aa = self.l1(aa, batch_a)
            loss_bb = self.l1(bb, batch_b)
            loss_ab = self.l1(ab, batch_b)
            loss_ba = self.l1(ba, batch_a)

            loss = loss_aa + loss_bb + loss_ab + loss_ba

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.cfg.clip_grad > 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)

        self.scaler.step(self.opt)
        self.scaler.update()

        return loss.item(), (aa, bb, ab, ba)

    # ---------------------------------------------------------
    # Preview hook
    # ---------------------------------------------------------
    def make_preview(self, outputs, batch_a, batch_b):
        """
        outputs: (aa, bb, ab, ba)
        """
        aa, bb, ab, ba = outputs

        return {
            "aa": aa.detach().cpu(),
            "bb": bb.detach().cpu(),
            "ab": ab.detach().cpu(),
            "ba": ba.detach().cpu(),
            "a_orig": batch_a.cpu(),
            "b_orig": batch_b.cpu(),
        }
