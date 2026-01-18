# app/trainers/trainer_liae.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.trainers.base_trainer import BaseTrainer
from app.models.autoencoder_liae import LIAEModel
from app.losses.loss_saehd_light import SAEHDLightLoss
from utils.preview import make_saehd_style_preview


class TrainerLIAE(BaseTrainer):
    """
    LIAE + SAEHD 風 Trainer
      - 再構成 loss
      - mask loss (XSeg)
      - landmark loss
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.model = LIAEModel(cfg).to(self.device)

        if cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        self.loss_fn = SAEHDLightLoss(resolution=cfg.model_size)
        self.bce = nn.BCEWithLogitsLoss()


    def train_step(self, batch_a, batch_b):
        img_a, lm_a, mask_a_gt = batch_a
        img_b, lm_b, mask_b_gt = batch_b

        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        lm_a = lm_a.to(self.device)
        lm_b = lm_b.to(self.device)
        mask_a_gt = mask_a_gt.to(self.device)
        mask_b_gt = mask_b_gt.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            (
                aa, bb, ab, ba,
                mask_a_pred, mask_b_pred,
                lm_a_pred, lm_b_pred
            ) = self.model(img_a, img_b, lm_a, lm_b)

            # recon
            loss_aa = self.loss_fn(aa, img_a, lm_a)
            loss_bb = self.loss_fn(bb, img_b, lm_b)
            loss_ab = self.loss_fn(ab, img_b, lm_b)
            loss_ba = self.loss_fn(ba, img_a, lm_a)
            recon_loss = loss_aa + loss_bb + loss_ab + loss_ba

            # mask
            mask_loss = (
                self.bce(mask_a_pred, mask_a_gt) +
                self.bce(mask_b_pred, mask_b_gt)
            )

            # landmarks
            lm_loss = (
                F.l1_loss(lm_a_pred, lm_a) +
                F.l1_loss(lm_b_pred, lm_b)
            )

            loss = (
                recon_loss +
                self.cfg.mask_loss_weight * mask_loss +
                self.cfg.landmark_loss_weight * lm_loss
            )

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()

        if self.cfg.clip_grad > 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)

        self.scaler.step(self.opt)
        self.scaler.update()

        return loss.item(), {
            "aa": aa, "bb": bb, "ab": ab, "ba": ba,
            "mask_a_pred": mask_a_pred, "mask_b_pred": mask_b_pred,
            "lm_a_pred": lm_a_pred, "lm_b_pred": lm_b_pred,
        }

    def make_preview(self, outputs, batch_a, batch_b):
        img_a, _, _ = batch_a
        img_b, _, _ = batch_b

        return {
            "aa": outputs["aa"].detach().cpu(),
            "bb": outputs["bb"].detach().cpu(),
            "ab": outputs["ab"].detach().cpu(),
            "ba": outputs["ba"].detach().cpu(),
            "mask_a": torch.sigmoid(outputs["mask_a_pred"]).detach().cpu(),
            "mask_b": torch.sigmoid(outputs["mask_b_pred"]).detach().cpu(),
            "a_orig": img_a.cpu(),
            "b_orig": img_b.cpu(),
        }

    @torch.no_grad()
    def get_preview_batch(self, batch_a, batch_b):
        img_a = batch_a["img"]
        img_b = batch_b["img"]

        aa, bb, ab, ba, mask_a_pred, mask_b_pred, _, _ = self.model(
            img_a, img_b, batch_a["landmarks"], batch_b["landmarks"]
        )

        preview = make_saehd_style_preview(
            img_a, img_b,
            aa, bb,
            ab,
            mask_b_pred,
        )

        p = preview[0].detach().cpu().permute(1, 2, 0).numpy()
        p = (p * 255).clip(0, 255).astype("uint8")
        return p
