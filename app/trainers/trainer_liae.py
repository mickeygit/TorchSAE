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

            # total
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

        # ★ 内訳を返す
        loss_dict = {
            "total": loss.item(),
            "recon": recon_loss.item(),
            "mask": mask_loss.item(),
            "landmark": lm_loss.item(),
        }

        return loss_dict, {
            "aa": aa, "bb": bb, "ab": ab, "ba": ba,
            "mask_a_pred": mask_a_pred, "mask_b_pred": mask_b_pred,
            "lm_a_pred": lm_a_pred, "lm_b_pred": lm_b_pred,
        }
    def make_preview(self, outputs, batch_a, batch_b):
        import torch

        # --- 値域補正（完全版） ---
        def to_float01(x):
            x = x.float()
            if x.min() < 0:          # -1〜1 の場合
                x = (x + 1) / 2
            if x.max() > 1.5:        # 0〜255 の float の場合
                x = x / 255.0
            return x.clamp(0.0, 1.0)

        # --- landmark 描画 ---
        def draw_landmarks_tensor(img, landmarks, color=(1.0, 1.0, 1.0)):
            out = img.clone()
            c = torch.tensor(color).view(3,1,1)
            for (x, y) in landmarks:
                x, y = int(x), int(y)
                if 1 <= x < out.shape[2]-1 and 1 <= y < out.shape[1]-1:
                    out[:, y-1:y+2, x-1:x+2] = c
            return out

        # --- 元画像と landmark ---
        img_a, lm_a, _ = batch_a
        img_b, lm_b, _ = batch_b

        lm_a = lm_a[0].cpu().numpy()
        lm_b = lm_b[0].cpu().numpy()

        # --- モデル出力（aa / ab / bb / ba） ---
        aa = to_float01(outputs["aa"][0].detach().cpu())
        bb = to_float01(outputs["bb"][0].detach().cpu())
        ab = to_float01(outputs["ab"][0].detach().cpu())
        ba = to_float01(outputs["ba"][0].detach().cpu())

        # --- 元画像に landmark を描画 ---
        a_orig_lm = to_float01(draw_landmarks_tensor(img_a[0].cpu(), lm_a))
        b_orig_lm = to_float01(draw_landmarks_tensor(img_b[0].cpu(), lm_b))

        # --- mask（sigmoid → 値域補正） ---
        mask_a = to_float01(torch.sigmoid(outputs["mask_a_pred"][0]).detach().cpu())
        mask_b = to_float01(torch.sigmoid(outputs["mask_b_pred"][0]).detach().cpu())

        if mask_a.ndim == 2:
            mask_a = mask_a.unsqueeze(0)
        if mask_b.ndim == 2:
            mask_b = mask_b.unsqueeze(0)

        return {
            "aa": aa,
            "bb": bb,
            "ab": ab,
            "ba": ba,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "a_orig": a_orig_lm,
            "b_orig": b_orig_lm,
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
