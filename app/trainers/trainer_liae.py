import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.trainers.base_trainer import BaseTrainer
from app.models import LIAEModel, LIAE_UD_256
from app.losses.loss_saehd_light import SAEHDLightLoss

# ★ debug_utils を追加
from app.utils.debug_utils import tensor_minmax, tensor_stats, check_nan_inf


class TrainerLIAE(BaseTrainer):
    """
    LIAE + SAEHD 風 Trainer
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ★ モデル切り替え
        if getattr(cfg, "model_type", "liae") == "liae_ud_256":
            self.model = LIAE_UD_256(
                e_dims=cfg.e_dims,
                ae_dims=cfg.ae_dims,
                d_dims=cfg.d_dims,
                d_mask_dims=cfg.d_mask_dims,
            ).to(self.device)
            print("[Model] Using LIAE_UD_256")
        else:
            self.model = LIAEModel(cfg).to(self.device)
            print("[Model] Using LIAEModel (standard LIAE)")

        # optimizer
        if cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        self.loss_fn = SAEHDLightLoss(resolution=cfg.model_size)
        self.bce = nn.BCEWithLogitsLoss()

        # AUTO モード
        self.auto_mode = getattr(cfg, "auto_mode", False)
        self.auto_params = None

        if self.auto_mode:
            auto_path = getattr(cfg, "auto_params_path", "auto_params.json")
            if not os.path.isabs(auto_path):
                auto_path = os.path.join(os.getcwd(), auto_path)
            if os.path.exists(auto_path):
                with open(auto_path, "r", encoding="utf-8") as f:
                    self.auto_params = json.load(f)
                print(f"[AUTO] Loaded auto params from: {auto_path}")
            else:
                print(f"[AUTO] auto_params file not found: {auto_path}. Disabling auto_mode.")
                self.auto_mode = False
                self.auto_params = None

    # ---------------------------------------------------------
    # AUTO モード用ヘルパ
    # ---------------------------------------------------------
    def _get_auto_value(self, name):
        if self.auto_params is None or name not in self.auto_params:
            return None

        spec = self.auto_params[name]
        mode = spec.get("mode", "linear_decay")
        step = self.global_step

        if mode == "linear_decay":
            start = float(spec["start"])
            end = float(spec["end"])
            decay_steps = float(spec["decay_steps"])
            if decay_steps <= 0:
                return end
            t = max(0.0, min(1.0, step / decay_steps))
            return start + (end - start) * t

        elif mode == "staged":
            schedule = spec.get("schedule", [])
            for s in schedule:
                until = s["until"]
                value = s["value"]
                if until == "end":
                    return value
                if step <= int(until):
                    return value
            if schedule:
                return schedule[-1]["value"]
            return None

        return None

    # ---------------------------------------------------------
    # 1 step 学習
    # ---------------------------------------------------------
    def train_step(self, batch_a, batch_b):
        img_a, lm_a, mask_a_gt = batch_a
        img_b, lm_b, mask_b_gt = batch_b

        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        lm_a = lm_a.to(self.device)
        lm_b = lm_b.to(self.device)
        mask_a_gt = mask_a_gt.to(self.device)
        mask_b_gt = mask_b_gt.to(self.device)

        # AUTO or 固定
        if self.auto_mode and self.auto_params is not None:
            lr = self._get_auto_value("learning_rate") or self.cfg.lr
            mask_w = self._get_auto_value("mask_weight") or self.cfg.mask_loss_weight
            landmark_w = self._get_auto_value("landmark_weight") or self.cfg.landmark_loss_weight
            clip_grad = self._get_auto_value("clip_grad") or self.cfg.clip_grad

            warp_prob = self._get_auto_value("warp_prob") or float(self.cfg.random_warp)
            hsv_power = self._get_auto_value("hsv_power") or float(self.cfg.random_hsv_power)
            noise_power = self._get_auto_value("noise_power") or float(self.cfg.random_noise_power)
            shell_power = self._get_auto_value("shell_power") or 0.0

        else:
            lr = self.cfg.lr
            mask_w = self.cfg.mask_loss_weight * (
                0.5 + 0.5 * torch.exp(-torch.tensor(self.global_step / 5000.0))
            )
            landmark_w = self.cfg.landmark_loss_weight
            clip_grad = self.cfg.clip_grad

            warp_prob = float(self.cfg.random_warp)
            hsv_power = float(self.cfg.random_hsv_power)
            noise_power = float(self.cfg.random_noise_power)
            shell_power = 0.0

        # lr 更新
        for g in self.opt.param_groups:
            g["lr"] = lr

        # forward
        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            (
                aa, bb, ab, ba,
                mask_a_pred, mask_b_pred,
                lm_a_pred, lm_b_pred
            ) = self.model(
                img_a, img_b, lm_a, lm_b,
                warp_prob=warp_prob,
                hsv_power=hsv_power,
                noise_power=noise_power,
                shell_power=shell_power,
            )

            # loss 計算
            loss_aa = self.loss_fn(aa, img_a, lm_a)
            loss_bb = self.loss_fn(bb, img_b, lm_b)
            loss_ab = self.loss_fn(ab, img_b, lm_b)
            loss_ba = self.loss_fn(ba, img_a, lm_a)
            recon_loss = loss_aa + loss_bb + loss_ab + loss_ba

            mask_loss = (
                self.bce(mask_a_pred, mask_a_gt) +
                self.bce(mask_b_pred, mask_b_gt)
            )

            lm_loss = (
                F.l1_loss(lm_a_pred, lm_a) +
                F.l1_loss(lm_b_pred, lm_b)
            )

            loss = recon_loss + mask_w * mask_loss + landmark_w * lm_loss

        # backward
        self.opt.zero_grad()
        self.scaler.scale(loss).backward()

        if clip_grad and clip_grad > 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

        self.scaler.step(self.opt)
        self.scaler.update()

        # # ★ debug_utils を使った統一デバッグ
        # tensor_minmax("aa", aa)
        # tensor_minmax("bb", bb)
        # tensor_minmax("mask_a_pred", mask_a_pred)
        # tensor_minmax("mask_b_pred", mask_b_pred)
        # check_nan_inf("aa", aa)
        # check_nan_inf("bb", bb)

        # ログ
        loss_dict = {
            "total": loss.item(),
            "recon": recon_loss.item(),
            "mask": mask_loss.item(),
            "landmark": lm_loss.item(),
            "lr": float(lr),
            "mask_w": float(mask_w),
            "landmark_w": float(landmark_w),
            "clip_grad": float(clip_grad),
            "warp_prob": float(warp_prob),
            "hsv_power": float(hsv_power),
            "noise_power": float(noise_power),
            "shell_power": float(shell_power),
        }

        return loss_dict, {
            "aa": aa, "bb": bb, "ab": ab, "ba": ba,
            "mask_a_pred": mask_a_pred, "mask_b_pred": mask_b_pred,
            "lm_a_pred": lm_a_pred, "lm_b_pred": lm_b_pred,
        }

    # ---------------------------------------------------------
    # preview_dict を返すだけ
    # ---------------------------------------------------------
    def make_preview(self, outputs, batch_a, batch_b):
        img_a, lm_a, _ = batch_a
        img_b, lm_b, _ = batch_b

        img_a_0 = img_a[0].detach().cpu()
        img_b_0 = img_b[0].detach().cpu()

        aa = outputs["aa"][0].detach().cpu()
        bb = outputs["bb"][0].detach().cpu()
        ab = outputs["ab"][0].detach().cpu()
        ba = outputs["ba"][0].detach().cpu()

        mask_a = torch.sigmoid(outputs["mask_a_pred"][0]).detach().cpu()
        mask_b = torch.sigmoid(outputs["mask_b_pred"][0]).detach().cpu()

        def to_01(x):
            return x.float().clamp(0.0, 1.0)

        return {
            "a_orig": to_01(img_a_0),
            "b_orig": to_01(img_b_0),
            "aa": to_01(aa),
            "bb": to_01(bb),
            "ab": to_01(ab),
            "ba": to_01(ba),
            "mask_a": to_01(mask_a),
            "mask_b": to_01(mask_b),
        }
