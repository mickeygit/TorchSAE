import os
import json

import torch
import torch.nn as nn

from app.trainers.base_trainer import BaseTrainer
from app.models import LIAEModel, LIAE_UD_256
from app.losses.loss_saehd_light import SAEHDLightLoss

from app.utils.model_output import ModelOutput
from app.utils.loss_utils import compute_total_loss
from app.utils.preview_utils import build_preview_dict


class TrainerLIAE(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

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

        if cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        self.loss_fn = SAEHDLightLoss(resolution=cfg.model_size)
        self.bce = nn.BCEWithLogitsLoss()

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

    def train_step(self, batch_a, batch_b):
        img_a, lm_a, mask_a_gt = batch_a
        img_b, lm_b, mask_b_gt = batch_b

        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        lm_a = lm_a.to(self.device)
        lm_b = lm_b.to(self.device)
        mask_a_gt = mask_a_gt.to(self.device)
        mask_b_gt = mask_b_gt.to(self.device)

        if self.auto_mode and self.auto_params is not None:
            lr = self._get_auto_value("learning_rate") or self.cfg.lr
            mask_w = self._get_auto_value("mask_weight") or self.cfg.mask_loss_weight
            landmark_w = self._get_auto_value("landmark_weight") or self.cfg.landmark_loss_weight
            clip_grad = self._get_auto_value("clip_grad") or self.cfg.clip_grad

            # ★ expr_w も auto 対応（なければ cfg かデフォルト 2.0）
            expr_w = self._get_auto_value("expr_weight")
            if expr_w is None:
                expr_w = float(getattr(self.cfg, "expr_loss_weight", 2.0))

            warp_prob = self._get_auto_value("warp_prob")
            if warp_prob is None:
                warp_prob = float(self.cfg.random_warp)

            hsv_power = self._get_auto_value("hsv_power")
            if hsv_power is None:
                hsv_power = float(self.cfg.random_hsv_power)

            noise_power = self._get_auto_value("noise_power")
            if noise_power is None:
                noise_power = float(self.cfg.random_noise_power)

            shell_power = self._get_auto_value("shell_power")
            if shell_power is None:
                shell_power = 0.0
        else:
            lr = self.cfg.lr
            mask_w = self.cfg.mask_loss_weight * (
                0.5 + 0.5 * torch.exp(-torch.tensor(self.global_step / 5000.0))
            )
            landmark_w = self.cfg.landmark_loss_weight
            clip_grad = self.cfg.clip_grad

            # ★ 非 auto 時は cfg.expr_loss_weight があれば使い、なければ 2.0
            expr_w = float(getattr(self.cfg, "expr_loss_weight", 2.0))

            warp_prob = float(self.cfg.random_warp)
            hsv_power = float(self.cfg.random_hsv_power)
            noise_power = float(self.cfg.random_noise_power)
            shell_power = 0.0

        for g in self.opt.param_groups:
            g["lr"] = lr

        # ★ LIAE_UD_256 側の debug_latents / debug_decoder 用に step を渡す
        if hasattr(self.model, "global_step"):
            self.model.global_step = self.global_step

        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            outputs: ModelOutput = self.model(
                img_a, img_b, lm_a, lm_b,
                warp_prob=warp_prob,
                hsv_power=hsv_power,
                noise_power=noise_power,
                shell_power=shell_power,
            )

            loss_dict_raw = compute_total_loss(
                outputs,
                img_a, img_b,
                lm_a, lm_b,
                mask_a_gt, mask_b_gt,
                self.loss_fn,
                self.bce,
                float(mask_w),
                float(landmark_w),
                float(expr_w),   # ★ 追加
            )

            loss = loss_dict_raw["total"]

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()

        if clip_grad is not None and clip_grad > 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

        self.scaler.step(self.opt)
        self.scaler.update()

        loss_dict = {
            "total": loss.item(),
            "recon": loss_dict_raw["recon"].item(),
            "mask": loss_dict_raw["mask"].item(),
            "landmark": loss_dict_raw["landmark"].item(),
            "expr": loss_dict_raw["expr"].item(),          # ★ 追加
            "lr": float(lr),
            "mask_w": float(mask_w),
            "landmark_w": float(landmark_w),
            "expr_w": float(expr_w),                       # ★ 追加
            "clip_grad": float(clip_grad),
            "warp_prob": float(warp_prob),
            "hsv_power": float(hsv_power),
            "noise_power": float(noise_power),
            "shell_power": float(shell_power),
        }

        return loss_dict, outputs

    def make_preview(self, outputs: ModelOutput, batch_a, batch_b):
        return build_preview_dict(outputs, batch_a, batch_b)
