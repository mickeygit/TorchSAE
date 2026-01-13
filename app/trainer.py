from app.utils.preview import save_preview_grid
from app.utils.checkpoint import save_checkpoint

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from app.data.dataset import FaceDataset
from app.models.autoencoder_df import DFModel


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # ---------------------------------------------------------
        # Device / AMP
        # ---------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

        # ---------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------
        self.dataset_a = FaceDataset(cfg.data_dir_a, cfg)
        self.dataset_b = FaceDataset(cfg.data_dir_b, cfg)

        self.loader_a = DataLoader(self.dataset_a, batch_size=cfg.batch_size, shuffle=True)
        self.loader_b = DataLoader(self.dataset_b, batch_size=cfg.batch_size, shuffle=True)

        self.iter_a = iter(self.loader_a)
        self.iter_b = iter(self.loader_b)

        # ---------------------------------------------------------
        # Model (DF only for now)
        # ---------------------------------------------------------
        if cfg.is_df():
            self.model = DFModel(cfg).to(self.device)
        else:
            raise NotImplementedError("Only DF is supported at this stage")

        # ---------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------
        if cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        # Loss
        self.l1 = nn.L1Loss()

        # Step counter
        self.global_step = 0

    # ---------------------------------------------------------
    # Utility: get next batch (looping)
    # ---------------------------------------------------------
    def _next_batch(self, iterator, loader):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    def run(self):
        print("=== TorchSAE Training Start ===")

        while self.global_step < self.cfg.max_steps:
            # -----------------------------
            # Load batch A / B
            # -----------------------------
            batch_a, self.iter_a = self._next_batch(self.iter_a, self.loader_a)
            batch_b, self.iter_b = self._next_batch(self.iter_b, self.loader_b)

            batch_a = batch_a.to(self.device)
            batch_b = batch_b.to(self.device)

            # -----------------------------
            # Forward (AMP)
            # -----------------------------
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                # A→A, B→B, A→B, B→A
                out_aa, out_bb, out_ab, out_ba = self.model(batch_a, batch_b)

                # Loss (本家の最小構成：L1 のみ)
                loss_aa = self.l1(out_aa, batch_a)
                loss_bb = self.l1(out_bb, batch_b)
                loss_ab = self.l1(out_ab, batch_b)
                loss_ba = self.l1(out_ba, batch_a)

                loss = loss_aa + loss_bb + loss_ab + loss_ba

            # -----------------------------
            # Backward
            # -----------------------------
            self.opt.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.cfg.clip_grad > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)

            self.scaler.step(self.opt)
            self.scaler.update()

            self.global_step += 1

            # -----------------------------
            # Logging
            # -----------------------------
            if self.global_step % 100 == 0:
                print(f"[{self.global_step}] loss={loss.item():.4f}")

            # -----------------------------
            # Preview
            # -----------------------------
            if self.global_step % self.cfg.preview_interval == 0:
                self._save_preview(out_aa, out_bb, out_ab, out_ba)

            # -----------------------------
            # Save
            # -----------------------------
            if self.global_step % self.cfg.save_interval == 0:
                self._save_checkpoint()

        print("=== Training Finished ===")

    # ---------------------------------------------------------
    # Save preview (簡易版)
    # ---------------------------------------------------------
    def _save_preview(self, aa, bb, ab, ba):
        save_preview_grid(
            step=self.global_step,
            aa=aa,
            bb=bb,
            ab=ab,
            ba=ba,
            out_dir="preview"
        )

    # ---------------------------------------------------------
    # Save checkpoint
    # ---------------------------------------------------------
    def _save_checkpoint(self):
        save_checkpoint(
            model=self.model,
            step=self.global_step,
            out_dir="checkpoints"
        )
