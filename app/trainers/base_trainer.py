# app/trainers/base_trainer.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import FaceDataset
from app.utils.preview import save_liae_preview_with_masks


def print_startup_banner(cfg):
    print("============================================================")
    print("==              TorchSAE Trainer (PyTorch)                ==")
    print("============================================================")
    print(f"==  Model: {cfg.model_type}")
    print(f"==  Resolution: {cfg.model_size}")
    print(f"==  Encoder dims: {cfg.e_dims}")
    print(f"==  AE dims: {cfg.ae_dims}")
    print(f"==  Decoder dims: {cfg.d_dims}")
    print(f"==  Mask dims: {cfg.d_mask_dims}")
    print(f"==  Learn Mask: {cfg.learn_mask}")
    print("============================================================")
    print("==  Starting training...")
    print("==  Press Ctrl+C to save and exit safely.")
    print("============================================================")


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

        # Dataset
        self.dataset_a = FaceDataset(cfg.data_dir_a, cfg)
        self.dataset_b = FaceDataset(cfg.data_dir_b, cfg)

        self.loader_a = DataLoader(
            self.dataset_a,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        self.loader_b = DataLoader(
            self.dataset_b,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        self.iter_a = iter(self.loader_a)
        self.iter_b = iter(self.loader_b)

        # モデルはサブクラスで定義される
        self.model = None
        self.opt = None

        self.global_step = 0

    # ---------------------------------------------------------
    def _next_batch(self, iterator, loader):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator

    # ---------------------------------------------------------
    def _load_resume(self):
        if self.cfg.resume_path is None:
            print("[Resume] No resume_path set.")
            return

        if not os.path.exists(self.cfg.resume_path):
            print(f"[Resume] File not found: {self.cfg.resume_path}")
            return

        print(f"[Resume] Loading checkpoint: {self.cfg.resume_path}")
        state = torch.load(self.cfg.resume_path, map_location=self.device)

        self.global_step = state.get("step", 0)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])

        print(f"[Resume] Resumed from step {self.global_step}")

    # ---------------------------------------------------------
    # Main training loop（landmarks 対応）
    # ---------------------------------------------------------
    def run(self):
        print_startup_banner(self.cfg)
        print("=== Training Start ===")

        self._load_resume()

        try:
            while self.global_step < self.cfg.max_steps:

                # (img, lm) を取得
                batch_a, self.iter_a = self._next_batch(self.iter_a, self.loader_a)
                batch_b, self.iter_b = self._next_batch(self.iter_b, self.loader_b)

                img_a, lm_a = batch_a
                img_b, lm_b = batch_b

                # preview 用に保存（画像のみ）
                self.last_batch_a = img_a.clone()
                self.last_batch_b = img_b.clone()

                # サブクラスの train_step を呼ぶ
                loss, outputs = self.train_step(batch_a, batch_b)

                self.global_step += 1

                if self.global_step % 100 == 0:
                    print(f"[{self.global_step}] loss={loss:.4f}")

                if self.global_step % self.cfg.preview_interval == 0:
                    aa, bb, ab, ba = outputs
                    self._save_preview(aa, bb, ab, ba)

                if self.global_step % self.cfg.save_interval == 0:
                    self._save_checkpoint()

        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user. Saving checkpoint...")
            self._save_checkpoint()
            print("[INFO] Checkpoint saved. Exiting safely.")

        print("=== Training Finished ===")

    # ---------------------------------------------------------
    def _save_preview(self, aa, bb, ab, ba):
        save_liae_preview_with_masks(
            step=self.global_step,
            aa=aa.detach().cpu(),
            bb=bb.detach().cpu(),
            ab=ab.detach().cpu(),
            ba=ba.detach().cpu(),
            a_orig=self.last_batch_a.cpu(),
            b_orig=self.last_batch_b.cpu(),
            out_dir="/workspace/logs/previews",
            ext="jpg",
        )

    # ---------------------------------------------------------
    def _save_checkpoint(self):
        state = {
            "step": self.global_step,
            "config": self.cfg.__dict__,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        out_path = os.path.join(
            "/workspace/models",
            f"resume_step_{self.global_step}.pth"
        )

        torch.save(state, out_path)
        print(f"[Checkpoint] Saved: {out_path}")
