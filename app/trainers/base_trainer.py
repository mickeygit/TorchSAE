# app/trainers/base_trainer.py

import os
import time
import torch
from torch.cuda.amp import GradScaler


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None          # 子クラスでセット
        self.opt = None            # 子クラスでセット
        self.scaler = GradScaler(enabled=cfg.amp)

        self.global_step = 0
        self.start_time = time.time()

        self.save_dir = cfg.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 子クラスで実装する必要があるメソッド
    # ---------------------------------------------------------
    def train_step(self, batch_a, batch_b):
        """
        1 step 分の学習を行う。
        戻り値:
          loss_dict, outputs
        """
        raise NotImplementedError

    def make_preview(self, outputs, batch_a, batch_b):
        """
        プレビュー用の dict を返す。
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # checkpoint 保存 / ロード
    # ---------------------------------------------------------
    def _save_checkpoint(self):
        state = {
            "step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
        }
        path = os.path.join(self.save_dir, f"step_{self.global_step}.pth")
        torch.save(state, path)
        print(f"[Save] Saved checkpoint: {path}")

    def _load_resume(self):
        if getattr(self.cfg, "resume_path", None) is None:
            return

        resume_path = self.cfg.resume_path
        if not os.path.exists(resume_path):
            print(f"[Resume] resume_path not found: {resume_path}")
            return

        print(f"[Resume] Loading checkpoint: {resume_path}")
        state = torch.load(resume_path, map_location=self.device)

        # ★ モデルは strict=False で部分ロード（lm_head 追加などに対応）
        self.model.load_state_dict(state["model"], strict=False)

        if "optimizer" in state:
            try:
                self.opt.load_state_dict(state["optimizer"])
            except Exception:
                print("[Warn] Optimizer state mismatch. Skipping optimizer state.")

        # step は引き継ぐ
        if "step" in state:
            self.global_step = state["step"]
            print(f"[Resume] Resumed from step {self.global_step}")

    # ---------------------------------------------------------
    # メインループ
    # ---------------------------------------------------------
    def run(self):
        from app.data.dataset import FaceDataset
        from torch.utils.data import DataLoader

        # Dataset / DataLoader 準備
        ds_a = FaceDataset(self.cfg.data_a, self.cfg)
        ds_b = FaceDataset(self.cfg.data_b, self.cfg)

        dl_a = DataLoader(
            ds_a,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        dl_b = DataLoader(
            ds_b,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
        )

        it_a = iter(dl_a)
        it_b = iter(dl_b)

        # resume
        self._load_resume()

        print("============================================================")
        print("==              TorchSAE Trainer (PyTorch)                ==")
        print("============================================================")
        print(f"==  Model: {self.cfg.model_name}")
        print(f"==  Resolution: {self.cfg.model_size}")
        print(f"==  Encoder dims: {self.cfg.e_dims}")
        print(f"==  AE dims: {self.cfg.ae_dims}")
        print(f"==  Decoder dims: {self.cfg.d_dims}")
        print(f"==  Mask dims: {self.cfg.d_mask_dims}")
        print(f"==  Learn Mask: {self.cfg.learn_mask}")
        print("============================================================")
        print("==  Starting training...")
        print("==  Press Ctrl+C to save and exit safely.")
        print("============================================================")
        print("=== Training Start ===")

        try:
            while True:
                try:
                    batch_a = next(it_a)
                except StopIteration:
                    it_a = iter(dl_a)
                    batch_a = next(it_a)

                try:
                    batch_b = next(it_b)
                except StopIteration:
                    it_b = iter(dl_b)
                    batch_b = next(it_b)

                # ★ ここを loss_dict 受け取りに変更
                loss_dict, outputs = self.train_step(batch_a, batch_b)
                self.global_step += 1

                if self.global_step % 50 == 0:
                    elapsed = time.time() - self.start_time
                    print(
                        f"[Step {self.global_step}] "
                        f"total={loss_dict['total']:.4f} "
                        f"recon={loss_dict['recon']:.4f} "
                        f"mask={loss_dict['mask']:.4f} "
                        f"lm={loss_dict['landmark']:.4f}  "
                        f"elapsed={elapsed/60:.1f} min"
                    )

                if self.global_step % self.cfg.save_interval == 0:
                    self._save_checkpoint()

                if self.global_step % self.cfg.preview_interval == 0:
                    try:
                        preview = self.make_preview(outputs, batch_a, batch_b)

                        import torchvision.utils as vutils
                        import os
                        import torch

                        # --- 画像取り出し ---
                        a_orig = preview["a_orig"][0]      # [3,128,128]
                        b_orig = preview["b_orig"][0]      # [3,128,128]

                        aa = preview["aa"][0]
                        bb = preview["bb"][0]
                        ab = preview["ab"][0]
                        ba = preview["ba"][0]

                        # ★ ここは make_preview 側で sigmoid 済み前提ならそのまま使う
                        mask_a = preview["mask_a"][0]      # [1,128,128]
                        mask_b = preview["mask_b"][0]      # [1,128,128]

                        # --- mask を RGB に変換 ---
                        mask_a_rgb = mask_a.repeat(3, 1, 1)
                        mask_b_rgb = mask_b.repeat(3, 1, 1)

                        # --- SAEHD 本家と同じ並び ---
                        # A_orig | AA | AB | mask_A
                        # B_orig | BB | BA | mask_B
                        grid = vutils.make_grid(
                            [
                                a_orig, aa, ab, mask_a_rgb,
                                b_orig, bb, ba, mask_b_rgb,
                            ],
                            nrow=4,
                            normalize=True,
                            value_range=(0, 1),
                        )

                        preview_path = os.path.join(self.save_dir, f"preview_{self.global_step}.jpg")
                        vutils.save_image(grid, preview_path)
                        print(f"[Preview] Saved: {preview_path}")

                    except Exception as e:
                        print(f"[Preview] Failed: {e}")

        except KeyboardInterrupt:
            print("\n[Exit] Caught Ctrl+C, saving final checkpoint...")
            self._save_checkpoint()
            print("[Exit] Done.")
