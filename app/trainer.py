from app.utils.preview import save_preview_grid
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import FaceDataset
from app.models.autoencoder_df import DFModel


class Trainer:
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

        # Model
        if cfg.is_df():
            self.model = DFModel(cfg).to(self.device)
        else:
            raise NotImplementedError("Only DF is supported at this stage")

        # Optimizer
        if cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        self.l1 = nn.L1Loss()
        self.global_step = 0

        # ============================
        # Resume checkpoint loading
        # ============================
        if cfg.resume_path is not None and os.path.exists(cfg.resume_path):
            print(f"[Resume] Loading checkpoint: {cfg.resume_path}")

            state = torch.load(cfg.resume_path, map_location=self.device)

            missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
            if missing:
                print(f"[Resume] Missing keys in model: {missing}")
            if unexpected:
                print(f"[Resume] Unexpected keys in checkpoint: {unexpected}")

            self.opt.load_state_dict(state["optimizer"])

            if "scaler" in state and state["scaler"]:
                try:
                    self.scaler.load_state_dict(state["scaler"])
                except Exception as e:
                    print(f"[Resume] Warning: scaler could not be restored ({e})")
            else:
                print("[Resume] Warning: scaler state is empty, skipping AMP scaler restore")

            self.global_step = state.get("step", 0)
            print(f"[Resume] Resumed from step {self.global_step}")

    def _next_batch(self, iterator, loader):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator

    def run(self):
        print("=== TorchSAE Training Start ===")

        while self.global_step < self.cfg.max_steps:
            batch_a, self.iter_a = self._next_batch(self.iter_a, self.loader_a)
            batch_b, self.iter_b = self._next_batch(self.iter_b, self.loader_b)

            # 保存してプレビューで使う
            self.last_batch_a = batch_a.clone()
            self.last_batch_b = batch_b.clone()

            batch_a = batch_a.to(self.device)
            batch_b = batch_b.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                out_aa, out_bb, out_ab, out_ba = self.model(batch_a, batch_b)

                loss_aa = self.l1(out_aa, batch_a)
                loss_bb = self.l1(out_bb, batch_b)
                loss_ab = self.l1(out_ab, batch_b)
                loss_ba = self.l1(out_ba, batch_a)

                loss = loss_aa + loss_bb + loss_ab + loss_ba

            self.opt.zero_grad()
            self.scaler.scale(loss).backward()

            if self.cfg.clip_grad > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)

            self.scaler.step(self.opt)
            self.scaler.update()

            self.global_step += 1

            if self.global_step % 100 == 0:
                print(f"[{self.global_step}] loss={loss.item():.4f}")

            if self.global_step % self.cfg.preview_interval == 0:
                self._save_preview(out_aa, out_bb, out_ab, out_ba)

            if self.global_step % self.cfg.save_interval == 0:
                self._save_checkpoint()

        print("=== Training Finished ===")

    def _save_preview(self, aa, bb, ab, ba):
        # last_batch_a / last_batch_b は (B,3,H,W)
        img_a = self.last_batch_a.to(self.device)
        img_b = self.last_batch_b.to(self.device)

        # モデル出力
        aa = aa.to(self.device)
        bb = bb.to(self.device)
        ab = ab.to(self.device)

        # ★ DFModel には mask が無いので「ab の差分から疑似 mask」を作る
        #   → LIAE/SAEHD なら pred_dst_dstm を使う
        mask_b = torch.mean(torch.abs(bb - img_b), dim=1, keepdim=True)
        mask_b = (mask_b / mask_b.max()).clamp(0,1)

        preview = make_saehd_style_preview(
            img_a, img_b,
            aa, bb,
            ab,
            mask_b
        )

        # 1枚だけ保存
        p = preview[0].detach().cpu().permute(1,2,0).numpy()
        p = (p * 255).clip(0,255).astype("uint8")

        cv2.imwrite(
            f"/workspace/logs/previews/preview_{self.global_step:06d}.jpg",
            cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
        )


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
        print(f"[Checkpoint] Saved resume checkpoint: {out_path}")
