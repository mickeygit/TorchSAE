# app/trainers/base_trainer.py

import os
import time
import torch
from torch.cuda.amp import GradScaler
import torch.nn as nn

from torchvision.utils import save_image


# ================================
# 決着テスト（EXP に A の表情があるか判定）
# ================================
from torchvision.utils import save_image
from app.utils.preview_utils import to_image_tensor

@torch.no_grad()
def decisive_exp_test(trainer, img_a_raw, img_b_raw):
    model = trainer.model
    device = trainer.device

    # preview_dict から来る a_orig / b_orig を tensor 化
    img_a = to_image_tensor(img_a_raw).unsqueeze(0).to(device)
    img_b = to_image_tensor(img_b_raw).unsqueeze(0).to(device)

    # ★ encode() を通して「本番と同じ EXP」を取る
    zA_exp, _ = model.encode(img_a)
    zB_exp, _ = model.encode(img_b)

    # ID はゼロ固定
    z_id = torch.zeros_like(zA_exp)

    out_Aexp = model.decoder_B(
        torch.cat([z_id, zA_exp], dim=1),
        torch.cat([zA_exp, zA_exp], dim=1),
    )

    out_Bexp = model.decoder_B(
        torch.cat([z_id, zB_exp], dim=1),
        torch.cat([zB_exp, zB_exp], dim=1),
    )

    return out_Aexp, out_Bexp



class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.opt = None
        self.scaler = GradScaler(enabled=cfg.amp)

        self.global_step = 0
        self.start_time = time.time()

        self.save_dir = cfg.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train_step(self, batch_a, batch_b):
        raise NotImplementedError

    def make_preview(self, outputs, batch_a, batch_b):
        raise NotImplementedError(
            "Trainer must implement make_preview() and return preview_dict"
        )

    @torch.no_grad()
    def save_preview(self, outputs, batch_a, batch_b, suffix=""):
        try:
            preview_dict = self.make_preview(outputs, batch_a, batch_b)


            if not hasattr(self.model, "make_preview_grid"):
                raise NotImplementedError(
                    "Model must implement make_preview_grid(preview_dict)"
                )

            grid = self.model.make_preview_grid(preview_dict)

            import torchvision.utils as vutils

            preview_path = os.path.join(
                self.save_dir,
                f"preview_{self.global_step}{suffix}.png"
            )
            vutils.save_image(grid, preview_path, normalize=False)
            print(f"[Preview] Saved: {preview_path}")

            return preview_dict # ★ これを追加

        except Exception as e:
            print(f"[Preview] Failed: {e}")

    # ---------------------------------------------------------
    # checkpoint 保存 / ロード（変更なし）
    # ---------------------------------------------------------
    def _save_checkpoint(self):
        state = {
            "step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
        }

        model_type = getattr(self.cfg, "model_type", "model").upper()
        model_size = getattr(self.cfg, "model_size", 128)
        ae_dims = getattr(self.cfg, "ae_dims", 512)
        d_dims = getattr(self.cfg, "d_dims", 128)
        d_mask_dims = getattr(self.cfg, "d_mask_dims", 128)

        filename = (
            f"{model_type}_{model_size}_ae{ae_dims}_d{d_dims}_"
            f"mask{d_mask_dims}_step{self.global_step}.pth"
        )
        path = os.path.join(self.save_dir, filename)

        torch.save(state, path)
        print(f"[Save] Saved checkpoint: {path}")

        # 古い checkpoint 削除
        ckpts = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith(".pth")],
            key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x))
        )
        if len(ckpts) > 2:
            for f in ckpts[:-2]:
                try:
                    os.remove(os.path.join(self.save_dir, f))
                except:
                    pass

        # 古い preview 削除
        previews = sorted(
            [f for f in os.listdir(self.save_dir) if f.startswith("preview_")],
            key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x))
        )
        if len(previews) > 5:
            for f in previews[:-5]:
                try:
                    os.remove(os.path.join(self.save_dir, f))
                except:
                    pass

    def _load_resume(self):
        if getattr(self.cfg, "resume_path", None) is None:
            return

        resume_path = self.cfg.resume_path
        if not os.path.exists(resume_path):
            print(f"[Resume] resume_path not found: {resume_path}")
            return

        print(f"[Resume] Loading checkpoint: {resume_path}")
        state = torch.load(resume_path, map_location=self.device)

        self.model.load_state_dict(state["model"], strict=False)

        if "optimizer" in state:
            try:
                self.opt.load_state_dict(state["optimizer"])
            except:
                print("[Warn] Optimizer state mismatch. Skipping optimizer state.")

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

        # ★ LIAE_UD_256 のときだけ encoder.to_id をリセット
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "to_id"):
            if getattr(self.cfg, "reset_encoder_z_id", False):
                print("[Reset] encoder.to_id reset")
                for m in self.model.encoder.to_id.modules():
                    if isinstance(m, nn.Conv2d):
                        m.reset_parameters()

        # ★ FULL encoder reset（down1〜to_id/to_exp すべて）
        if hasattr(self.model, "reset_encoder_full"):
            if getattr(self.cfg, "reset_encoder_full", False):
                print("[Reset] FULL encoder reset")
                self.model.reset_encoder_full()

        # ★ decoder_A の out_block も必要ならリセット
        if hasattr(self.model, "reset_decoder_A_out_block"):
            if getattr(self.cfg, "reset_decoder_a", False):
                print("[Reset] decoder_A.out_block reset")
                self.model.reset_decoder_A_out_block()

        # ★ decoder_B の out_block も必要ならリセット
        if hasattr(self.model, "reset_decoder_B_out_block"):
            if getattr(self.cfg, "reset_decoder_b", False):
                print("[Reset] decoder_B.out_block reset")
                self.model.reset_decoder_B_out_block()

        if getattr(self.cfg, "reset_encoder_id_block", False):
            self.model.reset_encoder_id_block()

        # --- resume 後の lr 即時適用（AUTO モード用） ---
        if hasattr(self, "_get_auto_value") and getattr(self.cfg, "auto_mode", False):
            try:
                lr_now = self._get_auto_value("learning_rate")
                if lr_now is not None:
                    for g in self.opt.param_groups:
                        g["lr"] = lr_now
                    print(f"[AUTO-RESUME] Applied LR immediately after resume: {lr_now}")
            except Exception as e:
                print(f"[AUTO-RESUME] Failed to apply LR: {e}")

        print("============================================================")
        print("==              TorchSAE Trainer (PyTorch)                ==")
        print("============================================================")
        print(f"==  Model: {self.cfg.model_name}")
        print(f"==  Type: {self.cfg.model_type}")
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

                # target_steps で停止
                if getattr(self, "stop_training", False):
                    print("[Target] Training stopped by target_steps.")
                    break

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

                loss_dict, outputs = self.train_step(batch_a, batch_b)
                self.global_step += 1

                # target_steps で停止
                target_steps = getattr(self.cfg, "target_steps", None)
                if target_steps is not None and self.global_step >= target_steps:
                    print(f"[Target] Reached target steps ({target_steps}). Stopping training.")

                    # ★ target 到達時にも checkpoint を保存
                    try:
                        self._save_checkpoint()
                        self.save_preview(outputs, batch_a, batch_b, suffix="_final")
                    except Exception as e:
                        print(f"[Target] Failed to save checkpoint: {e}")

                    self.stop_training = True
                    break

                # --- ログ出力（50 step ごと） ---
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

                    # --- AUTO モードの現在パラメータ表示 ---
                    if getattr(self.cfg, "auto_mode", False):
                        try:
                            print(
                                f"        lr={loss_dict['lr']:.6f} "
                                f"mask_w={loss_dict['mask_w']:.3f} "
                                f"lm_w={loss_dict['landmark_w']:.3f} "
                                f"clip={loss_dict['clip_grad']:.1f}"
                            )
                            print(
                                f"        warp={loss_dict['warp_prob']:.3f} "
                                f"hsv={loss_dict['hsv_power']:.3f} "
                                f"noise={loss_dict['noise_power']:.3f} "
                                f"shell={loss_dict.get('shell_power', 0.0):.3f}"
                            )
                        except KeyError:
                            pass

                # checkpoint 保存
                if self.global_step % self.cfg.save_interval == 0:
                    self._save_checkpoint()

                # preview 保存
                if self.global_step % self.cfg.preview_interval == 0:

                    aa = outputs.aa
                    bb = outputs.bb
                    ab = outputs.ab
                    ba = outputs.ba

                    # ★ GPU に移動
                    a_orig = batch_a[0].to(self.device)
                    b_orig = batch_b[0].to(self.device)

                    from app.utils.debug_utils import tensor_minmax, debug_swap_quality

                    tensor_minmax("a_orig", a_orig)
                    tensor_minmax("aa", aa)
                    tensor_minmax("b_orig", b_orig)
                    tensor_minmax("bb", bb)

                    # ★ SWAP デバッグは try で囲む
                    try:
                        debug_swap_quality(
                            self.global_step,
                            ab, ba,
                            a_orig, b_orig
                        )
                    except Exception as e:
                        print(f"[SWAP] failed ({e})")


                    # ★ preview 保存は必ず実行
                    try:
                        preview_dict = self.save_preview(outputs, batch_a, batch_b)

                        # ================================
                        # EXP 決着テスト（encode() を通す正しい形）
                        # ================================
                        img_a = to_image_tensor(preview_dict["a_orig"]).unsqueeze(0).to(self.device)
                        img_b = to_image_tensor(preview_dict["b_orig"]).unsqueeze(0).to(self.device)

                        # ★ encode() を通して「学習時と同じ EXP」を取得
                        zA_exp, _ = self.model.encode(img_a)
                        zB_exp, _ = self.model.encode(img_b)

                        # ID はゼロ固定
                        z_id = torch.zeros_like(zA_exp)

                        # A の EXP だけを使った出力
                        out_Aexp = self.model.decoder_B(
                            torch.cat([z_id, zA_exp], dim=1),
                            torch.cat([zA_exp, zA_exp], dim=1),
                        )

                        # B の EXP だけを使った出力
                        out_Bexp = self.model.decoder_B(
                            torch.cat([z_id, zB_exp], dim=1),
                            torch.cat([zB_exp, zB_exp], dim=1),
                        )

                        save_image(out_Aexp, f"{self.save_dir}/exp_Aexp.png")
                        save_image(out_Bexp, f"{self.save_dir}/exp_Bexp.png")

                        print("[DecisiveEXP] Saved exp_Aexp.png / exp_Bexp.png")

                    except Exception as e:
                        print(f"[Preview] Failed: {e}")



        except KeyboardInterrupt:
            print("\n[Exit] Caught Ctrl+C, saving final checkpoint...")
            self._save_checkpoint()

            # ★ 最終プレビュー保存
            try:
                self.save_preview(outputs, batch_a, batch_b, suffix="_final")
            except Exception as e:
                print(f"[Exit] Failed to save final preview: {e}")

            print("[Exit] Done.")
