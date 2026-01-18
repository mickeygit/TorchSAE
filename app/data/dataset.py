import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF

from app.utils.DFLJPG import DFLJPG


class FaceDataset(Dataset):
    """
    TorchSAE 用 FaceDataset（DFLJPG メタ対応版）
    - DFLJPG の APP15 に埋め込まれた
        ・画像
        ・landmarks (68,2)
        ・xseg_mask (H,W,1)
      が両方そろっているファイルだけを学習に使う
    """

    def __init__(self, root_dir, cfg):
        self.root_dir = root_dir
        self.cfg = cfg
        self.size = cfg.model_size

        exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        all_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]

        valid_files = []
        for path in all_files:
            jpg = DFLJPG.load(path)
            if jpg is None:
                continue

            lms = jpg.get_landmarks()
            mask = jpg.get_xseg_mask()

            # ★ landmarks と xseg が両方そろっているものだけ採用
            if lms is None or mask is None:
                continue

            # landmarks shape check
            if lms.shape != (68, 2):
                continue

            valid_files.append(path)

        if len(valid_files) == 0:
            raise RuntimeError(f"No valid images with landmarks + xseg found in {root_dir}")

        print(f"[FaceDataset] {root_dir}: {len(valid_files)} / {len(all_files)} images have valid meta.")
        self.files = valid_files

    def __len__(self):
        return len(self.files)

    # ============================================================
    # 画像読み込み（DFLJPG → PIL）
    # ============================================================
    def _load_image(self, jpg):
        img = jpg.get_img()  # BGR
        img = img[:, :, ::-1]  # BGR → RGB
        img = Image.fromarray(img)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        return img

    # ============================================================
    # augment
    # ============================================================
    def _random_warp(self, img):
        w, h = img.size
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    def _random_hsv(self, img):
        if self.cfg.random_hsv_power <= 0:
            return img
        factor = 1.0 + random.uniform(-self.cfg.random_hsv_power,
                                      self.cfg.random_hsv_power)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def _random_noise(self, tensor):
        if self.cfg.random_noise_power <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.cfg.random_noise_power
        return torch.clamp(tensor + noise, 0.0, 1.0)

    # ============================================================
    # main
    # ============================================================
    def __getitem__(self, idx):
        path = self.files[idx]

        jpg = DFLJPG.load(path)

        # 画像
        img = self._load_image(jpg)

        # augment
        if self.cfg.random_warp:
            img = self._random_warp(img)
        img = self._random_hsv(img)

        tensor = TF.to_tensor(img).float()
        tensor = self._random_noise(tensor)

        # landmarks
        landmarks = jpg.get_landmarks().astype(np.float32)

        # xseg mask (H,W,1)
        mask = jpg.get_xseg_mask().astype(np.float32)

        # ★ 0/1 に2値化してから uint8 に変換
        mask = (mask[:, :, 0] > 0.5).astype(np.uint8) * 255

        # ★ NEAREST でリサイズ（補間しない）
        mask = Image.fromarray(mask)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        # ★ float に戻す（0.0 or 1.0）
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = mask[None, :, :]  # (1,H,W)

        return tensor, landmarks, mask
