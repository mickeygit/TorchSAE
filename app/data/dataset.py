import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF


class FaceDataset(Dataset):
    """
    TorchSAE 用 FaceDataset（ランドマーク対応版）
    - 画像 + landmarks が両方そろっているファイルだけを使う
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
            base = os.path.splitext(path)[0]
            lm_path = base + "_landmarks.npy"
            if not os.path.exists(lm_path):
                continue
            lm = np.load(lm_path)
            if lm.shape != (68, 2):
                continue
            valid_files.append(path)

        if len(valid_files) == 0:
            raise RuntimeError(f"No valid images with landmarks found in {root_dir}")

        print(f"[FaceDataset] {root_dir}: {len(valid_files)} / {len(all_files)} images have valid landmarks.")
        self.files = valid_files

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        return img

    def _load_landmarks(self, img_path):
        base = os.path.splitext(img_path)[0]
        lm_path = base + "_landmarks.npy"
        lm = np.load(lm_path).astype(np.float32)
        return lm  # ここまで来た時点で shape は保証済み

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

    def __getitem__(self, idx):
        path = self.files[idx]

        img = self._load_image(path)

        if self.cfg.random_warp:
            img = self._random_warp(img)
        img = self._random_hsv(img)

        tensor = TF.to_tensor(img).float()
        tensor = self._random_noise(tensor)

        landmarks = self._load_landmarks(path)

        return tensor, landmarks
