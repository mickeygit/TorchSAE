import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF


class FaceDataset(Dataset):
    """
    TorchSAE 用の最小 FaceDataset（安全版）
    - 画像読み込み
    - リサイズ
    - augment（random-warp / hsv / noise）
    - dtype を float32 に統一（AMP との相性改善）
    """

    def __init__(self, root_dir, cfg):
        self.root_dir = root_dir
        self.cfg = cfg

        # 画像ファイル一覧
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        self.size = cfg.model_size

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        return img

    # ---------------------------------------------------------
    # Augment: random warp（簡易版）
    # ---------------------------------------------------------
    def _random_warp(self, img):
        w, h = img.size
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    # ---------------------------------------------------------
    # Augment: HSV
    # ---------------------------------------------------------
    def _random_hsv(self, img):
        if self.cfg.random_hsv_power <= 0:
            return img

        factor = 1.0 + random.uniform(-self.cfg.random_hsv_power,
                                      self.cfg.random_hsv_power)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    # ---------------------------------------------------------
    # Augment: Noise
    # ---------------------------------------------------------
    def _random_noise(self, tensor):
        if self.cfg.random_noise_power <= 0:
            return tensor

        noise = torch.randn_like(tensor) * self.cfg.random_noise_power
        return torch.clamp(tensor + noise, 0.0, 1.0)

    # ---------------------------------------------------------
    # __getitem__
    # ---------------------------------------------------------
    def __getitem__(self, idx):
        path = self.files[idx]
        img = self._load_image(path)

        # augment
        if self.cfg.random_warp:
            img = self._random_warp(img)

        img = self._random_hsv(img)

        # PIL → Tensor（float32）
        tensor = TF.to_tensor(img).float()

        # noise augment
        tensor = self._random_noise(tensor)

        return tensor
