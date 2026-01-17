from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    # ---------------------------------------------------------
    # Model selection
    # ---------------------------------------------------------
    model_type: str          # "df" or "liae"

    # ---------------------------------------------------------
    # Model structure
    # ---------------------------------------------------------
    model_size: int
    ae_dims: int
    e_dims: int
    d_dims: int
    d_mask_dims: int
    inter_dims: int          # LIAE 用
    learn_mask: bool

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    data_dir_a: str
    data_dir_b: str

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    batch_size: int
    max_steps: int
    optimizer: str
    lr: float
    clip_grad: float
    amp: bool

    # ---------------------------------------------------------
    # Augmentation（dataset.py が参照する）
    # ---------------------------------------------------------
    random_warp: bool = False
    random_hsv_power: float = 0.0
    random_noise_power: float = 0.0

    # ---------------------------------------------------------
    # Landmarks（★ 追加：DSSIM + eyes-mouth-prio に必須）
    # ---------------------------------------------------------
    use_landmarks: bool = True

    # ---------------------------------------------------------
    # Preview / Save
    # ---------------------------------------------------------
    preview_interval: int = 1000
    save_interval: int = 5000

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------
    resume_path: Optional[str] = None

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def summary(self):
        print("=== TrainConfig ===")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("===================")
