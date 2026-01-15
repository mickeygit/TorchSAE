from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    data_dir_a: str
    data_dir_b: str

    # ---------------------------------------------------------
    # Model structure
    # ---------------------------------------------------------
    archi: str
    model_size: int
    face_type: str

    ae_dims: int
    e_dims: int
    d_dims: int
    d_mask_dims: int

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    batch_size: int
    max_steps: int

    optimizer: str
    lr: float
    clip_grad: float
    seed: int

    # ---------------------------------------------------------
    # Augmentation
    # ---------------------------------------------------------
    random_warp: bool
    random_hsv_power: float
    random_noise_power: float

    # ---------------------------------------------------------
    # Preview / Save
    # ---------------------------------------------------------
    preview_interval: int
    save_interval: int

    # ---------------------------------------------------------
    # AMP
    # ---------------------------------------------------------
    amp: bool

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------
    resume_path: Optional[str] = None

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def is_df(self) -> bool:
        return self.archi == "df"

    def summary(self):
        print("=== TorchSAE TrainConfig ===")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("============================")
