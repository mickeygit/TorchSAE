from dataclasses import dataclass


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
    archi: str              # "df" (LIAE later)
    model_size: int         # 128 / 256 / 384 / 512
    face_type: str          # full / head / wf

    ae_dims: int            # autoencoder bottleneck dims
    e_dims: int             # encoder channel dims
    d_dims: int             # decoder channel dims
    d_mask_dims: int        # mask decoder dims

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    batch_size: int
    max_steps: int

    optimizer: str          # adam / adamw
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
    # Utility
    # ---------------------------------------------------------
    def is_df(self) -> bool:
        return self.archi == "df"

    def summary(self):
        """Optional: print config summary for debugging."""
        print("=== TorchSAE TrainConfig ===")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("============================")
