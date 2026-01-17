from dataclasses import dataclass
from typing import Optional


# app/train_config.py

class TrainConfig:
    def __init__(self, **kwargs):
        # JSON のキーを全部属性にする
        for k, v in kwargs.items():
            setattr(self, k, v)

        # デフォルト値
        defaults = {
            "model_name": "LIAE",
            "model_size": 128,

            "data_a": None,
            "data_b": None,

            "save_dir": "/workspace/models",
            "resume_path": None,

            "batch_size": 8,
            "num_workers": 4,

            "lr": 0.00008,
            "optimizer": "adam",
            "clip_grad": 1,

            "amp": True,

            "random_warp": False,
            "random_hsv_power": 0.0,
            "random_noise_power": 0.0,

            "e_dims": 128,
            "ae_dims": 512,
            "d_dims": 128,
            "d_mask_dims": 128,
            "learn_mask": True,

            "use_landmarks": True,
            "mask_loss_weight": 0.1,
            "landmark_loss_weight": 0.01,

            "preview_interval": 300,
            "save_interval": 500
        }

        # デフォルト値を埋める
        for k, v in defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)


    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def summary(self):
        print("=== TrainConfig ===")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("===================")
