# app/trainers/trainer_liae.py
from app.trainers.base_trainer import BaseTrainer
from app.models.autoencoder_liae import LIAEModel
import torch


class TrainerLIAE(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = LIAEModel(cfg).to(self.device)

        if cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
