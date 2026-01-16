# app/trainers/trainer_df.py
from app.trainers.base_trainer import BaseTrainer
from app.models.autoencoder_df import DFModel
import torch


class TrainerDF(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = DFModel(cfg).to(self.device)

        if cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
