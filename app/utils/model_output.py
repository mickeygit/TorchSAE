# app/utils/model_output.py

from dataclasses import dataclass
import torch

@dataclass
class ModelOutput:
    aa: torch.Tensor
    bb: torch.Tensor
    ab: torch.Tensor
    ba: torch.Tensor
    mask_a_pred: torch.Tensor
    mask_b_pred: torch.Tensor
    lm_a_pred: torch.Tensor
    lm_b_pred: torch.Tensor
