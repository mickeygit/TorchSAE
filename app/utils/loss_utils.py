# app/utils/loss_utils.py

import torch
import torch.nn.functional as F

from app.utils.model_output import ModelOutput


def recon_loss_fn(loss_fn, outputs: ModelOutput, img_a, img_b, lm_a, lm_b):
    loss_aa = loss_fn(outputs.aa, img_a, lm_a)
    loss_bb = loss_fn(outputs.bb, img_b, lm_b)
    loss_ab = loss_fn(outputs.ab, img_b, lm_b)
    loss_ba = loss_fn(outputs.ba, img_a, lm_a)
    return loss_aa + loss_bb + loss_ab + loss_ba


def mask_loss_fn(outputs: ModelOutput, mask_a_gt, mask_b_gt, bce):
    return (
        bce(outputs.mask_a_pred, mask_a_gt) +
        bce(outputs.mask_b_pred, mask_b_gt)
    )


def landmark_loss_fn(outputs: ModelOutput, lm_a, lm_b):
    return (
        F.l1_loss(outputs.lm_a_pred, lm_a) +
        F.l1_loss(outputs.lm_b_pred, lm_b)
    )


def compute_total_loss(
    outputs: ModelOutput,
    img_a, img_b,
    lm_a, lm_b,
    mask_a_gt, mask_b_gt,
    loss_fn,
    bce,
    mask_w: float,
    landmark_w: float,
):
    recon = recon_loss_fn(loss_fn, outputs, img_a, img_b, lm_a, lm_b)
    mask = mask_loss_fn(outputs, mask_a_gt, mask_b_gt, bce)
    lm = landmark_loss_fn(outputs, lm_a, lm_b)

    total = recon + mask_w * mask + landmark_w * lm

    return {
        "total": total,
        "recon": recon,
        "mask": mask,
        "landmark": lm,
    }
