import torch
import torch.nn.functional as F

from app.utils.model_output import ModelOutput


def recon_loss_fn(loss_fn, outputs: ModelOutput, img_a, img_b, lm_a, lm_b):
    loss_aa = loss_fn(outputs.aa, img_a, lm_a)
    loss_bb = loss_fn(outputs.bb, img_b, lm_b)
    # ★ ab / ba の再構成圧力はいったん切る
    return loss_aa + loss_bb


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


# ★ 先に定義（順番が重要）
def expression_loss_fn(outputs: ModelOutput, lm_a):
    # A→B のランドマークを A のランドマークに寄せる
    return F.l1_loss(outputs.lm_ab_pred, lm_a)


def compute_total_loss(
    outputs: ModelOutput,
    img_a, img_b,
    lm_a, lm_b,
    mask_a_gt, mask_b_gt,
    loss_fn,
    bce,
    mask_w: float,
    landmark_w: float,
    expr_w: float,   # ★ 追加
):
    recon = recon_loss_fn(loss_fn, outputs, img_a, img_b, lm_a, lm_b)
    mask = mask_loss_fn(outputs, mask_a_gt, mask_b_gt, bce)
    lm = landmark_loss_fn(outputs, lm_a, lm_b)
    expr = expression_loss_fn(outputs, lm_a)

    total = recon + mask_w * mask + landmark_w * lm + expr_w * expr

    return {
        "total": total,
        "recon": recon,
        "mask": mask,
        "landmark": lm,
        "expr": expr,
    }
