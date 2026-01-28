import torch
import torch.nn.functional as F

from app.utils.model_output import ModelOutput


def recon_loss_fn(loss_fn, outputs: ModelOutput, img_a, img_b, lm_a, lm_b):
    # A→A / B→B（主役）
    loss_aa = loss_fn(outputs.aa, img_a, lm_a)
    loss_bb = loss_fn(outputs.bb, img_b, lm_b)

    # ★ EXP-only（EXP を「顔から引きはがす」圧力を強める）
    loss_aa_exp_only = loss_fn(outputs.aa_exp_only, img_a, lm_a) * 0.35
    loss_bb_exp_only = loss_fn(outputs.bb_exp_only, img_b, lm_b) * 0.35

    # A→B / B→A（丸暗記を防ぐための弱い圧力）
    loss_ab = loss_fn(outputs.ab, img_b, lm_b) * 0.35
    loss_ba = loss_fn(outputs.ba, img_a, lm_a) * 0.35

    return (
        loss_aa
        + loss_bb
        + loss_aa_exp_only
        + loss_bb_exp_only
        + loss_ab
        + loss_ba
    )


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


# ★ A/B 対称な expression loss
def expression_loss_fn(outputs: ModelOutput, lm_a, lm_b):
    # A→B のランドマークを A に寄せる
    loss_ab = F.l1_loss(outputs.lm_ab_pred, lm_a)
    # B→A も将来的に入れるならここで（今は片側だけでもOK）
    # loss_ba = F.l1_loss(outputs.lm_ba_pred, lm_b)
    # return 0.5 * (loss_ab + loss_ba)
    return loss_ab


def compute_total_loss(
    outputs: ModelOutput,
    img_a, img_b,
    lm_a, lm_b,
    mask_a_gt, mask_b_gt,
    loss_fn,
    bce,
    mask_w: float,
    landmark_w: float,
    expr_w: float,
):
    recon = recon_loss_fn(loss_fn, outputs, img_a, img_b, lm_a, lm_b)
    mask = mask_loss_fn(outputs, mask_a_gt, mask_b_gt, bce)
    lm = landmark_loss_fn(outputs, lm_a, lm_b)
    expr = expression_loss_fn(outputs, lm_a, lm_b)

    # ★ expr_w は 0.05〜0.1 くらいからスタート推奨
    total = recon + mask_w * mask + landmark_w * lm + expr_w * expr

    return {
        "total": total,
        "recon": recon,
        "mask": mask,
        "landmark": lm,
        "expr": expr,
    }
