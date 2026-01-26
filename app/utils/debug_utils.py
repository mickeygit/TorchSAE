# app/utils/debug_utils.py

import torch

DEBUG = True   # ← ON/OFF を一括管理


def tensor_minmax(name, x):
    """
    汎用 min/max 出力
    """
    if not DEBUG:
        return
    try:
        x = x.detach()
        print(f"[DEBUG] {name}: min={x.min().item():.4f}, max={x.max().item():.4f}")
    except Exception as e:
        print(f"[DEBUG] {name}: failed ({e})")


def tensor_stats(name, x):
    """
    min/max + mean/std + shape をまとめて出す
    """
    if not DEBUG:
        return
    try:
        x = x.detach()
        print(
            f"[DEBUG] {name}: "
            f"shape={tuple(x.shape)}, "
            f"min={x.min().item():.4f}, "
            f"max={x.max().item():.4f}, "
            f"mean={x.mean().item():.4f}, "
            f"std={x.std().item():.4f}"
        )
    except Exception as e:
        print(f"[DEBUG] {name}: failed ({e})")


def check_nan_inf(name, x):
    """
    NaN / Inf チェック
    """
    if not DEBUG:
        return
    try:
        x = x.detach()
        nan = torch.isnan(x).any().item()
        inf = torch.isinf(x).any().item()
        print(f"[DEBUG] {name}: NaN={nan}, Inf={inf}")
    except Exception as e:
        print(f"[DEBUG] {name}: failed ({e})")


# ============================================================
# ★ フェーズ4（表情SWAP強化）用デバッグ関数
# ============================================================

def debug_latents(step, z_exp, z_id):
    """
    z_exp / z_id の分離度を確認
    """
    if not DEBUG:
        return
    try:
        print(
            f"[LATENT] step={step} "
            f"z_exp mean={z_exp.mean().item():.4f} std={z_exp.std().item():.4f} "
            f"z_id mean={z_id.mean().item():.4f} std={z_id.std().item():.4f}"
        )
    except Exception as e:
        print(f"[LATENT] failed ({e})")


def debug_decoder(step, aa, ab, ba):
    """
    decoder 出力の min/max を確認
    """
    if not DEBUG:
        return
    try:
        print(
            f"[DEC] step={step} "
            f"aa min={aa.min().item():.3f} max={aa.max().item():.3f} "
            f"ab min={ab.min().item():.3f} max={ab.max().item():.3f} "
            f"ba min={ba.min().item():.3f} max={ba.max().item():.3f}"
        )
    except Exception as e:
        print(f"[DEC] failed ({e})")


def debug_swap_quality(step, ab, ba, a_orig, b_orig):
    """
    SWAP の質を数値で評価
    ab が B にどれだけ近いか
    ba が A にどれだけ近いか
    """
    if not DEBUG:
        return
    try:
        diff_ab = (ab - b_orig).abs().mean().item()
        diff_ba = (ba - a_orig).abs().mean().item()
        print(
            f"[SWAP] step={step} "
            f"A→B diff={diff_ab:.4f} "
            f"B→A diff={diff_ba:.4f}"
        )
    except Exception as e:
        print(f"[SWAP] failed ({e})")
