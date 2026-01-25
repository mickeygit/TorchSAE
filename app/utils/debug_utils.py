# app/utils/debug_utils.py

import torch

def tensor_minmax(name, x):
    """
    汎用 min/max 出力
    x: Tensor
    """
    try:
        x = x.detach()
        print(f"[DEBUG] {name}: min={x.min().item():.4f}, max={x.max().item():.4f}")
    except Exception as e:
        print(f"[DEBUG] {name}: failed ({e})")


def tensor_stats(name, x):
    """
    min/max + mean/std + shape をまとめて出す
    """
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
    try:
        x = x.detach()
        nan = torch.isnan(x).any().item()
        inf = torch.isinf(x).any().item()
        print(f"[DEBUG] {name}: NaN={nan}, Inf={inf}")
    except Exception as e:
        print(f"[DEBUG] {name}: failed ({e})")
