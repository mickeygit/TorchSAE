import torch
import torchvision.utils as vutils
from pathlib import Path


def to_image_tensor(x):
    """
    x: (B, C, H, W) tensor in arbitrary range
    returns: clamped (0,1) tensor on CPU for saving
    """
    if x is None:
        return None

    # detach → cpu → clamp
    x = x.detach().float().cpu()
    x = torch.clamp(x, 0.0, 1.0)
    return x


def prepare_mask(mask):
    """
    mask: (B, 1, H, W)
    returns: (B, 3, H, W) for visualization
    """
    if mask is None:
        return None

    mask = mask.detach().float().cpu()
    mask = torch.clamp(mask, 0.0, 1.0)

    # (B,1,H,W) → (B,3,H,W)
    mask = mask.repeat(1, 3, 1, 1)
    return mask

def make_preview_grid(
    a_orig=None, aa=None, ab=None, aa_exp_only=None, mask_a=None,
    b_orig=None, bb=None, ba=None, bb_exp_only=None, mask_b=None,
    nrow=6
):
    """
    Returns a single grid image tensor (3,H,W) for saving.
    Missing entries are skipped safely.
    """

    rows = []

    # Row A
    row_a = []
    for x in [a_orig, aa, ab, aa_exp_only, mask_a]:
        if x is not None:
            row_a.append(to_image_tensor(x))
    if len(row_a) > 0:
        rows.append(torch.cat(row_a, dim=0))

    # Row B
    row_b = []
    for x in [b_orig, bb, ba, bb_exp_only, mask_b]:
        if x is not None:
            row_b.append(to_image_tensor(x))
    if len(row_b) > 0:
        rows.append(torch.cat(row_b, dim=0))

    if len(rows) == 0:
        return None

    # stack rows vertically
    grid = torch.cat(rows, dim=0)
    return grid


def save_preview(grid, save_path):
    """
    grid: (N, C, H, W) or (C, H, W)
    save_path: str or Path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # If grid is (C,H,W), add batch dim
    if grid.dim() == 3:
        grid = grid.unsqueeze(0)

    # Save as PNG
    vutils.save_image(grid, str(save_path), nrow=grid.size(0), normalize=False)

# app/utils/preview_utils.py

import torch
from app.utils.model_output import ModelOutput


def to_image_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()

    if x.dtype == torch.uint8:
        x = x / 255.0

    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    return x.clamp(0.0, 1.0)

@torch.no_grad()
def build_preview_dict(
    outputs: ModelOutput,
    batch_a,
    batch_b,
):
    img_a, lm_a, _ = batch_a
    img_b, lm_b, _ = batch_b

    img_a_0 = img_a[0].detach().cpu()
    img_b_0 = img_b[0].detach().cpu()

    aa = outputs.aa[0].detach().cpu()
    bb = outputs.bb[0].detach().cpu()
    ab = outputs.ab[0].detach().cpu()
    ba = outputs.ba[0].detach().cpu()

    # ★ EXP-only 追加
    aa_exp_only = outputs.aa_exp_only[0].detach().cpu()
    bb_exp_only = outputs.bb_exp_only[0].detach().cpu()

    mask_a = torch.sigmoid(outputs.mask_a_pred[0]).detach().cpu()
    mask_b = torch.sigmoid(outputs.mask_b_pred[0]).detach().cpu()

    def to_01(x):
        return x.float().clamp(0.0, 1.0)

    return {
        "a_orig": to_01(img_a_0),
        "b_orig": to_01(img_b_0),
        "aa": to_01(aa),
        "bb": to_01(bb),
        "ab": to_01(ab),
        "ba": to_01(ba),

        # ★ EXP-only を preview に追加
        "aa_exp_only": to_01(aa_exp_only),
        "bb_exp_only": to_01(bb_exp_only),

        "mask_a": to_01(mask_a),
        "mask_b": to_01(mask_b),
    }
