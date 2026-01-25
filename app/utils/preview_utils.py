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
    a_orig=None, aa=None, ab=None, mask_a=None,
    b_orig=None, bb=None, ba=None, mask_b=None,
    nrow=4
):
    """
    Returns a single grid image tensor (3,H,W) for saving.
    Missing entries are skipped safely.
    """

    rows = []

    # Row A
    row_a = []
    for x in [a_orig, aa, ab, mask_a]:
        if x is not None:
            row_a.append(to_image_tensor(x))
    if len(row_a) > 0:
        rows.append(torch.cat(row_a, dim=0))

    # Row B
    row_b = []
    for x in [b_orig, bb, ba, mask_b]:
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
