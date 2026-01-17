import os
import numpy as np
from PIL import Image


def tensor_to_image(t):
    arr = (t.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_preview_grid(step, aa, bb, ab, ba, a_orig, b_orig, out_dir, ext="jpg"):
    os.makedirs(out_dir, exist_ok=True)

    a0 = tensor_to_image(a_orig[0])
    b0 = tensor_to_image(b_orig[0])
    aa0 = tensor_to_image(aa[0])
    bb0 = tensor_to_image(bb[0])
    ab0 = tensor_to_image(ab[0])
    ba0 = tensor_to_image(ba[0])

    w, h = a0.size
    grid = Image.new("RGB", (w * 3, h * 2))

    grid.paste(a0, (0, 0))
    grid.paste(aa0, (w, 0))
    grid.paste(ab0, (w * 2, 0))

    grid.paste(b0, (0, h))
    grid.paste(bb0, (w, h))
    grid.paste(ba0, (w * 2, h))

    filename = os.path.join(out_dir, f"preview_{step}.{ext}")

    # ★ JPG → JPEG に変換（ここが重要）
    fmt = "JPEG" if ext.lower() == "jpg" else ext.upper()

    grid.save(filename, format=fmt)
    print(f"[Preview] saved: {filename}")

# app/utils/preview.py

import os
import torch
import torchvision.utils as vutils
import torch.nn.functional as F


def _extract_mask_from_rgba(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, C, H, W)
    RGBA を想定して A チャンネルをマスクとして取り出す。
    C==4 でない場合は None を返す。
    """
    if x.shape[1] == 4:
        # A チャンネルを [0,1] に正規化して返す
        mask = x[:, 3:4]
        mask = torch.clamp(mask, 0.0, 1.0)
        return mask
    return None


def _make_overlay(face: torch.Tensor, mask: torch.Tensor, color=(1.0, 0.0, 0.0), alpha=0.4):
    """
    face: (N, 3, H, W) [0,1]
    mask: (N, 1, H, W) [0,1]
    color: overlay color (R,G,B)
    alpha: overlay strength
    """
    if face.shape[1] == 4:
        face = face[:, :3]

    # マスクを 3ch に拡張
    mask3 = mask.repeat(1, 3, 1, 1)

    # カラーのテンソルを作成
    c = torch.tensor(color, dtype=face.dtype, device=face.device).view(1, 3, 1, 1)
    color_img = c.expand_as(face)

    # face と color_img をマスクでブレンド
    overlay = face * (1 - alpha * mask3) + color_img * (alpha * mask3)
    overlay = torch.clamp(overlay, 0.0, 1.0)
    return overlay

def save_liae_preview_with_masks(
    step: int,
    aa: torch.Tensor,
    bb: torch.Tensor,
    ab: torch.Tensor,
    ba: torch.Tensor,
    a_orig: torch.Tensor,
    b_orig: torch.Tensor,
    out_dir: str,
    ext: str = "jpg",
):
    import os
    import torchvision.utils as vutils
    import torch

    os.makedirs(out_dir, exist_ok=True)

    # RGB 化
    def to_rgb(x):
        return x[:, :3] if x.shape[1] == 4 else x

    a_rgb = to_rgb(a_orig)
    b_rgb = to_rgb(b_orig)
    aa_rgb = to_rgb(aa)
    bb_rgb = to_rgb(bb)
    ab_rgb = to_rgb(ab)
    ba_rgb = to_rgb(ba)

    # ---- 1行目（横に並べる）----
    row1 = torch.cat([a_rgb, aa_rgb, ab_rgb, ba_rgb], dim=3)[0:1]

    # ---- 2行目 ----
    row2 = torch.cat([b_rgb, bb_rgb, ba_rgb, ab_rgb], dim=3)[0:1]

    rows = [row1, row2]

    # ---- 3行目（マスクがある場合のみ）----
    if ab.shape[1] == 4:
        ab_mask = ab[:, 3:4]
        ba_mask = ba[:, 3:4]

        ab_mask3 = ab_mask.repeat(1, 3, 1, 1)
        ba_mask3 = ba_mask.repeat(1, 3, 1, 1)

        def overlay(face, mask):
            mask3 = mask.repeat(1, 3, 1, 1)
            red = torch.tensor([1, 0, 0], device=face.device).view(1, 3, 1, 1)
            return face * (1 - 0.5 * mask3) + red * (0.5 * mask3)

        ab_overlay = overlay(ab_rgb, ab_mask)
        ba_overlay = overlay(ba_rgb, ba_mask)

        row3 = torch.cat([ab_mask3, ab_overlay, ba_mask3, ba_overlay], dim=3)[0:1]
        rows.append(row3)

    # ---- 行を縦に積む（dim=2）----
    grid = torch.cat(rows, dim=2)

    # 保存
    path = os.path.join(out_dir, f"preview_{step:06d}.{ext}")
    vutils.save_image(grid, path, normalize=True, value_range=(0, 1))
