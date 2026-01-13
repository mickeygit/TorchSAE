import os
import torch
import torchvision.utils as vutils


def save_preview_grid(step, aa, bb, ab, ba, out_dir="preview"):
    """
    4つの出力画像を 2x2 グリッドで保存する。
    aa: A→A
    bb: B→B
    ab: A→B
    ba: B→A
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1枚だけ取り出す（バッチの先頭）
    aa = aa[0].detach().cpu()
    bb = bb[0].detach().cpu()
    ab = ab[0].detach().cpu()
    ba = ba[0].detach().cpu()

    # 2x2 グリッドに並べる
    grid = torch.stack([aa, ab, ba, bb], dim=0)

    # 保存
    path = os.path.join(out_dir, f"preview_{step}.png")
    vutils.save_image(grid, path, nrow=2)

    print(f"[Preview] saved: {path}")
