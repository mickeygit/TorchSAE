import torch
import torch.nn.functional as F
import random


def random_warp_tensor(
    x,
    warp_prob: float,
    grid_size: int = 5,
    max_magnitude: float = 0.05,
):
    """
    本家 SAEHD の random_warp をイメージした実装。
    - 低解像度グリッド（grid_size x grid_size）をランダム変形
    - バイキュービック補間で元解像度に戻す
    - warp_prob に応じてサンプルごとに適用

    x: [N, C, H, W]
    warp_prob: 0〜1
    """
    if warp_prob is None or warp_prob <= 0:
        return x

    N, C, H, W = x.shape
    device = x.device

    # warp するサンプルを決定
    mask = torch.rand(N, device=device) < warp_prob
    if not mask.any():
        return x

    x_out = x.clone()

    # 低解像度グリッド座標（[-1,1]）
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, grid_size, device=device),
        torch.linspace(-1, 1, grid_size, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([gx, gy], dim=-1)  # [G,G,2]

    for i in torch.nonzero(mask, as_tuple=False).view(-1):
        # ランダムオフセット
        offset = (torch.rand(grid_size, grid_size, 2, device=device) - 0.5) * 2.0 * max_magnitude
        warped_grid = base_grid + offset  # [G,G,2]

        # 元解像度にアップサンプル（バイキュービック）
        warped_grid_up = F.interpolate(
            warped_grid.permute(2, 0, 1).unsqueeze(0),  # [1,2,G,G]
            size=(H, W),
            mode="bicubic",
            align_corners=True,
        ).permute(0, 2, 3, 1)  # [1,H,W,2]

        x_out[i:i+1] = F.grid_sample(
            x[i:i+1],
            warped_grid_up,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    return x_out
