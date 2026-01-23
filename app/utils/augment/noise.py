import torch
import torch.nn.functional as F


def add_noise_tensor(x, noise_power: float):
    """
    本家 SAEHD 風の複合ノイズを簡略再現：
    - Gaussian ノイズ
    - JPEG 風ブロックノイズ（ダウンサンプル→アップサンプル→量子化）

    x: [N,C,H,W]
    """
    if noise_power is None or noise_power <= 0:
        return x

    x0 = x
    x = x0.clone()
    N, C, H, W = x.shape

    # Gaussian ノイズ
    if x.min() < 0:
        g_scale = 0.05 * noise_power
    else:
        g_scale = 0.02 * noise_power
    gauss = torch.randn_like(x) * g_scale
    x = x + gauss

    # JPEG 風ブロックノイズ
    down = max(4, int(16 * (1.0 - 0.7 * noise_power)))
    x_small = F.interpolate(x, size=(H // down, W // down), mode="area")
    x_block = F.interpolate(x_small, size=(H, W), mode="nearest")

    # 量子化
    if x.min() < 0:
        xq = ((x_block + 1) * 127.5).round() / 127.5 - 1
    else:
        xq = (x_block * 255.0).round() / 255.0

    alpha = 0.5 * noise_power
    x = x * (1 - alpha) + xq * alpha

    x = x.clamp(x0.min(), x0.max())
    return x
