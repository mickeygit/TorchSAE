import torch
import torch.nn.functional as F
import random


def shell_augment(
    x,
    mask,
    power: float,
    blur_max_ksize: int = 9,
    color_jitter_strength: float = 0.15,
    shadow_strength: float = 0.25,
):
    """
    本家 SAEHD の shell augment をイメージした実装。
    - 顔マスクの「外側」を中心に、ぼかし・色変化・影・ノイズを入れる。
    - power: 0〜1（AUTO モードで 0→1→0 みたいに変化させる想定）

    x:    [N,3,H,W]  （-1〜1 or 0〜1）
    mask: [N,1,H,W]  （0〜1, 顔=1, 背景=0）
    """
    if power is None or power <= 0:
        return x

    x0 = x
    x = x0.clone()
    N, C, H, W = x.shape
    device = x.device

    # 値域を 0〜1 に寄せて処理
    if x.min() < 0:
        x = (x + 1) / 2.0
    x = x.clamp(0.0, 1.0)

    # マスクを膨張させて「顔周辺」も含める
    k = 5
    kernel = torch.ones(1, 1, k, k, device=device) / (k * k)
    mask_blur = F.conv2d(mask, kernel, padding=k // 2)
    shell = 1.0 - mask_blur.clamp(0.0, 1.0)

    # 1) 背景ぼかし
    ksize = int(blur_max_ksize * power)
    if ksize % 2 == 0:
        ksize += 1
    if ksize >= 3:
        pad = ksize // 2
        blur_kernel = torch.ones(3, 1, ksize, ksize, device=device)
        blur_kernel = blur_kernel / (ksize * ksize)
        x_blur = F.conv2d(x, blur_kernel, padding=pad, groups=3)
    else:
        x_blur = x

    shell_w = (0.3 + 0.7 * power) * shell
    x = x * (1.0 - shell_w) + x_blur * shell_w

    # 2) 背景の色揺らし
    jitter = (torch.rand(N, 3, 1, 1, device=device) - 0.5) * 2.0 * color_jitter_strength * power
    x_jit = (x + jitter).clamp(0.0, 1.0)
    color_w = 0.5 * power * shell
    x = x * (1.0 - color_w) + x_jit * color_w

    # 3) 影
    ys = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
    angle = random.random() * 3.14159 * 2.0
    dx = torch.cos(torch.tensor(angle, device=device))
    dy = torch.sin(torch.tensor(angle, device=device))
    grad = (xs * dx + ys * dy)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    grad = grad ** 1.5

    shadow = 1.0 - shadow_strength * power * grad
    shadow = shadow.clamp(0.0, 1.0)

    shadow_map = 1.0 - shell * (1.0 - shadow)
    x = x * shadow_map

    # 4) 軽いノイズ
    noise_scale = 0.03 * power
    noise = torch.randn_like(x) * noise_scale
    x = (x + noise * shell).clamp(0.0, 1.0)

    if x0.min() < 0:
        x = x * 2.0 - 1.0

    return x
