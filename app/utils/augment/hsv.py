import torch


def rgb_to_hsv(x):
    """
    x: [N,C,H,W], C=3, 0〜1 前提
    戻り: [N,3,H,W] (H,S,V)
    """
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    deltac = maxc - minc + 1e-8

    h = torch.zeros_like(maxc)
    mask = maxc == r
    h[mask] = ((g - b)[mask] / deltac[mask]) % 6
    mask = maxc == g
    h[mask] = ((b - r)[mask] / deltac[mask]) + 2
    mask = maxc == b
    h[mask] = ((r - g)[mask] / deltac[mask]) + 4
    h = h / 6.0

    s = deltac / (maxc + 1e-8)

    hsv = torch.stack([h, s, v], dim=1)
    return hsv.clamp(0.0, 1.0)


def hsv_to_rgb(x):
    """
    x: [N,3,H,W] (H,S,V), 0〜1
    戻り: [N,3,H,W] (R,G,B), 0〜1
    """
    h, s, v = x[:, 0], x[:, 1], x[:, 2]
    h6 = h * 6.0
    i = torch.floor(h6).long()
    f = h6 - i.float()

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = i % 6

    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)

    mask = i == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = i == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = i == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = i == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = i == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = i == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    return torch.stack([r, g, b], dim=1).clamp(0.0, 1.0)


def random_hsv_tensor(x, hsv_power: float):
    """
    本家 SAEHD 風 HSV augmentation。
    - Hue: ±hsv_power * 0.1 回転
    - Saturation: 1 ± hsv_power * 0.5
    - Value: 1 ± hsv_power * 0.3

    x: [N,3,H,W], -1〜1 or 0〜1
    """
    if hsv_power is None or hsv_power <= 0:
        return x

    x0 = x
    x = x0.clone()
    if x.min() < 0:
        x = (x + 1) / 2.0
    x = x.clamp(0.0, 1.0)

    hsv = rgb_to_hsv(x)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    N = x.shape[0]
    device = x.device

    hue_shift = (torch.rand(N, 1, 1, 1, device=device) - 0.5) * 0.2 * hsv_power
    sat_scale = 1.0 + (torch.rand(N, 1, 1, 1, device=device) - 0.5) * 1.0 * hsv_power
    val_scale = 1.0 + (torch.rand(N, 1, 1, 1, device=device) - 0.5) * 0.6 * hsv_power

    h = (h + hue_shift.squeeze(1)) % 1.0
    s = (s * sat_scale.squeeze(1)).clamp(0.0, 1.0)
    v = (v * val_scale.squeeze(1)).clamp(0.0, 1.0)

    hsv_aug = torch.stack([h, s, v], dim=1)
    x_aug = hsv_to_rgb(hsv_aug)

    if x0.min() < 0:
        x_aug = x_aug * 2.0 - 1.0
    return x_aug
