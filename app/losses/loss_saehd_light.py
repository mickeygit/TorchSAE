import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  PyTorch DSSIM  (DFL の dssim と同等の構造)
# ============================================================

class DSSIM(nn.Module):
    def __init__(self, kernel_size=11, k1=0.01, k2=0.03):
        super().__init__()
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2

        # ガウシアンカーネル生成
        sigma = 1.5
        coords = torch.arange(kernel_size).float()
        coords -= (kernel_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = torch.outer(g, g)
        self.register_buffer("kernel", kernel_2d[None, None])

    def forward(self, x, y):
        C1 = (self.k1 ** 2)
        C2 = (self.k2 ** 2)

        kernel = self.kernel.to(x.device)
        ch = x.shape[1]
        kernel = kernel.expand(ch, 1, self.kernel_size, self.kernel_size)

        mu_x = F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=ch)
        mu_y = F.conv2d(y, kernel, padding=self.kernel_size // 2, groups=ch)

        sigma_x = F.conv2d(x * x, kernel, padding=self.kernel_size // 2, groups=ch) - mu_x ** 2
        sigma_y = F.conv2d(y * y, kernel, padding=self.kernel_size // 2, groups=ch) - mu_y ** 2
        sigma_xy = F.conv2d(x * y, kernel, padding=self.kernel_size // 2, groups=ch) - mu_x * mu_y

        ssim_num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        ssim = ssim_num / (ssim_den + 1e-8)
        dssim = (1 - ssim) / 2
        return dssim.mean()


# ============================================================
#  eyes-mouth-prio（ランドマーク重み付け）
# ============================================================

def make_eye_mouth_weight_map(landmarks, H, W, device):
    """
    landmarks: (68,2) numpy or tensor
    出力: (1,1,H,W) の weight map
    """

    if isinstance(landmarks, torch.Tensor):
        lmrk = landmarks.detach().cpu().numpy()
    else:
        lmrk = landmarks

    weight = torch.ones((1, 1, H, W), dtype=torch.float32, device=device)

    def draw_poly(idx_range, w):
        pts = lmrk[idx_range[0]:idx_range[1]]
        pts = torch.tensor(pts, dtype=torch.float32, device=device)
        mask = torch.zeros((1, 1, H, W), device=device)
        yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device),
                                indexing="ij")
        xx = xx.unsqueeze(0).unsqueeze(0)
        yy = yy.unsqueeze(0).unsqueeze(0)

        # 多角形内判定（簡易版）
        x = pts[:, 0]
        y = pts[:, 1]
        poly = torch.stack([x, y], dim=1)

        # バウンディングボックスで高速化
        xmin, xmax = int(x.min()), int(x.max())
        ymin, ymax = int(y.min()), int(y.max())

        if xmin < 0 or ymin < 0 or xmax >= W or ymax >= H:
            return

        # ポリゴン内点を塗る（ray casting）
        sub_x = xx[:, :, ymin:ymax, xmin:xmax]
        sub_y = yy[:, :, ymin:ymax, xmin:xmax]

        inside = torch.zeros_like(sub_x, dtype=torch.bool)
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            cond = ((y1 > sub_y) != (y2 > sub_y)) & \
                   (sub_x < (x2 - x1) * (sub_y - y1) / (y2 - y1 + 1e-6) + x1)
            inside ^= cond

        mask[:, :, ymin:ymax, xmin:xmax] = inside.float()
        weight[:] += mask * w

    # 目（強め）
    draw_poly((36, 42), w=3.0)
    draw_poly((42, 48), w=3.0)

    # 口（中程度）
    draw_poly((48, 60), w=2.0)

    return weight


# ============================================================
#  SAEHD-light loss（DSSIM + L1 + eyes-mouth-prio）
# ============================================================

class SAEHDLightLoss(nn.Module):
    def __init__(self, resolution=128):
        super().__init__()
        self.dssim = DSSIM(kernel_size=int(resolution / 11.6))

    def forward(self, pred, target, landmarks):
        """
        pred: (N,3,H,W)
        target: (N,3,H,W)
        landmarks: list of 68-point numpy arrays (len=N)
        """

        N, C, H, W = pred.shape
        device = pred.device

        total_loss = 0.0

        for i in range(N):
            p = pred[i:i+1]
            t = target[i:i+1]

            # DSSIM
            loss_dssim = self.dssim(p, t)

            # L1
            loss_l1 = F.l1_loss(p, t)

            # eyes-mouth-prio
            weight = make_eye_mouth_weight_map(landmarks[i], H, W, device)
            loss_weighted = (weight * torch.abs(p - t)).mean()

            # 合計（SAEHD の重み付けに近い）
            loss = loss_dssim * 10.0 + loss_l1 * 10.0 + loss_weighted * 5.0
            total_loss += loss

        return total_loss / N
