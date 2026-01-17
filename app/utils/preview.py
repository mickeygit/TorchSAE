import os
import cv2
import numpy as np
import torch
import torchvision.utils as vutils


def tensor_to_image(t):
    """
    Tensor (N,C,H,W) → uint8 image (H,W,3)
    [-1,1] or [0,1] どちらでも対応
    """
    t = t.detach().cpu()
    if t.min() < 0:
        t = (t + 1) / 2
    t = torch.clamp(t, 0, 1)
    img = (t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    heatmap: (H,W) 0〜1
    """
    hm = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1.0, hm_color, alpha, 0)


def save_liae_preview_with_masks(
    step,
    aa, bb, ab, ba,
    a_orig, b_orig,
    out_dir,
    ext="jpg",
    mask_a=None,
    mask_b=None,
    heatmap_a=None,
    heatmap_b=None,
    loss_value=None,
):
    """
    DeepFaceLab-style preview:
        A_orig | A→A | A→B | mask | blend
        B_orig | B→B | B→A | mask | blend
    """

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Convert tensors to images
    # -----------------------------
    A_orig = tensor_to_image(a_orig)
    B_orig = tensor_to_image(b_orig)

    AA = tensor_to_image(aa)
    BB = tensor_to_image(bb)
    AB = tensor_to_image(ab)
    BA = tensor_to_image(ba)

    H, W, _ = A_orig.shape

    # -----------------------------
    # Mask (if provided)
    # -----------------------------
    def mask_to_img(mask):
        if mask is None:
            return np.zeros((H, W), np.uint8)
        m = mask.detach().cpu()[0, 0].numpy()
        m = np.clip(m, 0, 1)
        return (m * 255).astype(np.uint8)

    mask_A = mask_to_img(mask_a)
    mask_B = mask_to_img(mask_b)

    # -----------------------------
    # Blend (DFL-style)
    # -----------------------------
    def blend(src, dst, mask):
        mask_f = mask.astype(np.float32) / 255.0
        mask_f = cv2.GaussianBlur(mask_f, (21, 21), 0)
        mask_f = np.clip(mask_f, 0, 1)
        return (src * mask_f[..., None] + dst * (1 - mask_f[..., None])).astype(np.uint8)

    blend_A = blend(AB, A_orig, mask_A)
    blend_B = blend(BA, B_orig, mask_B)

    # -----------------------------
    # Heatmap overlay (optional)
    # -----------------------------
    if heatmap_a is not None:
        hm_a = heatmap_a.detach().cpu()[0, 0].numpy()
        A_orig = overlay_heatmap(A_orig, hm_a)

    if heatmap_b is not None:
        hm_b = heatmap_b.detach().cpu()[0, 0].numpy()
        B_orig = overlay_heatmap(B_orig, hm_b)

    # -----------------------------
    # Build preview grid
    # -----------------------------
    row_A = np.hstack([
        A_orig,
        AA,
        AB,
        cv2.cvtColor(mask_A, cv2.COLOR_GRAY2BGR),
        blend_A,
    ])

    row_B = np.hstack([
        B_orig,
        BB,
        BA,
        cv2.cvtColor(mask_B, cv2.COLOR_GRAY2BGR),
        blend_B,
    ])

    preview = np.vstack([row_A, row_B])

    # -----------------------------
    # Draw text (step / loss)
    # -----------------------------
    text = f"step: {step}"
    if loss_value is not None:
        text += f"  loss: {loss_value:.4f}"

    cv2.putText(
        preview, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # -----------------------------
    # Save (2× upscale for visibility)
    # -----------------------------
    preview_big = cv2.resize(preview, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    out_path = os.path.join(out_dir, f"preview_{step:06d}.{ext}")
    cv2.imwrite(out_path, preview_big)

def mask_overlay_bgr(img, mask, color=(0,1,0), alpha=0.5):
    """
    img:  (B,3,H,W) float32 0-1
    mask: (B,1,H,W) float32 0-1
    """
    mask_bin = (mask > 0.5).float()
    color_t = torch.tensor(color, dtype=img.dtype, device=img.device).view(1,3,1,1)
    overlay = img * (1 - alpha * mask_bin) + color_t * (alpha * mask_bin)
    return overlay

def make_saehd_style_preview(img_a, img_b, aa, bb, ab, mask_b):
    """
    本家 SAEHD と同じ並び:
    [src] [dst]
    [src→src] [dst→dst]
    [src→dst] [src→dst + mask overlay]
    """
    ab_masked = mask_overlay_bgr(ab, mask_b, color=(0,1,0), alpha=0.5)

    row1 = torch.cat([img_a, img_b], dim=3)
    row2 = torch.cat([aa, bb], dim=3)
    row3 = torch.cat([ab, ab_masked], dim=3)

    preview = torch.cat([row1, row2, row3], dim=2)
    return preview
