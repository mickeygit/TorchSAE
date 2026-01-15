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
