import cv2
import numpy as np

def color_transfer_dfl(src, dst, mask=None):
    """
    DFL の color transfer と同じアルゴリズム。
    src = 変換後の顔（B側）
    dst = 元画像（A側）
    mask = マスク（0〜1）
    """

    # マスクがない場合は全領域
    if mask is None:
        mask = np.ones(src.shape[:2], dtype=np.float32)

    mask3 = np.expand_dims(mask, 2)

    # マスク領域を抽出
    src_masked = src * mask3
    dst_masked = dst * mask3

    # 平均と標準偏差を計算
    src_mean = np.sum(src_masked, axis=(0,1)) / np.sum(mask)
    dst_mean = np.sum(dst_masked, axis=(0,1)) / np.sum(mask)

    src_std = np.sqrt(np.sum(((src_masked - src_mean)**2) * mask3, axis=(0,1)) / np.sum(mask))
    dst_std = np.sqrt(np.sum(((dst_masked - dst_mean)**2) * mask3, axis=(0,1)) / np.sum(mask))

    # 標準偏差がゼロのチャンネルは無視
    src_std = np.where(src_std < 1e-6, 1.0, src_std)
    dst_std = np.where(dst_std < 1e-6, 1.0, dst_std)

    # DFL と同じ正規化処理
    result = (src - src_mean) * (dst_std / src_std) + dst_mean

    # 0〜1 にクリップ
    result = np.clip(result, 0, 1)

    return result
