import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

from merge_utils import (
    color_transfer_dfl,
    blur_mask,
    erode_mask
)

# ============================================================
# XSeg マスク生成
# ============================================================
def create_xseg_session(xseg_path):
    return ort.InferenceSession(
        xseg_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

def run_xseg_mask(session, img, mask_size=256):
    h, w = img.shape[:2]

    x = cv2.resize(img, (mask_size, mask_size))
    x = x.astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)  # NHWC

    input_name = session.get_inputs()[0].name
    mask = session.run(None, {input_name: x})[0]

    mask = mask.squeeze()
    mask = cv2.resize(mask, (w, h))
    return np.clip(mask, 0, 1)


# ============================================================
# ONNX 入力用画像変換
# ============================================================
def load_image_for_onnx_from_array(img, model_size=128):
    img = cv2.resize(img, (model_size, model_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)
    return img


# ============================================================
# JPG 保存
# ============================================================
def save_jpg_rgb(img, path):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# ============================================================
# ONNX Runtime セッション作成
# ============================================================
def create_session(onnx_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    used_provider = session.get_providers()[0]
    gpu_used = (used_provider == "CUDAExecutionProvider")

    print("[INFO] Execution Providers")
    print(f"  Available : {providers}")
    print(f"  Used      : {used_provider}")
    print(f"  GPU Used  : {'YES' if gpu_used else 'NO'}")

    return session, gpu_used


# ============================================================
# 最適化マージ（LIAE / DF 共通）
# ============================================================
def merge_faces_optimized(original, converted, mask):
    # 1. マスクのエッジ処理（erode → blur）
    mask_eroded = erode_mask(mask, ksize=5)
    mask_blur = blur_mask(mask_eroded, ksize=21)

    # 2. 色補正は “ぼかしたマスク” で行う
    mask_soft = blur_mask(mask, ksize=31)
    converted_adj = color_transfer_dfl(converted, original, mask_soft)

    # 3. 軽いガンマ補正で顔の浮きを防ぐ
    converted_adj = np.power(converted_adj, 1.05)

    # 4. マージ（背景は original を100%使う）
    merged = original * (1 - mask_blur[..., None]) + converted_adj * mask_blur[..., None]
    return merged


# ============================================================
# A→B 推論（最適化版）
# ============================================================
def run_inference_AtoB(onnx_path, xseg_path, folderA, out_folder, model_size=128):
    session, gpu_used = create_session(onnx_path)
    xseg_session = create_xseg_session(xseg_path)

    filesA = sorted([f for f in os.listdir(folderA) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not filesA:
        print(f"[ERROR] No images found in folder A: {folderA}")
        sys.exit(1)

    os.makedirs(out_folder, exist_ok=True)

    print("\n=== Input Parameters ===")
    print(f"ONNX Model       : {onnx_path}")
    print(f"XSeg Model       : {xseg_path}")
    print(f"Input Folder A   : {folderA}  ({len(filesA)} files)")
    print(f"Output Folder    : {out_folder}")
    print(f"ONNX Input Size  : {model_size}×{model_size}")
    print(f"GPU Used         : {'YES' if gpu_used else 'NO'}")
    print("========================\n")

    start_total = time.time()

    for idx, fnameA in enumerate(filesA):
        pathA = os.path.join(folderA, fnameA)
        original = cv2.cvtColor(cv2.imread(pathA), cv2.COLOR_BGR2RGB)

        # ① XSeg マスク生成
        mask = run_xseg_mask(xseg_session, original)

        # ② 推論入力は “マスク無しの A”
        imgA = load_image_for_onnx_from_array(original, model_size)

        # ③ TorchSAE（DF / LIAE）ONNX 推論
        inputs = {"input_a": imgA, "input_b": imgA}
        _, _, out_ab, _ = session.run(None, inputs)

        # ④ 出力を元解像度に戻す
        out_ab = out_ab.squeeze(0).transpose(1, 2, 0)
        out_ab = cv2.resize(out_ab, (original.shape[1], original.shape[0]))
        out_ab = np.clip(out_ab, 0, 1)

        # ⑤ 最適化マージ
        final = merge_faces_optimized(original / 255.0, out_ab, mask)

        # ⑥ 保存
        baseA = os.path.splitext(fnameA)[0]
        out_path = os.path.join(out_folder, f"{baseA}_AtoB.jpg")
        save_jpg_rgb(final, out_path)

        print(f"[{idx+1}/{len(filesA)}] {fnameA} → {out_path}")

    end_total = time.time()

    print("\n=== A→B Inference Summary ===")
    print(f"Processed Images : {len(filesA)}")
    print(f"Total Time       : {end_total - start_total:.3f} sec")
    print(f"Time per Image   : {(end_total - start_total)/len(filesA):.3f} sec")
    print(f"GPU Used         : {'YES' if gpu_used else 'NO'}")
    print("===============================")


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 onnx_infer_AtoB.py model.onnx xseg.onnx folderA output_folder [model_size]")
        sys.exit(1)

    onnx_path = sys.argv[1]
    xseg_path = sys.argv[2]
    folderA = sys.argv[3]
    out_folder = sys.argv[4]
    model_size = int(sys.argv[5]) if len(sys.argv) >= 6 else 128

    run_inference_AtoB(onnx_path, xseg_path, folderA, out_folder, model_size)


if __name__ == "__main__":
    main()
