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

    # XSeg 入力サイズへリサイズ
    x = cv2.resize(img, (mask_size, mask_size))
    x = x.astype(np.float32) / 255.0

    # ★ NHWC のまま
    x = np.expand_dims(x, 0)  # (1, H, W, 3)

    # ★ XSeg の入力名を自動取得
    input_name = session.get_inputs()[0].name

    # 推論
    mask = session.run(None, {input_name: x})[0]

    # (1, H, W, 1) → (H, W)
    mask = mask.squeeze()
    mask = cv2.resize(mask, (w, h))
    mask = np.clip(mask, 0, 1)
    return mask


# ============================================================
# ONNX 入力用画像変換
# ============================================================
def load_image_for_onnx_from_array(img, model_size=128):
    img = cv2.resize(img, (model_size, model_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
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
# DFL 風マージ
# ============================================================
def merge_faces(original, converted, mask, mode="color"):
    if mode == "raw":
        return original * (1 - mask[...,None]) + converted * mask[...,None]

    if mode == "alpha":
        mask_blur = blur_mask(mask)
        return original * (1 - mask_blur[...,None]) + converted * mask_blur[...,None]

    if mode == "erode":
        mask_eroded = erode_mask(mask)
        mask_blur = blur_mask(mask_eroded)
        return original * (1 - mask_blur[...,None]) + converted * mask_blur[...,None]

    if mode == "color":
        converted_adj = color_transfer_dfl(converted, original, mask)
        mask_blur = blur_mask(mask)
        return original * (1 - mask_blur[...,None]) + converted_adj * mask_blur[...,None]

    raise ValueError("Unknown merge mode")


# ============================================================
# A→B 推論（XSeg + TorchSAE DF/LIAE ONNX + DFL Merge）
# ============================================================
def run_inference_AtoB(onnx_path, xseg_path, folderA, out_folder, model_size=128, merge_mode="color"):
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
    print(f"Merge Mode       : {merge_mode}")
    print("Available Merge Modes : raw, alpha, erode, color")
    print(f"ONNX Input Size  : {model_size}×{model_size}")
    print(f"GPU Used         : {'YES' if gpu_used else 'NO'}")
    print("  ※ TorchSAE DF / LIAE ONNX は同一 IO 仕様 (input_a,input_b → out_ab)")
    print("========================\n")

    start_total = time.time()

    for idx, fnameA in enumerate(filesA):
        pathA = os.path.join(folderA, fnameA)

        original = cv2.cvtColor(cv2.imread(pathA), cv2.COLOR_BGR2RGB)

        # ① XSeg マスク生成
        mask = run_xseg_mask(xseg_session, original)

        # ② TorchSAE 入力用にマスク済み A を作成
        masked_A = original * mask[...,None]
        imgA = load_image_for_onnx_from_array(masked_A, model_size)

        # ③ TorchSAE（DF / LIAE）ONNX 推論
        inputs = {"input_a": imgA, "input_b": imgA}
        _, _, out_ab, _ = session.run(None, inputs)

        # ④ 出力を元解像度に戻す
        out_ab = out_ab.squeeze(0).transpose(1,2,0)
        out_ab = cv2.resize(out_ab, (original.shape[1], original.shape[0]))
        out_ab = np.clip(out_ab, 0, 1)

        # ⑤ DFL 風マージ
        final = merge_faces(original/255.0, out_ab, mask, mode=merge_mode)

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
    print(f"Merge Mode Used  : {merge_mode}")
    print("  ※ TorchSAE DF / LIAE ONNX 共通処理")
    print("===============================")


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 onnx_infer_AtoB.py model.onnx xseg.onnx folderA output_folder [model_size] [merge_mode]")
        print("  model.onnx : TorchSAE DF / LIAE どちらでも可（同一 IO 仕様）")
        sys.exit(1)

    onnx_path = sys.argv[1]
    xseg_path = sys.argv[2]
    folderA = sys.argv[3]
    out_folder = sys.argv[4]
    model_size = int(sys.argv[5]) if len(sys.argv) >= 6 else 128
    merge_mode = sys.argv[6] if len(sys.argv) >= 7 else "color"

    run_inference_AtoB(onnx_path, xseg_path, folderA, out_folder, model_size, merge_mode)

if __name__ == "__main__":
    main()
