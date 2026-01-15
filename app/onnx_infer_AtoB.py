import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# 画像読み込み（元画像の解像度は任意）
# ONNX 入力は model_size にリサイズ
# ============================================================
def load_image_for_onnx(path, model_size=128):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (model_size, model_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, 0)        # CHW → NCHW
    return img


# ============================================================
# JPG 保存
# ============================================================
def save_jpg(tensor, path):
    img = tensor.squeeze(0)
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# ============================================================
# ONNX Runtime セッション作成（CUDA → CPU fallback）
# ============================================================
def create_session(onnx_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(
        onnx_path,
        providers=providers
    )

    used_provider = session.get_providers()[0]
    gpu_used = (used_provider == "CUDAExecutionProvider")

    print("[INFO] Execution Providers")
    print(f"  Available : {providers}")
    print(f"  Used      : {used_provider}")
    print(f"  GPU Used  : {'YES' if gpu_used else 'NO'}")

    return session, gpu_used


# ============================================================
# A→B 推論（A フォルダ → B フォルダへ保存）
# ============================================================
def run_inference_AtoB(onnx_path, folderA, out_folder, model_size=128):
    session, gpu_used = create_session(onnx_path)

    # A の画像一覧
    filesA = sorted([f for f in os.listdir(folderA) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not filesA:
        print(f"[ERROR] No images found in folder A: {folderA}")
        sys.exit(1)

    # 出力フォルダ作成
    os.makedirs(out_folder, exist_ok=True)

    print("\n=== Input Parameters ===")
    print(f"ONNX Model       : {onnx_path}")
    print(f"Input Folder A   : {folderA}  ({len(filesA)} files)")
    print(f"Output Folder    : {out_folder}")
    print(f"ONNX Input Size  : {model_size}×{model_size}")
    print(f"GPU Used         : {'YES' if gpu_used else 'NO'}")
    print("========================\n")

    start_total = time.time()

    for idx, fnameA in enumerate(filesA):
        pathA = os.path.join(folderA, fnameA)

        # A の画像を読み込み → ONNX 入力サイズへ縮小
        imgA = load_image_for_onnx(pathA, model_size)

        # input_b は A と同じ画像で OK（DFModel の仕様）
        inputs = {
            "input_a": imgA,
            "input_b": imgA,
        }

        # A→B = out_ab
        _, _, out_ab, _ = session.run(None, inputs)

        baseA = os.path.splitext(fnameA)[0]
        out_path = os.path.join(out_folder, f"{baseA}_to_B.jpg")

        save_jpg(out_ab, out_path)

        print(f"[{idx+1}/{len(filesA)}] {fnameA} → {out_path}")

    end_total = time.time()

    total_time = end_total - start_total
    per_image = total_time / len(filesA)

    print("\n=== A→B Inference Summary ===")
    print(f"Processed Images : {len(filesA)}")
    print(f"Total Time       : {total_time:.3f} sec")
    print(f"Time per Image   : {per_image:.3f} sec")
    print(f"GPU Used         : {'YES' if gpu_used else 'NO'}")
    print("===============================")


# ============================================================
# main
# ============================================================
def main():
    if len(sys.argv) < 4:
        print("Usage: python3 onnx_infer_AtoB.py model.onnx folderA output_folder [model_size]")
        sys.exit(1)

    onnx_path = sys.argv[1]
    folderA = sys.argv[2]
    out_folder = sys.argv[3]
    model_size = int(sys.argv[4]) if len(sys.argv) >= 5 else 128

    run_inference_AtoB(onnx_path, folderA, out_folder, model_size)


if __name__ == "__main__":
    main()
