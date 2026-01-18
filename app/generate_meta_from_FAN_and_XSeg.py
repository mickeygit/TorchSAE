import os
import sys
import urllib.request
import numpy as np
import cv2
import face_alignment
import onnxruntime as ort

from app.utils.DFLJPG import DFLJPG


# ---------------------------------------------------------
#  パス設定
# ---------------------------------------------------------
DEFAULT_XSEG = "/workspace/xseg/XSeg_model_WF 5.0 model-20240130T133752Z-001.onnx"

FAN_MODEL_PATH = "/workspace/app/models/fan/2DFAN-4.pth.tar"
FAN_MODEL_URL = "https://www.adrianbulat.com/downloads/python-fan/2DFAN-4.pth.tar"

XSEG_INPUT_SIZE = 256


# ---------------------------------------------------------
#  FAN モデルを自動ダウンロード（無ければ）
# ---------------------------------------------------------
def ensure_fan_model():
    os.makedirs(os.path.dirname(FAN_MODEL_PATH), exist_ok=True)

    if os.path.exists(FAN_MODEL_PATH):
        print(f"[FA] FAN model already exists → {FAN_MODEL_PATH}")
        return

    print(f"[FA] FAN model not found. Downloading...")
    try:
        urllib.request.urlretrieve(FAN_MODEL_URL, FAN_MODEL_PATH)
        print(f"[FA] Downloaded FAN model → {FAN_MODEL_PATH}")
    except Exception as e:
        print(f"[FA] ERROR: Failed to download FAN model: {e}")
        raise


# ---------------------------------------------------------
#  FAN のモデルパスを強制的に固定（ダウンロード防止）
# ---------------------------------------------------------
def patch_fa_model_path(local_path):
    import face_alignment.utils as utils

    def _patched_get_model_path(*args, **kwargs):
        return local_path

    utils.get_model_path = _patched_get_model_path
    print(f"[FA] Patched FAN model path → {local_path}")


# ---------------------------------------------------------
#  face-alignment FAN をロード
# ---------------------------------------------------------
def load_fa_model():
    import torch

    print("[FA] Loading face-alignment FAN (2D 68 landmarks)")

    # ① モデルが無ければ自動ダウンロード
    ensure_fan_model()

    # ② face-alignment にローカルモデルを強制使用させる
    patch_fa_model_path(FAN_MODEL_PATH)

    # ③ FAN をロード
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=False,
        device=device
    )
    return fa


# ---------------------------------------------------------
#  XSeg ONNX モデルをロード
# ---------------------------------------------------------
def load_xseg_model(onnx_path):
    print(f"[XSeg] Loading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


# ---------------------------------------------------------
#  XSeg 推論（BGR → 0〜1 mask）
# ---------------------------------------------------------
def run_xseg(sess, input_name, output_name, img_bgr):
    img = cv2.resize(img_bgr, (XSEG_INPUT_SIZE, XSEG_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # NHWC のまま 0〜1 正規化
    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # (1,256,256,3)

    # ONNX 推論
    y = sess.run([output_name], {input_name: x})[0]

    mask = y[0]
    if mask.ndim == 2:
        mask = mask[..., None]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        pass
    else:
        raise ValueError("Unexpected XSeg output shape")

    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    return mask

# ---------------------------------------------------------
#  faceset を処理（landmarks + XSeg を DFLJPG に埋め込む・常に上書き）
# ---------------------------------------------------------
def process_faceset(root_dir, fa, xseg_sess, xseg_input_name, xseg_output_name):
    print(f"[META] Processing directory: {root_dir}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(exts)])

    print(f"[META] Found {len(files)} images")

    for f in files:
        img_path = os.path.join(root_dir, f)

        jpg = DFLJPG.load(img_path)
        if jpg is None:
            print(f"[Error] Cannot load DFLJPG: {img_path}")
            continue

        img_bgr = jpg.get_img()
        if img_bgr is None:
            print(f"[Error] Cannot read: {img_path}")
            continue

        # -----------------------------
        # 1) FAN landmarks
        # -----------------------------
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            lms_list = fa.get_landmarks(img_rgb)
        except Exception as e:
            print(f"[Error] FAN failed on {img_path}: {e}")
            continue

        if not lms_list:
            print(f"[Warn] No face detected: {img_path}")
            continue

        lms = np.array(lms_list[0], dtype=np.float32)  # (68,2)

        # -----------------------------
        # 2) XSeg mask
        # -----------------------------
        try:
            mask = run_xseg(xseg_sess, xseg_input_name, xseg_output_name, img_bgr)
        except Exception as e:
            print(f"[Error] XSeg failed on {img_path}: {e}")
            continue

        # -----------------------------
        # 3) DFLJPG に上書き保存
        # -----------------------------
        jpg.set_landmarks(lms)
        jpg.set_xseg_mask(mask)
        jpg.save()  # ← これで APP15 に上書き保存される




        print(f"[OK] Overwritten landmarks + XSeg → {img_path}")

    print("[META] Done.")


# ---------------------------------------------------------
#  エントリーポイント
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_meta_from_FAN_and_XSeg.py <faces_dir>")
        sys.exit(1)

    faces_dir = sys.argv[1]

    fa = load_fa_model()
    xseg_sess, xseg_input_name, xseg_output_name = load_xseg_model(DEFAULT_XSEG)

    process_faceset(faces_dir, fa, xseg_sess, xseg_input_name, xseg_output_name)
