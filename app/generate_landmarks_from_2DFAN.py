import os
import sys
import numpy as np
import cv2
import face_alignment


# ---------------------------------------------------------
#  face-alignment FAN をロード
# ---------------------------------------------------------
def load_fa_model():
    import torch
    print("[FA] Loading face-alignment FAN (2D 68 landmarks)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=False,
        device=device
    )
    return fa

# ---------------------------------------------------------
#  faceset を処理
# ---------------------------------------------------------
def process_faceset(root_dir, fa):
    print(f"[FA] Processing directory: {root_dir}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(root_dir) if f.lower().endswith(exts)]

    print(f"[FA] Found {len(files)} images")

    for f in files:
        img_path = os.path.join(root_dir, f)
        out_path = os.path.join(root_dir, os.path.splitext(f)[0] + "_landmarks.npy")

        if os.path.exists(out_path):
            print(f"[Skip] {out_path} already exists")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[Error] Cannot read: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            lms_list = fa.get_landmarks(img_rgb)
        except Exception as e:
            print(f"[Error] face-alignment failed on {img_path}: {e}")
            continue

        if not lms_list:
            print(f"[Warn] No face detected: {img_path}")
            continue

        # 最初の顔だけ採用
        lms = np.array(lms_list[0], dtype=np.float32)  # shape: (68, 2)

        np.save(out_path, lms)
        print(f"[OK] Saved: {out_path}")

    print("[FA] Done.")


# ---------------------------------------------------------
#  エントリーポイント
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_landmarks_face_alignment.py <faces_dir>")
        sys.exit(1)

    faces_dir = sys.argv[1]

    fa = load_fa_model()
    process_faceset(faces_dir, fa)
