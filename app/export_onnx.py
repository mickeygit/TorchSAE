import json
import torch
from app.config import TrainConfig
from app.models.autoencoder_df import DFModel
from app.models.autoencoder_liae import LIAEModel


def build_model(cfg):
    """DF / LIAE を model_type で切り替える"""
    mt = cfg.model_type.lower()
    if mt == "df":
        return DFModel(cfg)
    elif mt == "liae":
        return LIAEModel(cfg)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def export_onnx(cfg_path="config.json",
                ckpt_path=None,
                out_path=None):

    # ---------------------------------------------------------
    # Load config
    # ---------------------------------------------------------
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = TrainConfig(**cfg_dict)


    # 出力ファイル名が未指定なら model_type で決める
    if not out_path:
        out_path = f"{cfg.model_type.lower()}model.onnx"

    # ---------------------------------------------------------
    # Build CPU model
    # ---------------------------------------------------------
    model = build_model(cfg)

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------
    if ckpt_path is not None:
        print(f"[ONNX] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")

        if "model" in state:
            print("[ONNX] Detected resume checkpoint format")
            model.load_state_dict(state["model"], strict=False)
        else:
            print("[ONNX] Detected model-only checkpoint format")
            model.load_state_dict(state, strict=False)
    else:
        print("[ONNX] No checkpoint specified. Using untrained weights.")

    # ---------------------------------------------------------
    # GPU forward test
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        print("[GPU] Running forward test on GPU...")

        gpu_model = build_model(cfg).cuda()
        gpu_model.load_state_dict(model.state_dict(), strict=False)
        gpu_model.eval()

        dummy_a_gpu = torch.randn(1, 3, cfg.model_size, cfg.model_size).cuda()
        dummy_b_gpu = torch.randn(1, 3, cfg.model_size, cfg.model_size).cuda()

        with torch.no_grad():
            out = gpu_model(dummy_a_gpu, dummy_b_gpu)

        print("[GPU] Forward test OK.")
    else:
        print("[GPU] CUDA not available. Skipping GPU test.")

    # ---------------------------------------------------------
    # ONNX export
    # ---------------------------------------------------------
    print(f"[ONNX] Exporting {cfg.model_type} model → {out_path}")

    model.eval()
    dummy_a = torch.randn(1, 3, cfg.model_size, cfg.model_size)
    dummy_b = torch.randn(1, 3, cfg.model_size, cfg.model_size)

    try:
        torch.onnx.export(
            model,
            (dummy_a, dummy_b),
            out_path,
            input_names=["input_a", "input_b"],
            output_names=["out_aa", "out_bb", "out_ab", "out_ba"],
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
            dynamic_axes={
                "input_a": {0: "batch"},
                "input_b": {0: "batch"},
                "out_aa": {0: "batch"},
                "out_bb": {0: "batch"},
                "out_ab": {0: "batch"},
                "out_ba": {0: "batch"},
            }
        )
    except Exception as e:
        print("=== ONNX EXPORT ERROR ===")
        print(e)
        raise

    print(f"[ONNX] Exported to {out_path}")


if __name__ == "__main__":
    import sys
    cfg  = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    out  = sys.argv[3] if len(sys.argv) > 3 else None

    export_onnx(cfg_path=cfg, ckpt_path=ckpt, out_path=out)
