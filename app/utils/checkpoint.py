import os
import torch


def save_checkpoint(model, step, out_dir="checkpoints"):
    """
    Save encoder / decoder_A / decoder_B separately.
    """
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "encoder": os.path.join(out_dir, f"encoder_{step}.pt"),
        "decoder_a": os.path.join(out_dir, f"decoder_a_{step}.pt"),
        "decoder_b": os.path.join(out_dir, f"decoder_b_{step}.pt"),
    }

    torch.save(model.encoder.state_dict(), paths["encoder"])
    torch.save(model.decoder_a.state_dict(), paths["decoder_a"])
    torch.save(model.decoder_b.state_dict(), paths["decoder_b"])

    print(f"[Checkpoint] Saved encoder/decoder at step {step}")

    return paths
