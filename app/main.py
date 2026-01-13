import argparse
from app.config import TrainConfig
from app.trainer import Trainer


def build_parser():
    parser = argparse.ArgumentParser(description="TorchSAE Training")

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    parser.add_argument("--data-dir-a", type=str, required=True,
                        help="Directory of face set A")
    parser.add_argument("--data-dir-b", type=str, required=True,
                        help="Directory of face set B")

    # ---------------------------------------------------------
    # Model structure (DF 優先、本家準拠)
    # ---------------------------------------------------------
    parser.add_argument("--archi", type=str, default="df",
                        choices=["df"],
                        help="Model architecture (DF first. LIAE later)")

    parser.add_argument("--model-size", type=int, default=128,
                        choices=[128, 256, 384, 512],
                        help="Base resolution")

    parser.add_argument("--face-type", type=str, default="full",
                        choices=["full", "head", "wf"],
                        help="Face type")

    parser.add_argument("--ae-dims", type=int, default=256)
    parser.add_argument("--e-dims", type=int, default=64)
    parser.add_argument("--d-dims", type=int, default=64)
    parser.add_argument("--d-mask-dims", type=int, default=32)

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200000)

    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw"])

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip-grad", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=1234)

    # ---------------------------------------------------------
    # Augmentation
    # ---------------------------------------------------------
    parser.add_argument("--random-warp", action="store_true")
    parser.add_argument("--random-hsv-power", type=float, default=0.0)
    parser.add_argument("--random-noise-power", type=float, default=0.0)

    # ---------------------------------------------------------
    # Preview / Save
    # ---------------------------------------------------------
    parser.add_argument("--preview-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)

    # ---------------------------------------------------------
    # AMP
    # ---------------------------------------------------------
    parser.add_argument("--amp", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Config object
    cfg = TrainConfig(
        data_dir_a=args.data_dir_a,
        data_dir_b=args.data_dir_b,

        archi=args.archi,
        model_size=args.model_size,
        face_type=args.face_type,

        ae_dims=args.ae_dims,
        e_dims=args.e_dims,
        d_dims=args.d_dims,
        d_mask_dims=args.d_mask_dims,

        batch_size=args.batch_size,
        max_steps=args.max_steps,

        optimizer=args.optimizer,
        lr=args.lr,
        clip_grad=args.clip_grad,
        seed=args.seed,

        random_warp=args.random_warp,
        random_hsv_power=args.random_hsv_power,
        random_noise_power=args.random_noise_power,

        preview_interval=args.preview_interval,
        save_interval=args.save_interval,

        amp=args.amp,
    )

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
