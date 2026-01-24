import sys
import json
from app.config import TrainConfig

# 分割した Trainer を読み込む
from app.trainers.trainer_df import TrainerDF
from app.trainers.trainer_liae import TrainerLIAE


def load_train_config(path: str) -> TrainConfig:
    """Load TrainConfig from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return TrainConfig(**data)


def main():
    # ----------------------------------------
    # Parse config path
    # ----------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python main.py <config.json>")
        return

    cfg_path = sys.argv[1]
    print(f"[Main] Loading config: {cfg_path}")

    # ----------------------------------------
    # Load config
    # ----------------------------------------
    cfg = load_train_config(cfg_path)
    cfg.summary()

    # ----------------------------------------
    # Select trainer based on model_type
    # ----------------------------------------
    model_type = cfg.model_type.lower()

    if model_type == "df":
        trainer = TrainerDF(cfg)

    elif model_type in ["liae", "liae_ud_256"]:
        # ★ LIAE と LIAE_UD_256 は同じ TrainerLIAE を使う
        trainer = TrainerLIAE(cfg)

    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    # ----------------------------------------
    # Start training
    # ----------------------------------------
    trainer.run()


if __name__ == "__main__":
    main()
