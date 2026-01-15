import sys
import json
from app.config import TrainConfig
from app.trainer import Trainer


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
    # Start training
    # ----------------------------------------
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
