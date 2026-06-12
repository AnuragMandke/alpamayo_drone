"""
scripts/train.py — Entry point for training Alpamayo-Drone

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --wandb  # enable W&B logging
    python scripts/train.py --config configs/default.yaml --resume outputs/ckpt_epoch010
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint dir to resume from")
    p.add_argument("--device", type=str, default=None,
                   help="Force device (e.g. 'cuda:0', 'cpu')")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["training"]["seed"])

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Train] Device: {device}")

    # ------------------------------------------------------------------
    # Optional: W&B
    # ------------------------------------------------------------------
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="alpamayo-drone",
                config=cfg,
                name=f"lora_r{cfg['model']['lora']['rank']}",
            )
            print("[Train] W&B logging enabled")
        except ImportError:
            print("[Train] wandb not installed, skipping")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from models.alpamayo import AlpamayoDrone
    from models.lora import load_lora_weights

    model = AlpamayoDrone(cfg)
    model.to(device)

    # Resume from checkpoint if requested
    if args.resume:
        ckpt_dir = Path(args.resume)
        print(f"[Train] Resuming from {ckpt_dir}")
        load_lora_weights(model, str(ckpt_dir / "lora_weights.pt"))
        model.decoder.load_state_dict(
            torch.load(ckpt_dir / "decoder.pt", map_location=device)
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    from data.airsim_dataset import build_dataloaders
    train_loader, val_loader = build_dataloaders(cfg, model.tokenizer)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    from training.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.train()

    print("[Train] Done.")


if __name__ == "__main__":
    main()
