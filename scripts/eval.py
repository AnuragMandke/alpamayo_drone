"""
scripts/eval.py — Entry point for evaluation

Two modes:
    --mode offline   Compute L2 / ADE / FDE on val split (no AirSim needed)
    --mode online    Run live episodes in AirSim simulator

Usage:
    python scripts/eval.py --config configs/default.yaml --ckpt outputs/ckpt_epoch030 --mode offline
    python scripts/eval.py --config configs/default.yaml --ckpt outputs/ckpt_epoch030 --mode online
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import yaml
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/default.yaml")
    p.add_argument("--ckpt",    required=True, help="Checkpoint directory")
    p.add_argument("--mode",    choices=["offline", "online"], default="offline")
    p.add_argument("--device",  default=None)
    p.add_argument("--out",     default="outputs/eval_results.json",
                   help="Where to write results JSON")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Eval] Device: {device}  |  Mode: {args.mode}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from models.alpamayo import AlpamayoDrone
    from models.lora import load_lora_weights

    model = AlpamayoDrone(cfg)
    model.to(device)
    model.eval()

    ckpt_dir = Path(args.ckpt)
    print(f"[Eval] Loading checkpoint from {ckpt_dir}")
    load_lora_weights(model, str(ckpt_dir / "lora_weights.pt"))
    model.decoder.load_state_dict(
        torch.load(ckpt_dir / "decoder.pt", map_location=device)
    )

    # ------------------------------------------------------------------
    # Offline eval
    # ------------------------------------------------------------------
    if args.mode == "offline":
        from data.airsim_dataset import build_dataloaders
        from eval.evaluator import OfflineEvaluator

        _, val_loader = build_dataloaders(cfg, model.tokenizer)
        evaluator = OfflineEvaluator(
            model, val_loader, device,
            action_horizon=cfg["data"]["action_horizon"],
        )
        results = evaluator.evaluate()

        print("\n[Eval] Offline Results:")
        print(f"  L2 Action Error : {results['l2_action_error']:.4f}")
        print(f"  ADE             : {results['ade']:.4f}")
        print(f"  FDE             : {results['fde']:.4f}")

    # ------------------------------------------------------------------
    # Online eval (AirSim)
    # ------------------------------------------------------------------
    else:
        from eval.evaluator import AirSimOnlineEvaluator

        evaluator = AirSimOnlineEvaluator(model, cfg, device, model.tokenizer)
        results = evaluator.evaluate_all()

        print("\n[Eval] Online Results:")
        for r in results:
            print(f"  {r['task']:25s}  TSR={r['tsr']:.2%}  "
                  f"avg_steps={r['avg_steps']:.0f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Results saved -> {out_path}")


if __name__ == "__main__":
    main()
