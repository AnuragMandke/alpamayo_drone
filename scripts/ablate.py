"""
scripts/ablate.py — Automated ablation study runner

Runs the two ablations described in the paper:
    1. LoRA rank sweep     (rank ∈ {4, 16, 64})
    2. Decoder type sweep  (FlowMatchingDecoder vs MLP baseline)

Results are saved to outputs/ablation_results.json and printed as a table.

Usage:
    python scripts/ablate.py --config configs/default.yaml
    python scripts/ablate.py --config configs/default.yaml --ablation rank
    python scripts/ablate.py --config configs/default.yaml --ablation decoder
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import yaml
from pathlib import Path

import torch


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# MLP Baseline Action Head (for ablation 2)
# ------------------------------------------------------------------

import torch.nn as nn

class MLPActionHead(nn.Module):
    """Simple MLP policy head — baseline to compare against FlowMatchingDecoder."""

    def __init__(self, context_dim: int, action_dim: int, action_horizon: int):
        super().__init__()
        self.action_horizon = action_horizon
        self.net = nn.Sequential(
            nn.Linear(context_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * action_horizon),
        )
        self.action_dim = action_dim

    def _pool(self, context, memory_key_padding_mask):
        """Masked mean over the context (ignore padding columns)."""
        if memory_key_padding_mask is None:
            return context.mean(dim=1)
        valid = (~memory_key_padding_mask).unsqueeze(-1).to(context.dtype)
        return (context * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def cfm_loss(self, action_gt, context, memory_key_padding_mask=None):
        ctx_pooled = self._pool(context, memory_key_padding_mask)   # (B, context_dim)
        pred = self.net(ctx_pooled).view(
            action_gt.shape[0], self.action_horizon, self.action_dim
        )
        return nn.functional.mse_loss(pred, action_gt)

    def sample(self, context, action_horizon=4, memory_key_padding_mask=None):
        ctx_pooled = self._pool(context, memory_key_padding_mask)
        return self.net(ctx_pooled).view(
            context.shape[0], self.action_horizon, self.action_dim
        )


# ------------------------------------------------------------------

def run_single(cfg: dict, run_name: str, device: torch.device) -> dict:
    """Train + offline eval for a single config. Returns metrics dict."""
    from models.alpamayo import AlpamayoDrone
    from data.airsim_dataset import build_dataloaders
    from training.trainer import Trainer
    from eval.evaluator import OfflineEvaluator

    print(f"\n{'='*55}")
    print(f"  RUN: {run_name}")
    print(f"{'='*55}")

    model = AlpamayoDrone(cfg)
    model.prepare_device(device)

    train_loader, val_loader = build_dataloaders(cfg, model.tokenizer)
    stats = train_loader.dataset.compute_action_stats()
    model.set_action_stats(stats["mean"], stats["std"])

    # Override output dir per run
    cfg_run = copy.deepcopy(cfg)
    cfg_run["training"]["output_dir"] = f"outputs/ablations/{run_name}"

    trainer = Trainer(model, train_loader, val_loader, cfg_run, device)
    trainer.train()

    model.eval()
    evaluator = OfflineEvaluator(
        model, val_loader, device,
        action_horizon=cfg["data"]["action_horizon"],
    )
    metrics = evaluator.evaluate()
    metrics["run"] = run_name
    return metrics


def ablation_lora_rank(cfg: dict, device: torch.device) -> list[dict]:
    results = []
    for rank in [4, 16, 64]:
        c = copy.deepcopy(cfg)
        c["model"]["lora"]["rank"] = rank
        c["model"]["lora"]["alpha"] = rank * 2   # keep alpha = 2*rank convention
        metrics = run_single(c, f"lora_rank_{rank}", device)
        results.append(metrics)
    return results


def ablation_decoder_type(cfg: dict, device: torch.device) -> list[dict]:
    results = []

    # Baseline 1: FlowMatchingDecoder (default)
    metrics = run_single(cfg, "decoder_flow_matching", device)
    results.append(metrics)

    # Baseline 2: MLP head
    # Monkey-patch the decoder after model creation
    import copy as _copy
    from models.alpamayo import AlpamayoDrone
    from data.airsim_dataset import build_dataloaders
    from training.trainer import Trainer
    from eval.evaluator import OfflineEvaluator

    c = _copy.deepcopy(cfg)
    c["training"]["output_dir"] = "outputs/ablations/decoder_mlp"

    model = AlpamayoDrone(c)
    context_dim = model.backbone.config.hidden_size
    model.decoder = MLPActionHead(
        context_dim=context_dim,
        action_dim=cfg["model"]["flow_matching"]["action_dim"],
        action_horizon=cfg["data"]["action_horizon"],
    )
    model.prepare_device(device)

    train_loader, val_loader = build_dataloaders(c, model.tokenizer)
    stats = train_loader.dataset.compute_action_stats()
    model.set_action_stats(stats["mean"], stats["std"])
    trainer = Trainer(model, train_loader, val_loader, c, device)
    trainer.train()

    model.eval()
    evaluator = OfflineEvaluator(model, val_loader, device,
                                 action_horizon=cfg["data"]["action_horizon"])
    metrics = evaluator.evaluate()
    metrics["run"] = "decoder_mlp"
    results.append(metrics)

    return results


def print_table(results: list[dict]):
    print(f"\n{'Run':<30} {'L2 Error':>10} {'ADE':>10} {'FDE':>10}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['run']:<30} "
            f"{r['l2_action_error']:>10.4f} "
            f"{r['ade']:>10.4f} "
            f"{r['fde']:>10.4f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ablation", choices=["rank", "decoder", "all"], default="all")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    all_results = []

    if args.ablation in ("rank", "all"):
        print("\n[Ablation 1] LoRA Rank Sweep")
        results = ablation_lora_rank(cfg, device)
        all_results.extend(results)
        print_table(results)

    if args.ablation in ("decoder", "all"):
        print("\n[Ablation 2] Decoder Type")
        results = ablation_decoder_type(cfg, device)
        all_results.extend(results)
        print_table(results)

    # Save
    out = Path("outputs/ablation_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Ablation] Results saved -> {out}")


if __name__ == "__main__":
    main()
