"""
scripts/eval_openvla.py — offline evaluation for the OpenVLA-family arms.

Scores a trained checkpoint on the val split with teacher-forced action-token
accuracy + decoded drone-action error (eval/openvla_evaluator.py). This is the
metric the cross-embodiment ablation is read on; run it for each arm and compare:

    python scripts/eval_openvla.py --config configs/openvla.yaml \
        --init pretrained --ckpt outputs/openvla/pretrained/epoch005
    python scripts/eval_openvla.py --config configs/openvla.yaml \
        --init scratch    --ckpt outputs/openvla/scratch/epoch005
    python scripts/eval_openvla.py --config configs/openvla.yaml \
        --init prismatic  --ckpt outputs/openvla/prismatic/epoch005

Run in the matching env (OpenVLA env for pretrained/scratch; prismatic-vlms env
for prismatic) on the same GPU class as training. UNVALIDATED on this machine —
the model-load paths need the weights + lab GPU; see docs/OPENVLA.md.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/openvla.yaml")
    p.add_argument("--init", choices=["pretrained", "scratch", "prismatic"],
                   default="pretrained")
    p.add_argument("--ckpt", required=True,
                   help="Checkpoint dir, e.g. outputs/openvla/pretrained/epoch005")
    p.add_argument("--device", default=None)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _load_peft_adapters(model, ckpt_dir):
    """Load PEFT adapter weights (pretrained/scratch arms) into a freshly-built
    PEFT model. Mirrors how train_openvla.py saved them via save_pretrained."""
    from peft import set_peft_model_state_dict
    ckpt_dir = Path(ckpt_dir)
    safet = ckpt_dir / "adapter_model.safetensors"
    if safet.exists():
        from safetensors.torch import load_file
        sd = load_file(str(safet))
    else:
        sd = torch.load(ckpt_dir / "adapter_model.bin", map_location="cpu")
    set_peft_model_state_dict(model, sd)


def load_policy(args, cfg, device):
    """Rebuild the arm via the same builders used in training and load the
    trained weights. Returns (model, tokenizer, image_transform_or_None,
    processor_or_None)."""
    mc, tc = cfg["model"], cfg["training"]
    lora = mc["lora"]

    if args.init == "prismatic":
        from models.openvla_policy import build_prismatic_policy
        from models.lora import load_lora_weights
        model, tokenizer, image_transform = build_prismatic_policy(
            lora_rank=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
        )
        load_lora_weights(model, str(Path(args.ckpt) / "lora_weights.pt"))
        model = model.to(device)
        return model, tokenizer, image_transform, None

    from models.openvla_policy import build_openvla_policy
    # The scratch base is random-init; rebuild it under the training seed so the
    # frozen base matches the one the adapters were trained on.
    if args.init == "scratch":
        set_seed(tc["seed"])
    model, processor = build_openvla_policy(
        init=args.init,
        load_in_4bit=mc["load_in_4bit"] and args.init == "pretrained",
        lora_rank=lora["rank"], lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
    )
    if args.init == "scratch":
        model = model.to(device)
    _load_peft_adapters(model, args.ckpt)
    return model, processor.tokenizer, None, processor


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from models.action_tokenizer import ActionTokenizer
    from data.openvla_dataset import (
        OpenVLADroneDataset, PrismaticDroneDataset, compute_drone_norm_stats,
        make_openvla_collate, make_prismatic_collate,
    )
    from eval.openvla_evaluator import evaluate_openvla

    target_mode = dc.get("target_mode", "velocity")
    waypoint_horizon = dc.get("waypoint_horizon", 8)

    # ---- Drone normalization stats: prefer the ones saved with the run -------
    stats_path = Path(args.ckpt).parent / "drone_norm_stats.json"
    if stats_path.exists():
        stats = json.load(open(stats_path))
        print(f"[Eval] Loaded drone norm stats from {stats_path}")
    else:
        stats = compute_drone_norm_stats(
            dc["dataset_root"], dc["train_split"], tc["seed"],
            target_mode=target_mode, waypoint_horizon=waypoint_horizon,
        )
        print("[Eval] Recomputed drone norm stats from train split")

    # ---- Model + tokenizer ---------------------------------------------------
    model, tokenizer, image_transform, processor = load_policy(args, cfg, device)
    atok = ActionTokenizer(tokenizer)
    pad_id = tokenizer.pad_token_id or 0

    # ---- Val dataset ---------------------------------------------------------
    if args.init == "prismatic":
        val_ds = PrismaticDroneDataset(
            dc["dataset_root"], tokenizer, image_transform, atok, stats,
            split="val", train_split=dc["train_split"], seed=tc["seed"],
            predict_offset=dc["predict_offset"],
            target_mode=target_mode, waypoint_horizon=waypoint_horizon,
        )
        collate = make_prismatic_collate(pad_id)
    else:
        val_ds = OpenVLADroneDataset(
            dc["dataset_root"], processor, atok, stats, split="val",
            train_split=dc["train_split"], seed=tc["seed"],
            predict_offset=dc["predict_offset"],
            target_mode=target_mode, waypoint_horizon=waypoint_horizon,
        )
        collate = make_openvla_collate(pad_id)
    loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=dc["num_workers"], pin_memory=True, collate_fn=collate,
    )

    # ---- Evaluate ------------------------------------------------------------
    metrics = evaluate_openvla(model, loader, atok, stats, device, bf16=tc["bf16"])
    metrics["init"] = args.init
    metrics["ckpt"] = str(args.ckpt)

    print(f"\n[Eval] init={args.init}  ckpt={args.ckpt}")
    print(f"  action_token_accuracy : {metrics['action_token_accuracy']:.4f}")
    print(f"  action_l2             : {metrics['action_l2']:.4f}")
    pdm = metrics["per_dim_mae"]
    print(f"  per_dim_mae           : vx={pdm['vx']:.4f} vy={pdm['vy']:.4f} "
          f"vz={pdm['vz']:.4f} yaw_rate={pdm['yaw_rate']:.4f}")
    print(f"  n_samples             : {metrics['n_samples']}")

    out_path = Path(args.ckpt) / "eval_metrics.json"
    json.dump(metrics, open(out_path, "w"), indent=2)
    print(f"[Eval] Saved -> {out_path}")


if __name__ == "__main__":
    main()
