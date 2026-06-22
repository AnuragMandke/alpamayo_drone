"""
scripts/train_openvla.py — finetune OpenVLA-7B on drone trajectories.

Headline ablation for the cross-embodiment claim:
    --init pretrained   robot-pretrained OpenVLA + LoRA            (transfer arm)
    --init scratch      same arch, random init + LoRA              (weak control)
    --init prismatic    VL-pretrained, robot-naive base + LoRA     (clean control)

Run in the OpenVLA env (requirements-openvla.txt) on a >=16GB GPU:
    python scripts/train_openvla.py --config configs/openvla.yaml --init pretrained
    python scripts/train_openvla.py --config configs/openvla.yaml --init scratch

The --init prismatic arm additionally needs the prismatic-vlms package and a
>=24GB GPU (bf16 full backbone); see docs/OPENVLA.md.

NOTE: cannot run in the main (transformers 5.x) env or on the 6GB dev laptop.
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
    p.add_argument("--device", default=None)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def move_batch(batch, device, pixel_dtype):
    """Move a batch to device; cast pixel_values to pixel_dtype. pixel_values is a
    tensor (OpenVLA) or a dict of tensors (Prismatic dinosiglip)."""
    out = {}
    for k, v in batch.items():
        if k == "pixel_values":
            out[k] = ({kk: vv.to(device, pixel_dtype) for kk, vv in v.items()}
                      if isinstance(v, dict) else v.to(device, pixel_dtype))
        else:
            out[k] = v.to(device)
    return out


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]
    set_seed(tc["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from models.openvla_policy import (
        build_openvla_policy, build_prismatic_policy, trainable_parameters,
    )
    from models.action_tokenizer import ActionTokenizer
    from data.openvla_dataset import (
        OpenVLADroneDataset, PrismaticDroneDataset, compute_drone_norm_stats,
        make_openvla_collate, make_prismatic_collate,
    )

    is_prismatic = args.init == "prismatic"

    # ---- Model + processor ------------------------------------------------
    processor = image_transform = None
    if is_prismatic:
        model, tokenizer, image_transform = build_prismatic_policy(
            lora_rank=mc["lora"]["rank"],
            lora_alpha=mc["lora"]["alpha"],
            lora_dropout=mc["lora"]["dropout"],
        )
        model = model.to(device)
    else:
        model, processor = build_openvla_policy(
            init=args.init,
            load_in_4bit=mc["load_in_4bit"] and args.init == "pretrained",
            lora_rank=mc["lora"]["rank"],
            lora_alpha=mc["lora"]["alpha"],
            lora_dropout=mc["lora"]["dropout"],
        )
        if args.init == "scratch":
            model = model.to(device)
        tokenizer = processor.tokenizer

    # ---- Action normalization stats (saved with the run) ------------------
    out_dir = Path(tc["output_dir"]) / args.init
    out_dir.mkdir(parents=True, exist_ok=True)
    target_mode = dc.get("target_mode", "velocity")
    waypoint_horizon = dc.get("waypoint_horizon", 8)
    stats = compute_drone_norm_stats(
        dc["dataset_root"], dc["train_split"], tc["seed"],
        target_mode=target_mode, waypoint_horizon=waypoint_horizon,
    )
    json.dump(stats, open(out_dir / "drone_norm_stats.json", "w"), indent=2)
    print(f"[Train] target_mode={target_mode}"
          + (f" horizon={waypoint_horizon}" if target_mode == "waypoint" else ""))

    atok = ActionTokenizer(tokenizer)
    pad_id = tokenizer.pad_token_id or 0
    if is_prismatic:
        train_ds = PrismaticDroneDataset(
            dc["dataset_root"], tokenizer, image_transform, atok, stats,
            split="train", train_split=dc["train_split"], seed=tc["seed"],
            predict_offset=dc["predict_offset"],
            target_mode=target_mode, waypoint_horizon=waypoint_horizon,
        )
        collate = make_prismatic_collate(pad_id)
    else:
        train_ds = OpenVLADroneDataset(
            dc["dataset_root"], processor, atok, stats, split="train",
            train_split=dc["train_split"], seed=tc["seed"],
            predict_offset=dc["predict_offset"],
            target_mode=target_mode, waypoint_horizon=waypoint_horizon,
        )
        collate = make_openvla_collate(pad_id)
    loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=dc["num_workers"], pin_memory=True, drop_last=True,
        collate_fn=collate,
    )

    # ---- Optimizer --------------------------------------------------------
    optim = torch.optim.AdamW(
        trainable_parameters(model), lr=tc["optimizer"]["lr"],
        weight_decay=tc["optimizer"]["weight_decay"],
    )
    grad_accum = tc["gradient_accumulation_steps"]

    # ---- Train ------------------------------------------------------------
    model.train()
    step = 0
    for epoch in range(1, tc["epochs"] + 1):
        running = 0.0
        optim.zero_grad()
        for i, batch in enumerate(loader):
            batch = move_batch(batch, device, torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc["bf16"]):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = out.loss / grad_accum
            loss.backward()
            running += loss.item() * grad_accum

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), tc["max_grad_norm"])
                optim.step(); optim.zero_grad(); step += 1
                if step % tc["log_every_n_steps"] == 0:
                    print(f"  epoch {epoch} step {step}  loss={running / (i + 1):.4f}", flush=True)

        print(f"Epoch {epoch}/{tc['epochs']}  loss={running / len(loader):.4f}", flush=True)
        if epoch % tc["save_every_n_epochs"] == 0 or epoch == tc["epochs"]:
            ckpt = out_dir / f"epoch{epoch:03d}"
            ckpt.mkdir(exist_ok=True)
            if is_prismatic:
                # Hand-rolled LoRA (not PEFT) — save just the adapter tensors.
                from models.lora import save_lora_weights
                save_lora_weights(model, str(ckpt / "lora_weights.pt"))
            else:
                model.save_pretrained(str(ckpt))  # PEFT saves adapters; scratch saves full
            print(f"  [ckpt] {ckpt}", flush=True)

    print(f"[Done] init={args.init} -> {out_dir}")


if __name__ == "__main__":
    main()
