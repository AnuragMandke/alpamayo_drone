"""
scripts/diagnose_prismatic_eval.py — why does the prismatic arm score 0.0000?

The prismatic arm trained to a healthy loss (~2.63 windowed, better than
scratch's ~2.67) but the offline scorer reports action_token_accuracy exactly
0.0000 and an action_l2 twice scratch's. Those cannot both describe the same
model, and exact-zero is BELOW chance (~1/256 per token), so the argmax is
landing outside the action-token range entirely.

Training reads `out.loss` (scripts/train_openvla.py:216) — alignment is the
model's own business there. The scorer instead does its own causal shift over
`out.logits` (eval/openvla_evaluator.py:84-92), which is only valid when
logits and labels share a length. Prismatic splices image-patch embeddings into
the sequence, so its logits may be ~256 longer than the text-stream labels; the
scorer would then read predictions out of the patch region. The HF OpenVLA path
returns logits aligned to input_ids, which is why pretrained/scratch are fine.

This separates the two candidates in one forward pass:

  val loss ~2.6  AND  logits longer than labels
      -> model is fine and fully loaded; the SCORER is misaligned.
  val loss garbage (>>3)  OR  few matching LoRA keys
      -> the adapters never loaded (load_lora_weights reports the file's tensor
         count, not how many actually matched), and the scorer is a red herring.

Neither outcome needs retraining. Run in the OpenVLA env:

    CUDA_VISIBLE_DEVICES=6 python scripts/diagnose_prismatic_eval.py
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.openvla_dataset import PrismaticDroneDataset, make_prismatic_collate
from models.action_tokenizer import ActionTokenizer
from models.lora import load_lora_weights
from models.openvla_policy import (
    build_prismatic_policy, resolve_amp, resolve_arm_training_cfg,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/openvla.yaml")
    p.add_argument("--ckpt", default="outputs/openvla/prismatic/epoch003")
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]
    tc = resolve_arm_training_cfg(tc, "prismatic")
    amp_dtype, amp_enabled = resolve_amp(tc)
    ckpt = Path(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, image_transform = build_prismatic_policy(
        lora_rank=mc["lora"]["rank"],
        lora_alpha=mc["lora"]["alpha"],
        lora_dropout=mc["lora"]["dropout"],
        lora_targets=mc["lora"].get("targets", "all-linear"),
    )

    # --- 1. did the adapters actually land? --------------------------------
    # load_lora_weights prints len(file), not the number of keys that matched,
    # so a prefix mismatch is silent. Count the overlap ourselves.
    state = torch.load(ckpt / "lora_weights.pt", map_location="cpu")
    known = set(model.state_dict().keys())
    matched = sum(1 for k in state if k in known)
    print(f"\n>> LoRA tensors in file : {len(state)}")
    print(f">> matching model keys  : {matched}"
          f"{'   <-- MISMATCH' if matched != len(state) else ''}")

    load_lora_weights(model, str(ckpt / "lora_weights.pt"))
    model = model.to(device).eval()

    # --- 2. one val batch through the real forward -------------------------
    stats = json.load(open(ckpt.parent / "drone_norm_stats.json"))
    atok = ActionTokenizer(tokenizer)
    ds = PrismaticDroneDataset(
        dc["dataset_root"], tokenizer, image_transform, atok, stats,
        split="val", train_split=dc["train_split"], seed=tc["seed"],
        predict_offset=dc["predict_offset"],
        target_mode=dc.get("target_mode", "velocity"),
        waypoint_horizon=dc.get("waypoint_horizon", 8),
        waypoint_horizon_seconds=dc.get("waypoint_horizon_seconds"),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_prismatic_collate(tokenizer.pad_token_id or 0))
    batch = next(iter(loader))

    def to_dev(k, v):
        if k != "pixel_values":
            return v.to(device)
        if isinstance(v, dict):
            return {kk: vv.to(device, amp_dtype) for kk, vv in v.items()}
        return v.to(device, amp_dtype)

    batch = {k: to_dev(k, v) for k, v in batch.items()}

    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype,
                                         enabled=amp_enabled and device.type == "cuda"):
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )

    n_lab = batch["labels"].shape[1]
    n_log = out.logits.shape[1]
    print(f"\n>> input_ids : {tuple(batch['input_ids'].shape)}")
    print(f">> labels    : {tuple(batch['labels'].shape)}")
    print(f">> logits    : {tuple(out.logits.shape)}")
    print(f">> extra positions in logits: {n_log - n_lab}")
    print(f">> val loss  : {float(out.loss):.4f}   (training ended ~2.63)")

    # --- 3. what the scorer would actually read ----------------------------
    lo, hi = atok.action_token_id_range
    pred = out.logits[:, :-1].float().argmax(-1)
    gold = batch["labels"][:, 1:]
    pos = (gold[0] != -100).nonzero(as_tuple=True)[0][:7]
    if int(pos.max()) < pred.shape[1]:
        read = pred[0, pos].tolist()
        in_range = sum(lo <= t <= hi for t in read)
        print(f"\n>> action-token id range : [{lo}, {hi}]")
        print(f">> scorer reads at those positions: {read}")
        print(f">> of 7 reads, {in_range} fall inside the action range")

    print("\n>> VERDICT:", "LOGITS MISALIGNED — the scorer is reading the wrong "
          "positions; fix eval/openvla_evaluator.py, no retraining"
          if n_log != n_lab else
          "lengths match — the scorer's shift is fine, look at adapter loading")


if __name__ == "__main__":
    main()
