"""
scripts/integration_test.py — One-batch gate for the real backbone path.

Loads the model exactly as training would (4-bit backbone + LoRA + ViT +
FlowMatchingDecoder), runs a single forward/backward on a synthetic batch,
and asserts:
    1. CFM loss is finite
    2. LoRA adapters receive nonzero gradients
    3. The vision projection receives nonzero gradients
    4. The decoder receives nonzero gradients
    5. sample() produces finite actions of the right shape

Run this before any real training run; if it passes, the full pipeline is
mechanically sound on this machine.

Usage:
    python scripts/integration_test.py                                 # real backbone (downloads it)
    python scripts/integration_test.py --config configs/smoke_test.yaml # tiny backbone, fast
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--base_model", default=None,
                   help="Override model.base_model_id (e.g. a smaller model "
                        "to validate the 4-bit path before a big download)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--device", default=None)
    return p.parse_args()


def grad_norm(params) -> float:
    norms = [p.grad.norm().item() for p in params if p.grad is not None]
    return sum(norms) if norms else 0.0


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.base_model:
        cfg["model"]["base_model_id"] = args.base_model

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[ITest] Config: {args.config}  |  Device: {device}  |  bf16: {use_bf16}")

    from models.alpamayo import AlpamayoDrone
    from models.lora import LoRALinear

    model = AlpamayoDrone(cfg)
    model.prepare_device(device)
    model.train()

    if device.type == "cuda":
        load_gb = torch.cuda.memory_allocated() / 1024**3
        load_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[ITest] VRAM after load: {load_gb:.2f} GB (load peak: {load_peak_gb:.2f} GB)")
        torch.cuda.reset_peak_memory_stats()

    n_lora = sum(1 for m in model.backbone.modules() if isinstance(m, LoRALinear))
    assert n_lora > 0, "No LoRA layers were injected into the backbone"
    print(f"[ITest] {n_lora} LoRA layers active in backbone")

    # ---- Synthetic batch (same shapes the dataset produces) -------------
    B = args.batch_size
    S = cfg["data"]["sequence_length"]
    H = cfg["data"]["action_horizon"]
    A = cfg["model"]["flow_matching"]["action_dim"]

    enc = model.tokenizer(
        ["Instruction: Fly to the goal marker ahead.\nAction:"] * B,
        return_tensors="pt", padding=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    images = torch.randn(B, S, 3, 224, 224, device=device)
    actions_gt = torch.randn(B, H, A, device=device)

    # ---- Forward + backward ---------------------------------------------
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        enabled=use_bf16,
    ):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            actions_gt=actions_gt,
        )
    loss = out["loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f"[ITest] CFM loss: {loss.item():.4f}")

    loss.backward()

    lora_params = [
        p for m in model.backbone.modules() if isinstance(m, LoRALinear)
        for p in (m.lora_A, m.lora_B)
    ]
    checks = {
        "lora":     grad_norm(lora_params),
        "vis_proj": grad_norm(model.vis_proj.parameters()) if model.vis_proj else None,
        "decoder":  grad_norm(model.decoder.parameters()),
    }
    for name, g in checks.items():
        if g is None:
            print(f"[ITest] {name:8s} grad: (disabled)")
            continue
        assert g > 0, f"{name} received zero/no gradient"
        print(f"[ITest] {name:8s} grad norm sum: {g:.6f}")

    # ---- Inference path ---------------------------------------------------
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        enabled=use_bf16,
    ):
        out = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
    actions = out["actions"]
    assert actions.shape == (B, 4, A), f"Unexpected action shape: {actions.shape}"
    assert torch.isfinite(actions).all(), "Sampled actions contain non-finite values"
    print(f"[ITest] sample() OK: shape {tuple(actions.shape)}")

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[ITest] Peak VRAM during fwd/bwd/sample (excl. load): {peak_gb:.2f} GB")

    print("\n[ITest] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
