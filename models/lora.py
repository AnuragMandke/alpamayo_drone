"""
lora.py — LoRA injection for the Alpamayo backbone.

Injects trainable low-rank adapters into target attention projections
while keeping all base weights frozen.
"""

import math
from typing import List
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps a Linear-like layer with W + B @ A (LoRA).

    The base layer is kept intact and invoked through its own forward().
    This is required for quantized layers (e.g. bitsandbytes Linear4bit,
    whose packed weight cannot go through F.linear) and keeps any custom
    forward logic of the wrapped module. Only A and B are trained.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base layer (may be nn.Linear or a quantized subclass)
        self.base = original
        self.in_features = original.in_features
        self.out_features = original.out_features

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(
            torch.empty(rank, original.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(original.out_features, rank)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: A ~ N(0, 1/sqrt(rank)), B = 0  → adapter starts at 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        # LoRA params live in fp32; cast to the activation dtype so this
        # works both under autocast(bf16) and in plain eval.
        lora = (
            self.dropout(x)
            @ self.lora_A.T.to(x.dtype)
            @ self.lora_B.T.to(x.dtype)
        ) * self.scaling
        return base + lora.to(base.dtype)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Walk the model, replace every Linear whose name ends with one of
    `target_modules` with a LoRALinear.

    Freezes ALL base parameters first, then marks LoRA params as trainable.
    """
    # Step 1: freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Step 2: inject LoRA into target linear layers
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in target_modules):
            continue

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)

        # Adapters must live where the (possibly dispatched) base weight lives
        device = module.weight.device
        lora_layer.lora_A.data = lora_layer.lora_A.data.to(device)
        lora_layer.lora_B.data = lora_layer.lora_B.data.to(device)

        # Navigate to parent and swap the child
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)
        replaced += 1

    print(f"[LoRA] Injected into {replaced} linear layers "
          f"(rank={rank}, alpha={alpha})")

    # Step 3: unfreeze LoRA params
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)

    return model


def count_trainable_params(model: nn.Module) -> dict:
    """Returns total, trainable, and frozen param counts + % trainable."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "pct_trainable": 100.0 * trainable / total if total > 0 else 0.0,
    }


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA weights (tiny checkpoint)."""
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, path)
    print(f"[LoRA] Saved {len(lora_state)} tensors -> {path}")


def load_lora_weights(model: nn.Module, path: str, strict: bool = True):
    """Load LoRA weights back into an already-injected model."""
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    lora_keys = [k for k in state]
    print(f"[LoRA] Loaded {len(lora_keys)} LoRA tensors from {path}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    return model
