"""
trainer.py — Training loop for Alpamayo-Drone

Features:
    - Gradient accumulation (effective batch = batch_size * grad_accum_steps)
    - Mixed precision (bf16)
    - Cosine LR schedule with warmup
    - Best-N checkpoint saving (by val loss)
    - WandB logging (optional)
"""

import os
import time
import math
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        tc = cfg["training"]
        self.epochs = tc["epochs"]
        self.grad_accum = tc["gradient_accumulation_steps"]
        self.max_grad_norm = tc["max_grad_norm"]
        self.output_dir = Path(tc["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = tc["save_every_n_epochs"]
        self.keep_best_n = tc["keep_best_n"]
        self.use_bf16 = tc["bf16"] and torch.cuda.is_bf16_supported()

        # Optimizer — only trainable params
        oc = tc["optimizer"]
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=oc["lr"],
            weight_decay=oc["weight_decay"],
            betas=tuple(oc["betas"]),
        )

        # LR Scheduler
        total_steps = (
            len(train_loader) // self.grad_accum
        ) * self.epochs
        sc = tc["scheduler"]
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=sc["warmup_steps"],
            num_training_steps=total_steps,
        )

        # GradScaler only for fp16; bf16 doesn't need it
        self.scaler = GradScaler(enabled=False)

        # Checkpoint tracking
        self.best_checkpoints: list[tuple[float, Path]] = []  # (val_loss, path)
        self.global_step = 0

    # ------------------------------------------------------------------

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Training Alpamayo-Drone")
        print(f"  Epochs: {self.epochs}  |  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"time={elapsed:.0f}s"
            )

            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "epoch": epoch,
                })

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch, val_loss)

        print("\n[Trainer] Training complete.")

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actions = batch["actions"].to(self.device)
            images = batch["images"].to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
            ):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    actions_gt=actions,
                )
                loss = out["loss"] / self.grad_accum

            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.grad_accum

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % 50 == 0:
                    print(
                        f"  [step {self.global_step}] "
                        f"loss={total_loss / (step + 1):.4f}"
                    )

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------

    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                actions = batch["actions"].to(self.device)
                images = batch["images"].to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
                ):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                        actions_gt=actions,
                    )
                total_loss += out["loss"].item()

        return total_loss / len(self.val_loader)

    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float):
        from models.lora import save_lora_weights

        ckpt_dir = self.output_dir / f"ckpt_epoch{epoch:03d}"
        ckpt_dir.mkdir(exist_ok=True)

        # Save LoRA weights (tiny — just the adapters)
        lora_path = ckpt_dir / "lora_weights.pt"
        save_lora_weights(self.model, str(lora_path))

        # Save decoder weights
        decoder_path = ckpt_dir / "decoder.pt"
        torch.save(self.model.decoder.state_dict(), decoder_path)

        # Save the (trainable) vision projection; the ViT itself is frozen and
        # restored from its pretrained weights, so it is not checkpointed.
        if getattr(self.model, "vis_proj", None) is not None:
            torch.save(self.model.vis_proj.state_dict(), ckpt_dir / "vision.pt")

        # Save metadata (incl. action normalization stats — eval needs them)
        meta = {
            "epoch": epoch,
            "val_loss": val_loss,
            "global_step": self.global_step,
            "action_mean": self.model.action_mean.detach().cpu(),
            "action_std": self.model.action_std.detach().cpu(),
        }
        torch.save(meta, ckpt_dir / "meta.pt")

        print(f"  [Checkpoint] Saved -> {ckpt_dir}  (val_loss={val_loss:.4f})")

        # Keep only best N checkpoints
        self.best_checkpoints.append((val_loss, ckpt_dir))
        self.best_checkpoints.sort(key=lambda x: x[0])
        while len(self.best_checkpoints) > self.keep_best_n:
            _, old_dir = self.best_checkpoints.pop()
            shutil.rmtree(old_dir, ignore_errors=True)
            print(f"  [Checkpoint] Pruned old checkpoint: {old_dir}")
