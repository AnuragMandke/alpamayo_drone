"""
data/openvla_dataset.py — drone trajectories in OpenVLA's native training format.

Unlike the FlowMatchingDecoder pipeline (8-frame history + 4-action horizon),
OpenVLA is a single-frame, single-action VLA: observe the current RGB frame +
instruction, predict one 7-DoF action as discrete tokens. We reuse the same
on-disk trajectory layout (images/rgb_*.png, actions.npy, instructions.txt) but
emit OpenVLA-style supervised examples:

    pixel_values : from PrismaticProcessor (DINOv2+SigLIP, 6x224x224)
    input_ids    : [ "In: What action ... {instr}?\nOut:" tokens ] + [7 action tokens] + [EOS]
    labels       : [ -100 ... -100 (prompt masked) ]            + [7 action tokens] + [EOS]

Drone 4-DoF [vx,vy,vz,yaw_rate] is normalized per-dim to [-1,1] using [1%,99%]
quantiles (OpenVLA convention), embedded into the 7-DoF EEF layout, and
tokenized with models.action_tokenizer.ActionTokenizer.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from models.action_tokenizer import ActionTokenizer, drone_to_openvla

PROMPT = "In: What action should the robot take to {instruction}?\nOut:"


def compute_drone_norm_stats(root: str, train_split: float = 0.9,
                             seed: int = 42) -> dict:
    """
    Per-dim [1%, 99%] quantiles + min/max over the TRAIN trajectories.
    Mirrors OpenVLA's dataset_statistics so actions normalize to ~[-1, 1].
    """
    root = Path(root)
    trajs = sorted((root / "trajectories").glob("traj_*"))
    rng = random.Random(seed)
    idx = list(range(len(trajs))); rng.shuffle(idx)
    n_train = int(len(idx) * train_split)
    train_trajs = [trajs[i] for i in idx[:n_train]]

    actions = np.concatenate([np.load(t / "actions.npy") for t in train_trajs], axis=0)
    return {
        "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "num_transitions": int(len(actions)),
    }


def normalize_action(action: np.ndarray, stats: dict) -> np.ndarray:
    """Map raw drone action -> [-1, 1] via [q01, q99], clipped."""
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    denom = np.maximum(q99 - q01, 1e-6)
    norm = 2.0 * (action - q01) / denom - 1.0
    return np.clip(norm, -1.0, 1.0)


def denormalize_action(norm: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of normalize_action: [-1,1] -> raw drone units."""
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return (norm + 1.0) / 2.0 * (q99 - q01) + q01


class OpenVLADroneDataset(Dataset):
    """One supervised (image, prompt, action-tokens) example per timestep."""

    def __init__(self, root, processor, action_tokenizer: ActionTokenizer,
                 norm_stats: dict, split: str = "train", train_split: float = 0.9,
                 seed: int = 42, predict_offset: int = 0):
        self.root = Path(root)
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.norm_stats = norm_stats
        self.predict_offset = predict_offset   # action at t+offset for frame t

        trajs = sorted((self.root / "trajectories").glob("traj_*"))
        if not trajs:
            raise FileNotFoundError(f"No trajectories under {self.root/'trajectories'}")
        rng = random.Random(seed)
        idx = list(range(len(trajs))); rng.shuffle(idx)
        n_train = int(len(idx) * train_split)
        sel = idx[:n_train] if split == "train" else idx[n_train:]
        selected = [trajs[i] for i in sel]

        # Flat index of (traj, frame_t)
        self.samples = []
        for traj in selected:
            n = len(np.load(traj / "actions.npy"))
            for t in range(n - predict_offset):
                self.samples.append((traj, t))
        print(f"[OpenVLA-Dataset] {split}: {len(self.samples)} samples "
              f"from {len(selected)} trajectories")

        self.eos_token_id = processor.tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        traj, t = self.samples[i]
        img_files = sorted((traj / "images").glob("rgb_*.png"))
        image = Image.open(img_files[t]).convert("RGB")

        instrs = (traj / "instructions.txt").read_text().strip().splitlines()
        instruction = random.choice(instrs) if instrs else "navigate to the goal"
        prompt = PROMPT.format(instruction=instruction)

        action = np.load(traj / "actions.npy")[t + self.predict_offset]   # (4,)
        norm = normalize_action(action, self.norm_stats)                  # (4,) in [-1,1]
        action7 = drone_to_openvla(norm)                                  # (7,)
        action_tokens = self.action_tokenizer(action7).astype(np.int64)   # (7,)

        enc = self.processor(text=prompt, images=image, return_tensors="pt")
        prompt_ids = enc["input_ids"][0]                  # (P,)
        pixel_values = enc["pixel_values"][0]             # (6, 224, 224)

        act_ids = torch.from_numpy(action_tokens)
        eos = torch.tensor([self.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, act_ids, eos])
        labels = torch.cat([
            torch.full((len(prompt_ids),), -100, dtype=torch.long),
            act_ids, eos,
        ])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "attention_mask": torch.ones_like(input_ids),
        }


def make_openvla_collate(pad_token_id: int):
    """Right-pad input_ids/labels/attention_mask; stack pixel_values."""
    def collate(batch):
        maxlen = max(b["input_ids"].shape[0] for b in batch)
        B = len(batch)
        input_ids = torch.full((B, maxlen), pad_token_id, dtype=torch.long)
        labels = torch.full((B, maxlen), -100, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, :L] = b["input_ids"]
            labels[i, :L] = b["labels"]
            attn[i, :L] = b["attention_mask"]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        }
    return collate
