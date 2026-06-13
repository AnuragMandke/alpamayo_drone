"""
airsim_dataset.py — PyTorch Dataset for AirSim drone trajectories

Expected directory layout (produced by scripts/download_data.py):
    data/airsim/
        trajectories/
            traj_0000/
                images/          # rgb_000.png, rgb_001.png, ...
                actions.npy      # (T, 4)  [vx, vy, vz, yaw_rate]
                instructions.txt # one natural-language goal per line
            traj_0001/
            ...

Each sample returned by __getitem__:
    {
        "input_ids":      (seq_len,)           int64
        "attention_mask": (seq_len,)           int64
        "actions":        (action_horizon, 4)  float32
        "traj_id":        str                  (for debugging)
    }
"""

import os
import json
import glob
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ------------------------------------------------------------------
# Image preprocessing matching ViT-B/16 input requirements
# ------------------------------------------------------------------
IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class AirSimDroneDataset(Dataset):
    """
    Sliding-window dataset over AirSim drone trajectories.

    sequence_length:  number of past frames fed as visual context
    action_horizon:   number of future actions to predict
    """

    def __init__(
        self,
        root: str,
        tokenizer,
        split: str = "train",
        train_split: float = 0.9,
        sequence_length: int = 8,
        action_horizon: int = 4,
        max_seq_tokens: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.action_horizon = action_horizon
        self.max_seq_tokens = max_seq_tokens

        # Discover all trajectories
        all_trajs = sorted(
            (self.root / "trajectories").glob("traj_*")
        )
        if not all_trajs:
            raise FileNotFoundError(
                f"No trajectories found under {self.root / 'trajectories'}. "
                "Run scripts/download_data.py first."
            )

        # Deterministic train/val split
        rng = random.Random(seed)
        indices = list(range(len(all_trajs)))
        rng.shuffle(indices)
        n_train = int(len(indices) * train_split)

        if split == "train":
            selected = [all_trajs[i] for i in indices[:n_train]]
        else:
            selected = [all_trajs[i] for i in indices[n_train:]]

        # Build index: list of (traj_path, start_frame)
        self.samples = []
        for traj in selected:
            actions = np.load(traj / "actions.npy")      # (T, 4)
            T_len = len(actions)
            min_start = sequence_length
            max_start = T_len - action_horizon
            for t in range(min_start, max_start + 1):
                self.samples.append((traj, t))

        print(f"[Dataset] {split}: {len(self.samples)} samples "
              f"from {len(selected)} trajectories")

    def __len__(self) -> int:
        return len(self.samples)

    def compute_action_stats(self) -> dict:
        """
        Per-dimension mean/std over all actions of this split's trajectories.

        Used to normalize actions for flow matching (the decoder integrates
        from N(0, I), so targets should be roughly unit-scale).
        """
        trajs = sorted({traj for traj, _ in self.samples})
        all_actions = np.concatenate(
            [np.load(t / "actions.npy") for t in trajs], axis=0
        )  # (sum_T, action_dim)
        mean = all_actions.mean(axis=0)
        # Floor the std so near-constant dims don't blow up the normalization
        std = np.maximum(all_actions.std(axis=0), 1e-3)
        return {
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std":  torch.tensor(std, dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> dict:
        traj_path, t = self.samples[idx]

        # ---- Load past images ----------------------------------------
        image_files = sorted((traj_path / "images").glob("rgb_*.png"))
        frame_indices = range(t - self.sequence_length, t)
        images = []
        for i in frame_indices:
            i_clamped = max(0, i)
            img = Image.open(image_files[i_clamped]).convert("RGB")
            images.append(IMAGE_TRANSFORM(img))          # (3, 224, 224)

        images = torch.stack(images, dim=0)              # (seq_len, 3, 224, 224)

        # ---- Load instruction ----------------------------------------
        instr_file = traj_path / "instructions.txt"
        instructions = instr_file.read_text().strip().splitlines()
        instruction = random.choice(instructions) if instructions else "Navigate to the goal."

        # ---- Build prompt --------------------------------------------
        # Format: "<image> Instruction: <text> \nAction:"
        # The tokenizer will handle image token insertion for VLA models.
        # For non-VLA tokenizers, we include a placeholder.
        prompt = f"Instruction: {instruction}\nAction:"

        # No fixed-length padding here: batches are padded to the longest
        # prompt in the batch by make_collate_fn (saves memory and avoids
        # attending over hundreds of pad tokens).
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_seq_tokens,
            truncation=True,
        )

        # ---- Load future actions -------------------------------------
        actions_all = np.load(traj_path / "actions.npy")   # (T, 4)
        actions = actions_all[t: t + self.action_horizon]   # (action_horizon, 4)

        # Pad if trajectory ends early
        if len(actions) < self.action_horizon:
            pad = np.zeros((self.action_horizon - len(actions), 4), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=0)

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),      # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
            "actions":        torch.tensor(actions, dtype=torch.float32),
            "images":         images,                                  # (seq_len, 3, 224, 224)
            "traj_id":        traj_path.name,
        }


def make_collate_fn(tokenizer):
    """Pad variable-length prompts to the longest in the batch (right-padded)."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, :L] = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "actions":        torch.stack([b["actions"] for b in batch]),
            "images":         torch.stack([b["images"] for b in batch]),
            "traj_id":        [b["traj_id"] for b in batch],
        }

    return collate


def build_dataloaders(cfg: dict, tokenizer) -> tuple:
    """Returns (train_loader, val_loader)."""
    from torch.utils.data import DataLoader

    dc = cfg["data"]
    tc = cfg["training"]

    train_ds = AirSimDroneDataset(
        root=dc["dataset_root"],
        tokenizer=tokenizer,
        split="train",
        train_split=dc["train_split"],
        sequence_length=dc["sequence_length"],
        action_horizon=dc["action_horizon"],
    )
    val_ds = AirSimDroneDataset(
        root=dc["dataset_root"],
        tokenizer=tokenizer,
        split="val",
        train_split=dc["train_split"],
        sequence_length=dc["sequence_length"],
        action_horizon=dc["action_horizon"],
    )

    collate = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=dc["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=dc["num_workers"],
        pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, val_loader
