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

PrismaticDroneDataset emits the same examples for the `--init prismatic` control
arm, differing only in pixel format (a dinosiglip {"dino","siglip"} dict instead
of a stacked 6-channel tensor).
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from models.action_tokenizer import (
    ActionTokenizer, drone_to_openvla, OPENVLA_ACTION_DIM,
)

PROMPT = "In: What action should the robot take to {instruction}?\nOut:"


# ----------------------------------------------------------------------------
# Target derivation
#
# "velocity"  : instantaneous body-frame [vx, vy, vz, yaw_rate] (read straight
#               from actions.npy). NOT recoverable from a single RGB frame —
#               motion is invisible in one image, so this is an ill-posed
#               single-frame regression (see docs/OPENVLA.md).
# "waypoint"  : body-frame displacement to the pose `horizon` steps ahead,
#               [dx, dy, dz, dyaw] (derived from poses.npy). Well-posed from a
#               single frame and semantically matches OpenVLA's native action
#               (relative EEF translation), so the pretrained prior transfers
#               more directly. Requires poses.npy (re-run convert_uzh_fpv.py).
# ----------------------------------------------------------------------------

def build_waypoint_targets(poses: np.ndarray, horizon: int) -> np.ndarray:
    """
    poses: (T, 7) world-frame [x, y, z, qx, qy, qz, qw].
    Returns (T - horizon, 4) body-frame [dx, dy, dz, dyaw]: the displacement to
    the pose `horizon` steps ahead expressed in the body frame at t, plus the net
    heading change. No dt (displacement, not a rate) — so it's free of the
    finite-difference noise that plagues velocity targets.
    """
    from scipy.spatial.transform import Rotation
    pos = poses[:, :3]
    rot = Rotation.from_quat(poses[:, 3:7])
    rot_inv = rot.inv()
    n = len(poses) - horizon
    out = np.zeros((n, 4), dtype=np.float32)
    for t in range(n):
        out[t, :3] = rot_inv[t].apply(pos[t + horizon] - pos[t])
        out[t, 3] = (rot_inv[t] * rot[t + horizon]).as_euler("zyx")[0]
    return out


def _single_waypoint(poses: np.ndarray, t: int, horizon: int) -> np.ndarray:
    """Body-frame [dx, dy, dz, dyaw] for one frame t (cheap; avoids rebuilding
    the whole trajectory's targets on every __getitem__)."""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(poses[[t, t + horizon], 3:7])   # R_t, R_{t+h}
    out = np.zeros(4, dtype=np.float32)
    out[:3] = rot[0].inv().apply(poses[t + horizon, :3] - poses[t, :3])
    out[3] = (rot[0].inv() * rot[1]).as_euler("zyx")[0]
    return out


def _train_targets(root: str, train_split: float, seed: int,
                   target_mode: str, waypoint_horizon: int) -> np.ndarray:
    """(N, 4) raw targets over the TRAIN trajectories. Shared by the norm stats
    and the marginal-loss floor so both see exactly the same split."""
    root = Path(root)
    trajs = sorted((root / "trajectories").glob("traj_*"))
    if not trajs:
        # Without this, the empty list reaches np.concatenate and surfaces as
        # "need at least one array to concatenate" — which says nothing about the
        # actual cause. On Colab the usual cause is that the data never got
        # extracted, or that re-running the clone cell deleted it (the dataset
        # lives inside the repo dir, so `rm -rf` takes it with the code).
        raise FileNotFoundError(
            f"No trajectories found at {(root / 'trajectories').resolve()} "
            f"(looking for traj_*/). On Colab: re-run the Drive/extract cell — "
            f"the clone cell wipes the repo dir, and the dataset lives inside it. "
            f"Locally: python scripts/convert_uzh_fpv.py --src data/raw/uzh_fpv "
            f"--dst {root}"
        )
    rng = random.Random(seed)
    idx = list(range(len(trajs))); rng.shuffle(idx)
    n_train = int(len(idx) * train_split)
    train_trajs = [trajs[i] for i in idx[:n_train]]

    if target_mode == "waypoint":
        missing = [t.name for t in train_trajs if not (t / "poses.npy").exists()]
        if missing:
            raise FileNotFoundError(
                f"target_mode='waypoint' needs poses.npy, but {len(missing)} of "
                f"{len(train_trajs)} train trajectories lack it (e.g. {missing[0]}). "
                f"That data predates commit 6789b8f — re-convert it with "
                f"scripts/convert_uzh_fpv.py and re-upload."
            )
        return np.concatenate(
            [build_waypoint_targets(np.load(t / "poses.npy"), waypoint_horizon)
             for t in train_trajs], axis=0)
    return np.concatenate([np.load(t / "actions.npy") for t in train_trajs], axis=0)


def marginal_loss_floor(root: str, stats: dict, train_split: float = 0.9,
                        seed: int = 42, target_mode: str = "velocity",
                        waypoint_horizon: int = 8, bins: int = 256):
    """
    The mean CE a model scores by predicting the MARGINAL action distribution and
    ignoring the image entirely. This is the yardstick for reading a training
    curve: a run that settles here has learned the action prior and NOTHING about
    what the camera sees, however smooth its loss looks.

    Why it is (sum of per-dim entropies)/8 and not their mean: the labels
    supervise 7 action tokens + EOS = 8 positions, but drone_to_openvla writes a
    constant 0.0 into roll/pitch/gripper on every sample, so those 3 dims — and
    EOS — carry ~no entropy and are free once learned. Only the 4 dims at
    DRONE_TO_OPENVLA_IDX carry drone data, so the reported loss is diluted ~2x
    relative to the error on the dims that matter. (Measured 2026-07-16 on the
    TRAIN split: velocity floor 2.576 vs a 50-step run that plateaued at 2.450;
    waypoint floor 2.563 vs 2.482. Both runs were ~0.1 nats off their floor, i.e.
    at 200 samples neither had learned anything much from the image.)

    Returns (floor_nats, per_dim_entropies).
    """
    targets = _train_targets(root, train_split, seed, target_mode, waypoint_horizon)
    norm = normalize_action(targets, stats)
    edges = np.linspace(-1.0, 1.0, bins)          # mirrors ActionTokenizer.bins
    H = []
    for d in range(norm.shape[1]):
        _, counts = np.unique(np.digitize(norm[:, d], edges), return_counts=True)
        p = counts / counts.sum()
        H.append(float(-(p * np.log(p)).sum()))
    return sum(H) / (OPENVLA_ACTION_DIM + 1), H


def compute_drone_norm_stats(root: str, train_split: float = 0.9,
                             seed: int = 42, target_mode: str = "velocity",
                             waypoint_horizon: int = 8) -> dict:
    """
    Per-dim [1%, 99%] quantiles + min/max over the TRAIN trajectories' targets.
    Mirrors OpenVLA's dataset_statistics so targets normalize to ~[-1, 1].
    `target_mode` selects velocity (actions.npy) vs waypoint (poses.npy).
    """
    actions = _train_targets(root, train_split, seed, target_mode, waypoint_horizon)
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
                 seed: int = 42, predict_offset: int = 0,
                 target_mode: str = "velocity", waypoint_horizon: int = 8):
        self.root = Path(root)
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.norm_stats = norm_stats
        self.predict_offset = predict_offset   # action at t+offset for frame t
        self.target_mode = target_mode
        self.waypoint_horizon = waypoint_horizon

        trajs = sorted((self.root / "trajectories").glob("traj_*"))
        if not trajs:
            raise FileNotFoundError(f"No trajectories under {self.root/'trajectories'}")
        rng = random.Random(seed)
        idx = list(range(len(trajs))); rng.shuffle(idx)
        n_train = int(len(idx) * train_split)
        sel = idx[:n_train] if split == "train" else idx[n_train:]
        selected = [trajs[i] for i in sel]

        # Flat index of (traj, frame_t). Valid frames depend on the target: a
        # waypoint needs the pose `horizon` ahead; a velocity needs t+offset.
        self.samples = []
        for traj in selected:
            if target_mode == "waypoint":
                poses_file = traj / "poses.npy"
                if not poses_file.is_file():
                    raise FileNotFoundError(
                        f"target_mode='waypoint' needs {poses_file}, which is "
                        "missing. Re-run scripts/convert_uzh_fpv.py to emit "
                        "poses.npy alongside actions.npy.")
                usable = len(np.load(poses_file)) - waypoint_horizon
            else:
                usable = len(np.load(traj / "actions.npy")) - predict_offset
            for t in range(usable):
                self.samples.append((traj, t))
        print(f"[OpenVLA-Dataset] {split}: {len(self.samples)} samples "
              f"from {len(selected)} trajectories (target={target_mode})")

        self.eos_token_id = processor.tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def _raw_target(self, traj, t):
        """The raw 4-DoF target for frame t (velocity or waypoint), pre-norm."""
        if self.target_mode == "waypoint":
            return _single_waypoint(np.load(traj / "poses.npy"), t,
                                    self.waypoint_horizon)
        return np.load(traj / "actions.npy")[t + self.predict_offset]

    def _load_inputs(self, i):
        """Processor-agnostic part: (PIL image, prompt str, action tokens (7,))."""
        traj, t = self.samples[i]
        img_files = sorted((traj / "images").glob("rgb_*.png"))
        image = Image.open(img_files[t]).convert("RGB")

        instrs = (traj / "instructions.txt").read_text().strip().splitlines()
        instruction = random.choice(instrs) if instrs else "navigate to the goal"
        prompt = PROMPT.format(instruction=instruction)

        action = self._raw_target(traj, t)                               # (4,)
        norm = normalize_action(action, self.norm_stats)                 # (4,) in [-1,1]
        action7 = drone_to_openvla(norm)                                 # (7,)
        action_tokens = self.action_tokenizer(action7).astype(np.int64)  # (7,)
        return image, prompt, action_tokens

    def _assemble(self, prompt_ids, action_tokens):
        """prompt_ids (P,) + 7 action tokens + EOS -> (input_ids, labels); the
        prompt positions are masked (-100) so only the action is supervised."""
        act_ids = torch.from_numpy(action_tokens)
        eos = torch.tensor([self.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, act_ids, eos])
        labels = torch.cat([
            torch.full((len(prompt_ids),), -100, dtype=torch.long),
            act_ids, eos,
        ])
        return input_ids, labels

    def __getitem__(self, i):
        image, prompt, action_tokens = self._load_inputs(i)
        enc = self.processor(text=prompt, images=image, return_tensors="pt")
        prompt_ids = enc["input_ids"][0]                  # (P,)
        input_ids, labels = self._assemble(prompt_ids, action_tokens)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": enc["pixel_values"][0],       # (6, 224, 224)
            "attention_mask": torch.ones_like(input_ids),
        }


class _TokenizerOnly:
    """Shim so OpenVLADroneDataset.__init__ can read .tokenizer.eos_token_id when
    handed a bare tokenizer (Prismatic path) instead of an HF processor."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class PrismaticDroneDataset(OpenVLADroneDataset):
    """Same supervised examples as OpenVLADroneDataset, but pixel_values come from
    Prismatic's dinosiglip image_transform (a dict {"dino","siglip"} of
    (3,224,224) tensors) rather than an HF processor's stacked (6,224,224).
    Used by the `--init prismatic` control arm (build_prismatic_policy)."""

    def __init__(self, root, tokenizer, image_transform, action_tokenizer,
                 norm_stats, split: str = "train", train_split: float = 0.9,
                 seed: int = 42, predict_offset: int = 0,
                 target_mode: str = "velocity", waypoint_horizon: int = 8):
        super().__init__(root, _TokenizerOnly(tokenizer), action_tokenizer,
                         norm_stats, split=split, train_split=train_split,
                         seed=seed, predict_offset=predict_offset,
                         target_mode=target_mode, waypoint_horizon=waypoint_horizon)
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __getitem__(self, i):
        image, prompt, action_tokens = self._load_inputs(i)
        pixel_values = self.image_transform(image)        # dict of (3,224,224)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        input_ids, labels = self._assemble(prompt_ids, action_tokens)
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


def make_prismatic_collate(pad_token_id: int):
    """Like make_openvla_collate, but pixel_values is a dict of tensors (Prismatic
    dinosiglip {"dino","siglip"}); stack each key into a batch tensor."""
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
        pixel_values = {
            k: torch.stack([b["pixel_values"][k] for b in batch])
            for k in batch[0]["pixel_values"]
        }
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
            "pixel_values": pixel_values,
        }
    return collate
