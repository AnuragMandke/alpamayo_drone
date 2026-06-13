"""
models/action_tokenizer.py — OpenVLA-compatible action tokenization for drones.

OpenVLA emits actions as discrete tokens: each continuous action dim is
normalized to [-1, 1] (via dataset [1%, 99%] quantiles), uniformly binned into
256 bins, and each bin is mapped to one of the *last* 256 ids of the Llama-2
vocabulary (token_id = vocab_size - bin). Training is next-token cross-entropy
over these action-token positions; this reuses OpenVLA's pretrained action head.

We keep OpenVLA's native 7-DoF EEF-delta layout [x, y, z, roll, pitch, yaw,
gripper] and place the drone's 4-DoF velocity command into it:

    drone [vx, vy, vz, yaw_rate]  ->  OpenVLA [x=vx, y=vy, z=vz,
                                                roll=0, pitch=0, yaw=yaw_rate,
                                                gripper=0]

so the pretrained weights for the x/y/z/yaw action tokens carry over directly,
and the unused roll/pitch/gripper dims are supervised toward their neutral bin.

This module is pure numpy + a tokenizer; it has no heavy model dependency and
is fully unit-testable on CPU (see tests/test_action_tokenizer.py).
"""

import numpy as np

# OpenVLA's native action layout
OPENVLA_ACTION_DIM = 7
# Indices in the 7-DoF vector that the drone actually drives:
# x, y, z, yaw  (roll=3, pitch=4, gripper=6 are held neutral)
DRONE_TO_OPENVLA_IDX = [0, 1, 2, 5]


class ActionTokenizer:
    """
    Discretize/​de-discretize continuous actions to OpenVLA action tokens.

    Args:
        tokenizer:  the OpenVLA/Llama tokenizer (needs .vocab_size)
        bins:       number of discretization bins (OpenVLA uses 256)
        min_action / max_action: normalized action range (OpenVLA uses [-1, 1])
    """

    def __init__(self, tokenizer, bins: int = 256,
                 min_action: float = -1.0, max_action: float = 1.0):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        # Uniform bin edges and centers over [min, max]
        self.bins = np.linspace(min_action, max_action, bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    # ------------------------------------------------------------------

    def __call__(self, action: np.ndarray) -> np.ndarray:
        """
        Continuous (already-normalized) action -> action token ids.

        action: (..., A) float in roughly [min, max]
        returns: (..., A) int token ids in [vocab_size - bins, vocab_size - 1]
        """
        action = np.clip(action, self.min_action, self.max_action)
        # np.digitize returns 1..len(bins); OpenVLA maps that to vocab_size - idx
        disc = np.digitize(action, self.bins)
        return self.tokenizer.vocab_size - disc

    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """Inverse of __call__: action token ids -> bin-center continuous values."""
        disc = self.tokenizer.vocab_size - token_ids
        # disc is 1..bins; bin_centers is indexed 0..bins-2. Clamp to valid range
        # (mirrors OpenVLA, which shifts by one and clips at the boundary).
        idx = np.clip(disc - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[idx]

    @property
    def action_token_id_range(self) -> tuple:
        """[low, high] inclusive range of ids reserved for action tokens."""
        return (self.tokenizer.vocab_size - self.n_bins, self.tokenizer.vocab_size - 1)


def drone_to_openvla(action4: np.ndarray) -> np.ndarray:
    """Embed a drone 4-DoF action (..., 4) into OpenVLA's 7-DoF layout (..., 7)."""
    out = np.zeros(action4.shape[:-1] + (OPENVLA_ACTION_DIM,), dtype=np.float32)
    out[..., DRONE_TO_OPENVLA_IDX] = action4
    return out


def openvla_to_drone(action7: np.ndarray) -> np.ndarray:
    """Extract the drone 4-DoF action (..., 4) from OpenVLA's 7-DoF output."""
    return action7[..., DRONE_TO_OPENVLA_IDX]
