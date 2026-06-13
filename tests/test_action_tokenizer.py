"""
CPU round-trip tests for the OpenVLA-compatible ActionTokenizer.

Run: python tests/test_action_tokenizer.py
Uses a tiny fake tokenizer (just needs .vocab_size), so no model download.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.action_tokenizer import (
    ActionTokenizer, drone_to_openvla, openvla_to_drone,
    OPENVLA_ACTION_DIM, DRONE_TO_OPENVLA_IDX,
)


class FakeTokenizer:
    vocab_size = 32000   # Llama-2


def test_round_trip_within_bin_width():
    tok = ActionTokenizer(FakeTokenizer(), bins=256)
    bin_width = 2.0 / 255            # span [-1,1] over 255 intervals
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=(1000, 7)).astype(np.float32)
    ids = tok(a)
    recon = tok.decode_token_ids_to_actions(ids)
    err = np.abs(recon - a).max()
    assert err <= bin_width, f"max round-trip error {err:.5f} > bin width {bin_width:.5f}"
    print(f"[ok] round-trip max err {err:.5f} <= bin width {bin_width:.5f}")


def test_token_ids_in_reserved_range():
    tok = ActionTokenizer(FakeTokenizer(), bins=256)
    lo, hi = tok.action_token_id_range
    ids = tok(np.random.default_rng(1).uniform(-1, 1, size=(500, 7)).astype(np.float32))
    assert ids.min() >= lo and ids.max() <= hi, f"ids escape reserved range [{lo},{hi}]"
    assert (lo, hi) == (32000 - 256, 31999)
    print(f"[ok] all action token ids within reserved range [{lo}, {hi}]")


def test_clipping_out_of_range():
    tok = ActionTokenizer(FakeTokenizer(), bins=256)
    # Values beyond [-1,1] must clip to the extreme bins, not wrap or crash.
    a = np.array([[-5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    recon = tok.decode_token_ids_to_actions(tok(a))
    assert recon[0, 0] <= -0.9 and recon[0, 1] >= 0.9
    print(f"[ok] out-of-range clips to extremes: {recon[0,:2]}")


def test_drone_embedding_roundtrip():
    a4 = np.random.default_rng(2).uniform(-1, 1, size=(10, 4)).astype(np.float32)
    a7 = drone_to_openvla(a4)
    assert a7.shape[-1] == OPENVLA_ACTION_DIM
    # Unused dims (roll=3, pitch=4, gripper=6) are zero
    for i in range(7):
        if i not in DRONE_TO_OPENVLA_IDX:
            assert np.allclose(a7[..., i], 0.0)
    assert np.allclose(openvla_to_drone(a7), a4)
    print("[ok] drone<->openvla 4<->7 embedding is exact, neutral dims zero")


if __name__ == "__main__":
    test_round_trip_within_bin_width()
    test_token_ids_in_reserved_range()
    test_clipping_out_of_range()
    test_drone_embedding_roundtrip()
    print("\nALL ACTION-TOKENIZER TESTS PASSED")
