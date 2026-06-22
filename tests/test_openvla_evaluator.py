"""
CPU test for the OpenVLA evaluator's scoring math (_score_tokens). No model or
weights needed — uses a fake tokenizer and synthetic drone norm stats.

Run: python tests/test_openvla_evaluator.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from models.action_tokenizer import ActionTokenizer, drone_to_openvla
from data.openvla_dataset import normalize_action
from eval.openvla_evaluator import _score_tokens


class FakeTokenizer:
    vocab_size = 32000   # Llama-2


# A plausible per-dim drone action range (vx,vy,vz in m/s, yaw_rate in rad/s).
STATS = {
    "q01": [-3.0, -2.0, -1.5, -2.5],
    "q99": [4.0, 2.0, 1.5, 2.5],
}


def _tokens_for(action4):
    """Drone 4-DoF -> normalized 7-DoF -> 7 action-token ids."""
    norm = normalize_action(np.asarray(action4, dtype=np.float32), STATS)
    return ActionTokenizer(FakeTokenizer())(drone_to_openvla(norm)).astype(np.int64)


def test_perfect_prediction_zero_error():
    atok = ActionTokenizer(FakeTokenizer())
    ids = _tokens_for([2.0, -1.0, 0.5, 1.0])
    n_correct, err, l2 = _score_tokens(ids, ids, atok, STATS)
    assert n_correct == 7, n_correct
    # Identical tokens decode identically -> exactly zero error.
    assert l2 == 0.0 and np.allclose(err, 0.0), (err, l2)
    print(f"[ok] perfect prediction: acc 7/7, l2={l2:.4f}")


def test_one_bin_offset_small_bounded_error():
    atok = ActionTokenizer(FakeTokenizer())
    g = _tokens_for([2.0, -1.0, 0.5, 1.0])
    p = g.copy()
    p[0] = g[0] - 1   # shift the x/vx token by one bin (ids decrease as value rises)
    n_correct, err, l2 = _score_tokens(p, g, atok, STATS)
    assert n_correct == 6, n_correct
    # Only vx differs; its error is ~one bin in physical units, others zero.
    vx_bin = (STATS["q99"][0] - STATS["q01"][0]) * (2.0 / 255) / 2.0
    assert err[0] <= (STATS["q99"][0] - STATS["q01"][0]) / 255 + 1e-6, err
    assert np.allclose(err[1:], 0.0), err
    assert l2 > 0.0
    print(f"[ok] one-bin offset: acc 6/7, vx_err={err[0]:.4f} (~bin {vx_bin:.4f})")


def test_neutral_dims_do_not_affect_drone_error():
    # roll/pitch/gripper (idx 3,4,6) are held neutral; corrupting those tokens
    # must not change the 4-DoF drone error (only the drone dims are scored).
    atok = ActionTokenizer(FakeTokenizer())
    g = _tokens_for([2.0, -1.0, 0.5, 1.0])
    p = g.copy()
    p[3] = g[3] - 10   # roll token (non-drone dim)
    n_correct, err, l2 = _score_tokens(p, g, atok, STATS)
    assert np.allclose(err, 0.0) and l2 == 0.0, (err, l2)
    assert n_correct == 6   # token acc still counts the corrupted position
    print("[ok] corrupting a neutral (non-drone) dim leaves drone error at zero")


if __name__ == "__main__":
    test_perfect_prediction_zero_error()
    test_one_bin_offset_small_bounded_error()
    test_neutral_dims_do_not_affect_drone_error()
    print("\nALL OPENVLA-EVALUATOR TESTS PASSED")
