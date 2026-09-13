"""
CPU test for the OpenVLA evaluator's scoring math (_score_tokens). No model or
weights needed — uses a fake tokenizer and synthetic drone norm stats.

Run: python tests/test_openvla_evaluator.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.action_tokenizer import ActionTokenizer, drone_to_openvla
from data.openvla_dataset import normalize_action
from eval.openvla_evaluator import _score_tokens, evaluate_openvla


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


# ---------------------------------------------------------------------------
# Logits/labels alignment.
#
# The prismatic arm splices ~256 image-patch embeddings INTO the sequence, so
# its logits run longer than the text-stream labels, while the HF OpenVLA path
# returns logits aligned to input_ids. The scorer's manual causal shift silently
# read predictions out of the patch region and reported action_token_accuracy of
# exactly 0.0000 — BELOW the ~1/256 chance rate — for a model that had trained
# to a healthy loss. These two tests pin both offsets.
# ---------------------------------------------------------------------------

class _FakeOut:
    def __init__(self, logits, loss):
        self.logits, self.loss = logits, loss


class _FakeModel:
    """Emits logits that argmax to `gold`, optionally prefixed by `n_patches`
    positions of junk — mimicking a VLM that splices image tokens into the
    sequence."""

    def __init__(self, gold_ids, vocab, n_patches):
        self.gold_ids, self.vocab, self.n_patches = gold_ids, vocab, n_patches

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None,
                 pixel_values=None, labels=None):
        B, T = labels.shape
        logits = torch.zeros(B, T + self.n_patches, self.vocab)
        # Junk region: argmaxes to token 0, never a valid action token.
        logits[:, :self.n_patches, 0] = 10.0
        # Text region: position i-1 predicts labels[:, i] (causal shift).
        for b in range(B):
            for i in range(1, T):
                logits[b, self.n_patches + i - 1, int(labels[b, i])] = 10.0
        return _FakeOut(logits, torch.tensor(1.234))


def _alignment_case(n_patches):
    atok = ActionTokenizer(FakeTokenizer())
    ids = _tokens_for([2.0, -1.0, 0.5, 1.0])
    # [BOS] + 7 action tokens + EOS; only the action tokens are supervised.
    labels = torch.full((2, 9), -100, dtype=torch.long)
    for b in range(2):
        labels[b, 1:8] = torch.from_numpy(ids)
    batch = {
        "input_ids": labels.clone(),
        "attention_mask": torch.ones_like(labels),
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "labels": labels,
    }
    model = _FakeModel(ids, FakeTokenizer.vocab_size, n_patches)
    return evaluate_openvla(model, [batch], atok, STATS, torch.device("cpu"),
                            amp_enabled=False)


def test_alignment_no_patches_scores_perfectly():
    m = _alignment_case(n_patches=0)
    assert m["action_token_accuracy"] == 1.0, m["action_token_accuracy"]
    assert m["action_l2"] == 0.0, m["action_l2"]
    print("[ok] offset 0 (HF OpenVLA path): acc 1.0000")


def test_alignment_with_spliced_patches_scores_perfectly():
    # THE REGRESSION: before the fix this scored exactly 0.0000, because the
    # scorer read its predictions from the 256 junk patch positions.
    m = _alignment_case(n_patches=256)
    assert m["action_token_accuracy"] == 1.0, (
        f"spliced-patch logits mis-scored: {m['action_token_accuracy']:.4f} "
        "(0.0000 means the scorer is reading the patch region again)")
    assert m["action_l2"] == 0.0, m["action_l2"]
    assert abs(m["val_loss"] - 1.234) < 1e-5, m["val_loss"]
    print("[ok] offset 256 (prismatic path): acc 1.0000, val_loss passed through")


if __name__ == "__main__":
    test_perfect_prediction_zero_error()
    test_one_bin_offset_small_bounded_error()
    test_neutral_dims_do_not_affect_drone_error()
    test_alignment_no_patches_scores_perfectly()
    test_alignment_with_spliced_patches_scores_perfectly()
    print("\nALL OPENVLA-EVALUATOR TESTS PASSED")
