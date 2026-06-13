"""
CPU test for the OpenVLA drone dataset: builds a real batch with OpenVLA's
processor/tokenizer (weights NOT required) and checks shapes + that the
action-token labels decode back to the original drone action.

Run: python tests/test_openvla_dataset.py
Requires the OpenVLA tokenizer/processor files in the HF cache (small files).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer

from models.action_tokenizer import ActionTokenizer, DRONE_TO_OPENVLA_IDX
from data.openvla_dataset import (
    OpenVLADroneDataset, compute_drone_norm_stats, make_openvla_collate,
    normalize_action, denormalize_action,
)
from torch.utils.data import DataLoader

ROOT = "data/uzh_fpv"


class ShimProcessor:
    """
    Minimal stand-in for PrismaticProcessor to validate the dataset adapter on
    the dev env (OpenVLA's real image processor needs transformers~=4.40; see
    requirements-openvla.txt). Uses the REAL OpenVLA tokenizer so action tokens
    and input_ids are exactly what training will see; pixel_values are a
    correctly-shaped (6,224,224) placeholder.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, return_tensors="pt"):
        enc = self.tokenizer(text, return_tensors=return_tensors)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pixel_values": torch.zeros(1, 6, 224, 224),
        }


def main():
    tok = AutoTokenizer.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    proc = ShimProcessor(tok)
    atok = ActionTokenizer(tok)
    print(f"[ok] tokenizer loaded (vocab={tok.vocab_size}); using ShimProcessor "
          f"for pixel_values")

    stats = compute_drone_norm_stats(ROOT)
    print(f"[ok] drone norm stats q01={np.round(stats['q01'],2)} "
          f"q99={np.round(stats['q99'],2)}")

    # normalize/denormalize round-trip (within [q01,q99])
    a = np.array([stats["mean"]], dtype=np.float32)
    rt = denormalize_action(normalize_action(a, stats), stats)
    assert np.allclose(a, rt, atol=1e-2), f"norm round-trip off: {a} vs {rt}"
    print("[ok] normalize/denormalize round-trip")

    ds = OpenVLADroneDataset(ROOT, proc, atok, stats, split="train")
    s = ds[0]
    assert s["pixel_values"].shape[-2:] == (224, 224)
    assert s["pixel_values"].shape[0] in (3, 6), s["pixel_values"].shape
    # last 8 label positions = 7 action tokens + EOS; prompt is masked (-100)
    assert (s["labels"][:-8] == -100).all(), "prompt positions must be masked"
    assert (s["labels"][-8:-1] != -100).all(), "action tokens must be supervised"
    print(f"[ok] sample shapes: pixel_values {tuple(s['pixel_values'].shape)}, "
          f"input_ids {tuple(s['input_ids'].shape)}")

    # Decode the supervised action tokens -> should recover the stored action
    traj, t = ds.samples[0]
    raw = np.load(traj / "actions.npy")[t]
    action_token_ids = s["labels"][-8:-1].numpy()
    recon_norm7 = atok.decode_token_ids_to_actions(action_token_ids)
    recon_raw = denormalize_action(recon_norm7[DRONE_TO_OPENVLA_IDX], stats)
    err = np.abs(recon_raw - raw).max()
    bin_span = (np.asarray(stats["q99"]) - np.asarray(stats["q01"])).max() / 255
    assert err <= bin_span + 1e-3, f"action recon err {err:.4f} > bin span {bin_span:.4f}"
    print(f"[ok] action token round-trip: raw={np.round(raw,3)} "
          f"recon={np.round(recon_raw,3)} (err {err:.4f} <= bin {bin_span:.4f})")

    # Collate a batch
    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=make_openvla_collate(proc.tokenizer.pad_token_id or 0))
    batch = next(iter(loader))
    assert batch["pixel_values"].shape[0] == 4
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    print(f"[ok] batched: input_ids {tuple(batch['input_ids'].shape)}, "
          f"pixel_values {tuple(batch['pixel_values'].shape)}")

    print("\nALL OPENVLA-DATASET TESTS PASSED")


if __name__ == "__main__":
    main()
