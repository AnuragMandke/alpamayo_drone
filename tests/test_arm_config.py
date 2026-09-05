"""
CPU test for the per-arm training-config resolver (no data/model needed).

The three ablation arms must differ ONLY in backbone init. On a 24GB card they
cannot share a per-step batch (4-bit base vs two full bf16 backbones), so
`training.per_arm` reshapes batch_size/grad_accum per arm — but the EFFECTIVE
batch must stay identical, or every cross-arm number is confounded. These tests
pin that invariant, plus the real configs/openvla.yaml satisfying it.

Run: python tests/test_arm_config.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from models.openvla_policy import resolve_arm_training_cfg

ARMS = ["pretrained", "scratch", "prismatic"]

BASE = {
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "seed": 42,
    "per_arm": {
        "scratch": {"batch_size": 4, "gradient_accumulation_steps": 4},
        "prismatic": {"batch_size": 2, "gradient_accumulation_steps": 8},
    },
}


def eff(tc):
    return tc["batch_size"] * tc["gradient_accumulation_steps"]


def test_arm_without_override_is_unchanged():
    tc = resolve_arm_training_cfg(BASE, "pretrained")
    assert tc["batch_size"] == 8 and tc["gradient_accumulation_steps"] == 2, tc
    print("[ok] an arm with no per_arm entry keeps the config defaults")


def test_override_reshapes_but_preserves_effective_batch():
    effs = {arm: eff(resolve_arm_training_cfg(BASE, arm)) for arm in ARMS}
    assert len(set(effs.values())) == 1, effs
    shapes = {arm: resolve_arm_training_cfg(BASE, arm)["batch_size"] for arm in ARMS}
    assert shapes == {"pretrained": 8, "scratch": 4, "prismatic": 2}, shapes
    print(f"[ok] per-arm batch {shapes} -> one effective batch {set(effs.values()).pop()}")


def test_unmatched_effective_batch_is_rejected():
    # The failure this guard exists for: halving the batch for memory and
    # FORGETTING to double the accumulation, so the control arm quietly trains on
    # half as many samples per step as the transfer arm.
    bad = dict(BASE, per_arm={"scratch": {"batch_size": 4,
                                          "gradient_accumulation_steps": 2}})
    try:
        resolve_arm_training_cfg(bad, "scratch")
    except ValueError as e:
        assert "effective batch" in str(e), e
        print("[ok] an override that changes the effective batch raises")
    else:
        raise AssertionError("unmatched effective batch was accepted")


def test_only_memory_shape_keys_may_be_overridden():
    # per_arm must not become a back door for per-arm hyperparameters: a
    # different lr or epoch count per arm would make the arms differ by more
    # than their init, which is the one thing the ablation forbids.
    bad = dict(BASE, per_arm={"scratch": {"optimizer": {"lr": 1e-5}}})
    try:
        resolve_arm_training_cfg(bad, "scratch")
    except ValueError as e:
        assert "batch_size" in str(e), e
        print("[ok] a non-memory key in per_arm raises")
    else:
        raise AssertionError("a disallowed per_arm key was accepted")


def test_shipped_lab_config_is_matched_across_all_three_arms():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(here, "configs", "openvla.yaml")))
    effs = {arm: eff(resolve_arm_training_cfg(cfg["training"], arm)) for arm in ARMS}
    assert len(set(effs.values())) == 1, effs
    print(f"[ok] configs/openvla.yaml: all three arms at effective batch "
          f"{set(effs.values()).pop()}")


if __name__ == "__main__":
    test_arm_without_override_is_unchanged()
    test_override_reshapes_but_preserves_effective_batch()
    test_unmatched_effective_batch_is_rejected()
    test_only_memory_shape_keys_may_be_overridden()
    test_shipped_lab_config_is_matched_across_all_three_arms()
    print("\nALL ARM-CONFIG TESTS PASSED")
