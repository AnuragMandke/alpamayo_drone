# OpenVLA cross-embodiment transfer pipeline

Tests the project's core claim — *a VLA pretrained on ground-robot tasks
transfers to aerial navigation through lightweight finetuning alone* — using
**OpenVLA-7B** (pretrained on Open X-Embodiment manipulation data) finetuned on
UZH-FPV drone trajectories, **reusing OpenVLA's native action head**.

This pipeline is separate from the Qwen + FlowMatchingDecoder pipeline
(`configs/default.yaml`); that remains as an alternative-head comparison.

## Why a separate environment

OpenVLA's `trust_remote_code` files target **transformers 4.40.x**. They are
incompatible with the main env's transformers 5.x. Use an isolated venv:

```bash
python -m venv .venv-openvla
.venv-openvla/Scripts/activate          # Windows
# source .venv-openvla/bin/activate     # Linux
pip install -r requirements-openvla.txt
```

Hardware: **>=16 GB GPU** for the 4-bit pretrained arm, **>=24 GB** for the
from-scratch arm (full fine-tune, bf16). The 6 GB dev laptop cannot train this.

## Data

Reuses `data/uzh_fpv/` (built by `scripts/download_uzh_fpv.py` +
`scripts/convert_uzh_fpv.py`). Each timestep becomes one supervised example:
single RGB frame + `In: What action should the robot take to {instruction}?\nOut:`
+ 7 action tokens. Drone `[vx,vy,vz,yaw_rate]` maps into OpenVLA's 7-DoF
`[x,y,z,roll,pitch,yaw,gripper]` layout (roll/pitch/gripper held neutral),
normalized per-dim to [-1,1] via [q01,q99].

## The headline ablation

```bash
# Transfer arm: robot-pretrained OpenVLA + LoRA
python scripts/train_openvla.py --config configs/openvla.yaml --init pretrained

# Control arm: same architecture, random init, full fine-tune
python scripts/train_openvla.py --config configs/openvla.yaml --init scratch
```

The claim is supported if **pretrained + LoRA** reaches clearly better drone
action accuracy (and/or with far less data) than **from-scratch**, at a fraction
of the trainable parameters.

### Stronger control (recommended follow-up)

The `scratch` arm here (random init, full FT) tests *whether pretraining helps at
all*. To isolate the **robot** pretraining specifically — vs the underlying
vision-language pretraining — add a third arm that finetunes the **Prismatic VLM
base** (`prism-dinosiglip-224px`, VL-pretrained but never robot-trained) with the
same LoRA recipe. That requires the OpenVLA/Prismatic training repo
(`TRI-ML/prismatic-vlms`), not plain HF AutoModel. This is the cleanest test of
the cross-embodiment-transfer mechanism and the strongest result for a writeup.

## Verification status

- **CPU-verified in the main env** (`python tests/test_action_tokenizer.py`,
  `python tests/test_openvla_dataset.py`): action discretization round-trip,
  drone<->7-DoF embedding, q01/q99 normalization, prompt/label masking,
  batched shapes — all with the real OpenVLA tokenizer.
- **Pending lab-GPU validation** (needs the OpenVLA env + weights):
  `models/openvla_policy.py` model load + LoRA, and the `model(**batch).loss`
  training step in `scripts/train_openvla.py`. Syntax-checked only.

## Weights

`openvla/openvla-7b` (MIT). On Windows, the HF cache symlink step needs
Developer Mode/admin; otherwise download with
`huggingface-cli download openvla/openvla-7b --local-dir <path>` and load from
that path.
