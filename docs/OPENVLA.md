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

## Target: waypoint vs velocity (single-frame observability)

OpenVLA observes **one** RGB frame. Instantaneous body-frame velocity
`[vx,vy,vz,yaw_rate]` is **not recoverable from a single image** — motion is
invisible in one frame — so velocity targets make this an ill-posed regression
(the expert's speed through a given scene aliases, and the model regresses to a
blurred conditional mean). Pipeline B sidesteps this with an 8-frame history;
OpenVLA cannot.

The fix is to change the *target*, not the architecture. `target_mode: waypoint`
(default in `configs/openvla.yaml`) predicts the **body-frame displacement to the
pose `waypoint_horizon` frames ahead** `[dx,dy,dz,dyaw]`, derived from `poses.npy`:

- **Well-posed from a single frame** — "where should I go next" is largely
  determined by the visible scene (gate ahead-left → waypoint ahead-left), unlike
  instantaneous speed.
- **Semantically matches OpenVLA's native action** — OpenVLA's `[x,y,z]` are
  *relative EEF translation deltas*, i.e. displacement waypoints. So a body-frame
  position delta is far closer to the pretrained action prior than a velocity is;
  this should *strengthen* transfer, not just patch observability.
- **No `dt`** — a displacement, not a finite-difference rate, so it avoids the
  timestamp-noise sensitivity of velocity.

`target_mode: velocity` is retained for comparison. **Waypoint mode needs
`poses.npy`**, emitted by a (re-)run of `scripts/convert_uzh_fpv.py`; existing
velocity-only datasets must be reconverted.

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

### Scoring the arms

Training produces loss curves; the ablation is read on the **val split** with
`scripts/eval_openvla.py` (teacher-forced action-token accuracy + decoded
drone-action error):

```bash
python scripts/eval_openvla.py --config configs/openvla.yaml \
    --init pretrained --ckpt outputs/openvla/pretrained/epoch005
# repeat with --init scratch and --init prismatic against their checkpoints
```

Reported per arm: `action_token_accuracy` (top-1 over the 7 action tokens,
quantization-free), `action_l2` (mean per-sample 4-DoF error), and `per_dim_mae`
for `[vx, vy, vz, yaw_rate]`. Both predicted and gold tokens are decoded through
the **drone** q01/q99 stats — not OpenVLA's `predict_action` denorm, which uses
the Open-X statistics and would report on the wrong scale.

**One command for the whole ablation.** `scripts/ablate_openvla.py` trains +
scores all three arms and prints a comparison table with an automatic verdict
(OpenVLA−Prismatic gap = robot-pretraining contribution; Prismatic−scratch gap =
VL-pretraining contribution):

```bash
python scripts/ablate_openvla.py --config configs/openvla.yaml
# or, when arms live in different envs, run subsets then aggregate:
#   --arms pretrained scratch          (OpenVLA env)
#   --arms prismatic                   (prismatic-vlms env)
#   --skip-train --skip-eval           (aggregate completed arms into one table)
```

### The clean control: Prismatic base (`--init prismatic`)

The `scratch` arm (random init + LoRA) only tests *whether pretraining helps at
all*, and it's a weak control: with a frozen random LM head, attention-only LoRA
can barely learn the readout, so a win is almost guaranteed and uninformative.

The proper control finetunes the **Prismatic VLM base** `prism-dinosiglip-224px+7b`
— the *same* DINOv2+SigLIP vision + Llama-2-7B as OpenVLA, with the *same*
vision-language pretraining, but **never robot/action-trained** — using the
**same** action-token LoRA recipe:

```bash
# Clean control: VL-pretrained, robot-naive base + same LoRA
python scripts/train_openvla.py --config configs/openvla.yaml --init prismatic
```

How to read the three arms:

- **OpenVLA ≫ Prismatic ≈ scratch** → the gain is *robot* pretraining
  specifically. **Supports** the cross-embodiment claim.
- **OpenVLA ≈ Prismatic ≫ scratch** → the gain is generic vision-language
  features, not robot transfer. **Falsifies** the headline framing.

Only the Prismatic arm can distinguish these, so it is the result that matters
for a writeup. `scratch` cannot.

**Environment / hardware.** Needs the Prismatic training repo (not plain HF
AutoModel) and a >=24GB GPU (bf16 full backbone; no 4-bit on the custom wrapper):

```bash
pip install git+https://github.com/TRI-ML/prismatic-vlms
```

The action-token scheme, drone normalization, and prompt/label masking are
reused unchanged (Prismatic's LLM is Llama-2-7B, so the 256 action token ids
carry over); only the pixel format differs (a dinosiglip `{"dino","siglip"}`
dict vs OpenVLA's stacked 6-channel tensor — handled by
`PrismaticDroneDataset`).

## Verification status

- **CPU-verified in the main env** (`python tests/test_action_tokenizer.py`,
  `python tests/test_openvla_dataset.py`, `python tests/test_openvla_evaluator.py`,
  `python tests/test_waypoint.py`): action discretization round-trip,
  drone<->7-DoF embedding, q01/q99 normalization, prompt/label masking, batched
  shapes, the evaluator's decode/scoring math, and the body-frame waypoint
  derivation — all with the real OpenVLA tokenizer (the evaluator/waypoint tests
  use a fake one / pure geometry, no download).
- **Pending lab-GPU validation** (needs the OpenVLA env + weights):
  `models/openvla_policy.py` model load + LoRA, and the `model(**batch).loss`
  training step in `scripts/train_openvla.py`. Syntax-checked only.
- **The `--init prismatic` control is unvalidated** and written against the
  Prismatic API (`prismatic.load`, `vlm.llm_backbone`, `vlm.vision_backbone`).
  Before a run, verify on the lab GPU: (1) the exact model id in the
  TRI-ML/prismatic-vlms registry, (2) that `forward(...)` accepts `labels`
  aligned to the text stream and masks inserted image tokens internally, and
  (3) the batched dinosiglip `pixel_values` dict signature.

## Weights

`openvla/openvla-7b` (MIT). On Windows, the HF cache symlink step needs
Developer Mode/admin; otherwise download with
`huggingface-cli download openvla/openvla-7b --local-dir <path>` and load from
that path.
