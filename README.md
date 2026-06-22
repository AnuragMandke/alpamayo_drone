# Alpamayo-Drone: Cross-Embodiment VLA Finetuning for UAV Navigation

**Thesis:** a vision-language-action (VLA) model pretrained on *ground*-robot
tasks can transfer to *aerial* navigation through lightweight finetuning alone —
without redesigning the architecture.

The project is named for NVIDIA's **Alpamayo-R1** driving VLA (the inspiration
for the cross-embodiment idea; its weights were released publicly in Dec 2025 as
`nvidia/Alpamayo-R1-10B`). The transfer experiments here use **OpenVLA-7B**, an
open ground-robot VLA, as the pretrained base on UZH-FPV drone-racing data.

The drone action space is 4-DoF velocity control: `[vx, vy, vz, yaw_rate]`.

---

## Two pipelines

The repo contains two finetuning approaches that share the same UZH-FPV dataset:

| | **OpenVLA transfer** (primary) | **Qwen + FlowMatchingDecoder** |
|---|---|---|
| Backbone | OpenVLA-7B (robot-pretrained VLA) | Qwen2.5-3B-Instruct (LLM) + frozen ViT |
| Action head | OpenVLA's **native** action tokens (reused prior) | FlowMatchingDecoder trained from scratch |
| What transfers | vision-language **and** action prior | vision-language representation only |
| Loss | cross-entropy on action tokens | conditional flow matching |
| Config | `configs/openvla.yaml` | `configs/default.yaml` |
| Entry | `scripts/train_openvla.py` | `scripts/train.py` |
| Env | `requirements-openvla.txt` (transformers 4.40) | `requirements.txt` (transformers 5.x) |
| Tests the claim | **directly** (reuses the robot action prior) | partially (representation transfer) |

The OpenVLA pipeline is the proper test of the cross-embodiment claim; the
Qwen + FlowMatchingDecoder pipeline is kept as an alternative-head comparison.
See [docs/OPENVLA.md](docs/OPENVLA.md) for the OpenVLA pipeline in detail.

---

## Project structure

```
alpamayo_drone/
├── configs/
│   ├── default.yaml          # Qwen + FlowMatchingDecoder pipeline
│   ├── openvla.yaml          # OpenVLA transfer pipeline
│   └── smoke_test.yaml       # tiny local backbone, fast end-to-end check
├── data/
│   ├── airsim_dataset.py     # sliding-window dataset (8-frame history, action horizon)
│   └── openvla_dataset.py    # OpenVLA single-frame, action-token dataset
├── models/
│   ├── alpamayo.py           # Qwen+ViT+FlowMatchingDecoder assembly
│   ├── flow_matching.py      # FlowMatchingDecoder (CFM action head)
│   ├── lora.py               # hand-rolled LoRA (works with bitsandbytes 4-bit)
│   ├── action_tokenizer.py   # OpenVLA action discretization + 4↔7-DoF mapping
│   └── openvla_policy.py     # OpenVLA load + LoRA (transfer/scratch arms)
├── training/
│   └── trainer.py            # training loop for the Qwen+FlowMatching pipeline
├── eval/
│   └── evaluator.py          # offline (L2/ADE/FDE) + online AirSim evaluators
├── scripts/
│   ├── train.py              # train: Qwen + FlowMatchingDecoder
│   ├── train_openvla.py      # train: OpenVLA transfer (lab GPU, OpenVLA env)
│   ├── eval.py               # evaluate the Qwen+FlowMatching pipeline
│   ├── ablate.py             # LoRA-rank and decoder-type ablations
│   ├── integration_test.py   # one-batch forward/backward gate
│   ├── download_uzh_fpv.py   # download UZH-FPV (hardened, resumable)
│   ├── convert_uzh_fpv.py    # UZH-FPV → trajectory format (both raw layouts)
│   └── download_data.py      # synthetic AirSim trajectories (smoke tests only)
├── tests/                    # CPU unit tests (action tokenizer, OpenVLA dataset)
├── docs/OPENVLA.md           # OpenVLA pipeline: env, run, ablation design
├── requirements.txt          # main env
└── requirements-openvla.txt  # isolated OpenVLA env (transformers 4.40)
```

---

## Data

Real data is **UZH-FPV** drone-racing sequences (snapdragon, with ground truth),
converted to per-trajectory `images/ + actions.npy + instructions.txt`. Body-frame
velocity actions are derived from the ground-truth poses.

```bash
python scripts/download_uzh_fpv.py --out data/raw/uzh_fpv          # ~12 GB, resumable
python scripts/convert_uzh_fpv.py  --src data/raw/uzh_fpv --dst data/uzh_fpv --chunk_size 120
# -> data/uzh_fpv/: 118 trajectories, ~14k frames (11.4k train / 1.3k val samples)
```

For a no-download smoke test, `scripts/download_data.py` generates synthetic
trajectories under `data/airsim/`. Datasets are git-ignored (regenerable).

---

## Pipeline A — OpenVLA transfer (primary)

Runs in an **isolated environment** on a **≥16 GB GPU** (OpenVLA's remote code
requires transformers 4.40, incompatible with the main env's 5.x).

```bash
python -m venv .venv-openvla && .venv-openvla/Scripts/activate   # Windows
pip install -r requirements-openvla.txt

# Headline cross-embodiment ablation:
python scripts/train_openvla.py --config configs/openvla.yaml --init pretrained  # transfer arm
python scripts/train_openvla.py --config configs/openvla.yaml --init scratch     # weak control
python scripts/train_openvla.py --config configs/openvla.yaml --init prismatic   # clean control
```

The claim is supported if **pretrained + LoRA** beats from-scratch on drone
action accuracy at a fraction of the trainable parameters. The `prismatic` arm
(VL-pretrained but robot-naive, same architecture + LoRA) is the control that
isolates *robot* pretraining and is the only one that can falsify the claim —
full design in [docs/OPENVLA.md](docs/OPENVLA.md).

---

## Pipeline B — Qwen + FlowMatchingDecoder

Runs in the main env; fits a 6 GB GPU (4-bit QLoRA).

```bash
pip install -r requirements.txt

# Fast end-to-end sanity check (tiny local backbone, no large download):
python scripts/integration_test.py --config configs/smoke_test.yaml
python scripts/train.py --config configs/smoke_test.yaml

# Real run:
python scripts/train.py --config configs/default.yaml
python scripts/eval.py  --config configs/default.yaml --ckpt outputs/<run>/<ckpt> --mode offline
```

**Design:** frozen ViT-B/16 encodes each frame → projected and prepended to the
Qwen hidden states as cross-attention memory for a FlowMatchingDecoder that emits
4 future actions. Only LoRA (rank 16) + the vision projection + the decoder train
(~0.4% of params); the backbone is 4-bit quantized and otherwise frozen. Actions
are normalized per-dim; the decoder cross-attention masks text padding.

---

## Status

- **Pipeline B** is end-to-end verified on the dev laptop (smoke train + offline
  eval pass; full UZH-FPV run is a standard GPU job).
- **Pipeline A** data layer and the offline scorer's decode/scoring math are
  CPU-verified (`python tests/test_action_tokenizer.py`,
  `python tests/test_openvla_dataset.py`, `python tests/test_openvla_evaluator.py`).
  The three training arms (`--init pretrained|scratch|prismatic`) and the
  checkpoint-loading paths in `scripts/eval_openvla.py` are written against the
  OpenVLA/Prismatic APIs but run on the lab GPU in the matching env — they cannot
  be exercised on a 6 GB laptop.
