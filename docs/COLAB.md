# Running the OpenVLA transfer arm on Colab (T4)

A free Colab **T4 (16 GB, Turing)** can run the **robot-pretrained OpenVLA-7B
transfer arm** in 4-bit as a smoke test — enough to prove the model loads and
trains a few steps without OOM/dtype errors, and to burn down the *pending
lab-GPU validation* blocker before booking the real GPU.

> **Scope.** This is a ~50-step smoke run in `velocity` mode. It is **not** a
> result. The `scratch` and `prismatic` control arms need **>=24 GB bf16** and
> will not fit a T4 — run the real 3-arm ablation on the lab GPU with
> `configs/openvla.yaml` (see [OPENVLA.md](OPENVLA.md)).

There are two ways to run it: the **notebook** (easiest) or the **manual
commands** below (same steps, if you prefer to paste cells yourself).

---

## Why a T4 needs special handling

| Constraint | Consequence |
|---|---|
| Turing has **no bf16 tensor cores** | Must train in **fp16**. `configs/openvla_colab.yaml` sets `precision: fp16`; the trainer adds an fp16 GradScaler automatically. |
| 16 GB VRAM | Only the **4-bit pretrained** arm fits. `scratch`/`prismatic` (24 GB bf16) are lab-GPU only. |
| `data/uzh_fpv/` has no `poses.npy` | Smoke uses `target_mode: velocity`. Waypoint mode (the meaningful target) needs a local re-run of `scripts/convert_uzh_fpv.py` first. |
| Ephemeral disk, session limits | Weights (~15 GB) re-download every session; the `max_steps: 50` cap keeps the run short enough to finish in one. |

The dtype is flag-gated (`training.precision: bf16 | fp16 | fp32`), so this
changes nothing for the lab GPU — that path still defaults to bf16.

---

## Prerequisites (do these once, on your laptop)

**1. Push the repo.** The Colab steps clone `main` from GitHub, so the fp16
config/code must be pushed:

```bash
git push origin main
```

**2. Upload the data to Google Drive.** `data/uzh_fpv/` (572 MB) is git-ignored,
so it isn't in the clone. From the repo root:

```bash
tar czf uzh_fpv.tar.gz -C data uzh_fpv
```

Upload `uzh_fpv.tar.gz` to Google Drive at `MyDrive/alpamayo/uzh_fpv.tar.gz`
(any path works — just match it in the data step below).

---

## Option A — the notebook (recommended)

1. Open [`notebooks/openvla_colab_t4.ipynb`](../notebooks/openvla_colab_t4.ipynb)
   in Colab (GitHub → *Open in Colab*, or upload the `.ipynb`).
2. **Runtime → Change runtime type → T4 GPU.**
3. Run the cells top to bottom. It clones, installs, mounts Drive, extracts the
   data, and runs the smoke test.

---

## Option B — manual commands

Set **Runtime → Change runtime type → T4 GPU** first, then run each block in a
cell.

**1. Confirm the GPU is a T4**

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**2. Clone the repo**

```python
%cd /content
!rm -rf alpamayo_drone
!git clone https://github.com/AnuragMandke/alpamayo_drone.git
%cd /content/alpamayo_drone
```

**3. Install the OpenVLA deps** — OpenVLA's `trust_remote_code` needs
**transformers 4.40.1**. Do **not** reinstall torch/torchvision: Colab's build
is matched to the runtime CUDA, and replacing it from PyPI usually breaks the GPU.

```python
!pip install -q \
  "transformers==4.40.1" "tokenizers>=0.19,<0.20" "timm==0.9.10" \
  "accelerate>=0.29.0" "peft>=0.11.0" "bitsandbytes>=0.43.0" \
  "Pillow>=10.0.0" "PyYAML>=6.0"
```

If Colab had already imported `transformers`, do **Runtime → Restart session**
and re-run from step 2.

**4. Bring the data over from Drive**

```python
from google.colab import drive
drive.mount('/content/drive')

TARBALL = '/content/drive/MyDrive/alpamayo/uzh_fpv.tar.gz'  # edit if needed
!mkdir -p data
!tar xzf "$TARBALL" -C data
!ls data/uzh_fpv/trajectories | head -3
```

**5. Point the HF cache at local disk** (weights ~15 GB, ephemeral)

```python
import os
os.environ['HF_HOME'] = '/content/hf_cache'
```

**6. Run the smoke test**

```python
!python scripts/train_openvla.py --config configs/openvla_colab.yaml --init pretrained
```

---

## What success looks like

- `[Train] precision=torch.float16 (autocast=on)` near the top.
- Trainable-params line shows only LoRA adapters (<1 % of the model).
- A finite, non-NaN `loss=` that trends **down** over 50 steps, ending with
  `[max_steps=50 reached — stopping]` and a saved
  `outputs/openvla/pretrained/epoch001`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Lower `batch_size` to 2 (or 1) in `configs/openvla_colab.yaml`. |
| `loss=nan` | fp16 underflow; the GradScaler should catch it, but if it persists drop `optimizer.lr` and/or `batch_size`. |
| bf16 error inside the OpenVLA remote code | The remote `modeling_prismatic.py` may hardcode a bf16 path in the vision tower. That's exactly the finding this smoke test exists to surface — report it; the fix is a small patch forcing the vision dtype. |
| Session disconnects mid-download | Free Colab idles out — keep the tab active; the `max_steps` cap keeps the run short. |

## Config knobs (`configs/openvla_colab.yaml`)

| Key | Smoke value | Notes |
|---|---|---|
| `training.precision` | `fp16` | **T4-mandatory.** `bf16` on the lab GPU, `fp32` to debug. |
| `training.max_steps` | `50` | Optimizer-step cap; remove for a full epoch. |
| `training.batch_size` | `4` | Raise on bigger VRAM, lower on OOM. |
| `data.target_mode` | `velocity` | Switch to `waypoint` only after regenerating `poses.npy`. |
