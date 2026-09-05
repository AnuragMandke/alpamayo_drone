# Alpamayo-Drone — Design & Decisions

This document explains what the project does, how it is built, and **why** each
significant decision was made — including the alternatives that were considered
and rejected. It is the "reasoning" companion to the `README.md` (which is the
"how to run" reference) and `docs/OPENVLA.md` (the primary pipeline's operational
guide).

---

## 1. The thesis

**Claim under test:** a vision-language-action (VLA) model pretrained on
*ground*-robot manipulation can transfer to *aerial* drone navigation through
lightweight finetuning alone — without redesigning the architecture or the
action head.

If true, this is a statement about **cross-embodiment transfer**: the visuomotor
prior a model learns from tabletop robot arms (Open X-Embodiment) contains
structure that is useful for flying a quadrotor, even though the embodiments,
cameras, dynamics, and action spaces differ. The whole project is an apparatus
for producing evidence for or against that claim, cleanly enough that the result
is interpretable.

The project is named after NVIDIA's **Alpamayo-R1** driving VLA, the inspiration
for the cross-embodiment framing. The transfer experiments themselves use
**OpenVLA-7B**, an openly-released ground-robot VLA, because Alpamayo-R1's
weights were not available when the work began.

The drone action space is 4-DoF: `[vx, vy, vz, yaw_rate]`.

---

## 2. Two pipelines, and why both exist

The repo contains two finetuning approaches over the same UZH-FPV data.

| | **Pipeline A — OpenVLA transfer** (primary) | **Pipeline B — Qwen + FlowMatchingDecoder** |
|---|---|---|
| Backbone | OpenVLA-7B (robot-pretrained VLA) | Qwen2.5-3B-Instruct (LLM) + frozen ViT |
| Action head | OpenVLA's **native** action tokens (reused prior) | FlowMatchingDecoder trained **from scratch** |
| What transfers | vision-language **and** the action prior | vision-language representation only |
| Loss | cross-entropy on action tokens | conditional flow matching |
| Tests the thesis | **directly** | partially (representation only) |

**Why Pipeline A is the real test.** The thesis is about reusing a *robot action
prior*. Only OpenVLA lets us keep the pretrained action head and ask whether it
transfers. That is the experiment.

**Why Pipeline B still exists.** It was built first, on a 6 GB laptop, before the
pivot to OpenVLA. It is kept as an **alternative-head comparison**: it transfers
only the vision-language representation and learns a brand-new action head, so
comparing it to Pipeline A separates "the representation transfers" from "the
action prior transfers." It also runs on hardware Pipeline A cannot, so it stays
useful as a fast sanity/dev pipeline. Deleting it would have thrown away a
working baseline for no gain — the pivot was **additive, not a rewrite**.

The rest of this document is mostly about Pipeline A.

---

## 3. Data — UZH-FPV

Real data is **UZH-FPV** drone-racing sequences (snapdragon camera, with
ground-truth poses), not simulation. Racing footage was chosen because it has
genuine 6-DoF motion with ground truth, which synthetic AirSim trajectories
(kept only for smoke tests) do not.

`scripts/convert_uzh_fpv.py` turns each raw sequence into per-trajectory chunks:

```
data/uzh_fpv/trajectories/traj_<seq>_t<i>/
    images/          rgb_000.png ...      (224×224 RGB)
    actions.npy      (T, 4)  [vx, vy, vz, yaw_rate]  body-frame velocity
    poses.npy        (T, 7)  world [x,y,z,qx,qy,qz,qw] per frame
    timestamps.npy   (T,)    per-frame time (s) on the ground-truth clock
    instructions.txt one natural-language goal per line
```

Actions are derived from the ground-truth poses by finite differencing into the
**body frame**. Two conversion details carry real weight:

- **Velocity clip = 20 m/s / 10 rad/s, not 5 m/s / π rad/s.** The clip exists
  only to guard against numerical spikes (finite-differencing a near-zero `dt`,
  or a GT glitch); it must sit *above* the true flight envelope. The original
  5 m/s ceiling was **below** what a racing drone actually flies, so it saturated
  ~16% of forward velocities onto a single value, which the normalizer then
  collapsed into one repeated action token — the model earned free loss by
  predicting a bin that shouldn't have existed. **Shaping the action
  distribution is the job of the q01/q99 normalizer, not the safety clip.**

- **Per-dim [1%, 99%] quantile normalization**, mirroring OpenVLA's
  `dataset_statistics`. Min/max would let a single outlier stretch the whole
  range; percentile normalization is robust to the spikes the clip is guarding
  against, and it matches the statistic OpenVLA's pretrained action tokens were
  trained under.

---

## 4. The central modelling problem: single-frame observability

OpenVLA observes **one** RGB frame per action. That constraint drives the two
most important design decisions in the project.

### 4.1 Why the target is a *waypoint*, not a velocity

Instantaneous body-frame velocity `[vx, vy, vz, yaw_rate]` is **not recoverable
from a single image** — motion is invisible in one static frame. Two runs
through the same corridor at 3 m/s and 8 m/s produce the *same pixels* but
different velocity labels, so a velocity target is an **ill-posed regression**:
the label aliases, and the model can do no better than regress to the blurred
conditional mean. Empirically this is exactly what happened — every arm converged
to the marginal action distribution (~2.58 nats), a **null result by
construction** regardless of pretraining.

The fix is to change the **target**, not the architecture:

**`target_mode: waypoint`** predicts the body-frame *displacement to a pose ahead*
`[dx, dy, dz, dyaw]`, derived from `poses.npy`. This is chosen for three
compounding reasons:

1. **It is well-posed from one frame.** "Where should I go next" is largely
   determined by the visible scene (gate ahead-left → waypoint ahead-left),
   unlike instantaneous speed.
2. **It matches OpenVLA's native action semantics.** OpenVLA's `[x, y, z]` are
   *relative end-effector translation deltas* — i.e. displacement waypoints. So a
   body-frame position delta is far closer to the pretrained action prior than a
   velocity is. This should *strengthen* transfer, not merely patch
   observability — it is the difference between reusing the action head and
   fighting it.
3. **No `dt`.** A displacement is not a finite-difference rate, so it sidesteps
   the timestamp-noise sensitivity that plagues velocity targets.

**Rejected alternative — give OpenVLA a frame history.** Pipeline B solves the
observability problem with an 8-frame history buffer. OpenVLA is architecturally
single-frame; bolting on history would mean redesigning the model, which
*defeats the thesis* ("through lightweight finetuning alone, without redesigning
the architecture"). Changing the target keeps the architecture untouched.

`target_mode: velocity` is retained, but only as a documented negative control —
it demonstrates the null result rather than hiding it.

### 4.2 Why a fixed *time* horizon, not a fixed *frame* horizon

The waypoint originally targeted the pose a fixed **8 frames** ahead. But UZH-FPV
**drops frames**: the median frame `dt` is 0.0339 s (≈29.5 Hz), but the p90 is
0.066 s and the worst observed 8-frame window spans 0.43 s versus the 0.27 s
nominal. So "8 frames ahead" is a **variable amount of time** ahead — an
irreducible source of target noise that caps how low any arm's loss can go.

The fix (added in this work) is to emit **per-frame timestamps** from the
converter and offer a **fixed-time** waypoint: target the pose nearest
`waypoint_horizon_seconds` (e.g. 0.27 s) ahead on the clock, snapping to whatever
actual frame is closest. Where frames were dropped, it uses 6–7 frames to span
the same time instead of a fixed 8. Measured effect on the full train split: the
fixed-time target range tightens (q99 on `dx` drops from 3.37 to 2.70) and the
marginal floor moves from 2.563 to 2.590 nats — the expected signature of
removing displacement inflation from over-long windows.

**Now on by default for the lab run (2026-09-05).** It was initially opt-in
because enabling it requires re-converting the dataset (older data has no
`timestamps.npy`) and the 2.201-nats probe was produced under the fixed-frame
target. Both 118 trajectories and the staged tarball now carry timestamps, so
`configs/openvla.yaml` sets `waypoint_horizon_seconds: 0.27` and the lab run is
read against the **2.590** floor, not 2.563. Commenting out that one line
restores the fixed-frame path for older data or to reproduce the probe exactly.
The change hits **all three ablation arms equally**, so it improves the signal
without biasing the comparison.

---

## 5. How the OpenVLA pipeline works

Each timestep becomes one supervised example in OpenVLA's native format:

- **Prompt:** `In: What action should the robot take to {instruction}?\nOut:`
- **Target:** the 4-DoF waypoint, normalized to [-1, 1] via q01/q99, embedded
  into OpenVLA's 7-DoF layout, and tokenized as 7 action tokens + EOS.
- **Labels:** prompt positions masked (-100); only the action tokens + EOS are
  supervised.
- **Loss:** `model(**batch).loss` — OpenVLA's own next-token cross-entropy over
  the action-token positions. We reuse its action head verbatim.

### 5.1 Action tokenization and the 4→7-DoF mapping

OpenVLA discretizes each continuous action dim into 256 uniform bins and maps
each bin to one of the **last 256 ids** of the Llama-2 vocabulary
(`token_id = vocab_size - bin`). Training is next-token CE over those positions —
so the pretrained action head is literally the LM head, and it transfers for free.

The drone's 4 DoF are placed into OpenVLA's 7-DoF EEF layout so the pretrained
per-dim action weights line up:

```
drone [vx, vy, vz, yaw_rate] → OpenVLA [x=vx, y=vy, z=vz, roll=0, pitch=0, yaw=yaw_rate, gripper=0]
```

The unused `roll/pitch/gripper` dims are held at their neutral bin. Only dims
`[0, 1, 2, 5]` carry drone data.

**Consequence — the reported training loss is diluted ~2×.** It averages 8
supervised positions (7 action tokens + EOS), but 3 of them (the constant-zero
neutral dims) plus EOS are free to predict once learned. So the offline
evaluator scores **only the 4 real dims**, or half the effect size between arms
would be washed out.

**Why not a custom 4-token scheme?** A drone-only 4-token head would not reuse
OpenVLA's pretrained action tokens at all — it would be a new head, which is
Pipeline B's job. Keeping the 7-DoF layout is what makes this a *transfer* test.

### 5.2 LoRA, quantization, and target modules

Finetuning is **LoRA-only**: the backbone is frozen, adapters train. This is both
a hardware necessity (a 7B full finetune needs ~70 GB of AdamW state) and the
right experiment — the thesis is about *lightweight* finetuning, and LoRA touches
<1% of parameters.

- **Pretrained arm:** 4-bit NF4 QLoRA (double-quantized), fits a 16 GB GPU.
- **LoRA targets = `all-linear`** (attention q/k/v/o **plus** the MLP
  gate/up/down), OpenVLA's official recipe. This was changed from attention-only
  (`q/k/v/o_proj`) because attention-only starves the transfer arm of capacity
  and was the prime suspect for a run stalling at the marginal floor. It is now a
  **config knob** (`model.lora.targets`) so the attention-only-vs-all-linear
  question can itself be run as an ablation without code edits. Cost: trainable
  params rise ~33.5M → ~110M.

**Why config-driven rather than just flipping the constant?** Because "which
modules to LoRA" is a legitimate experimental variable, and hardcoding it would
force a code edit (and a re-review) every time we wanted to sweep it. The knob
also lets us guarantee **all three ablation arms use the same target set**, which
is required for the comparison to be clean.

---

## 6. The ablation — the heart of the project

Breaking the marginal-loss floor proves the pipeline *learns from pixels*. It
does **not** prove that *robot pretraining* is why. Three arms separate the
possible explanations. All share the same data, target, LoRA recipe, and action
head — they differ **only in the initialization of the backbone**.

| Arm (`--init`) | Backbone | What it isolates |
|---|---|---|
| `pretrained` | OpenVLA-7B (VL + **robot** pretrained) | the full transfer claim |
| `prismatic` | Prismatic VLM (**VL** pretrained, robot-naive) | generic VL features |
| `scratch` | same architecture, **random** init | whether pretraining helps *at all* |

Reading the three:

- **OpenVLA ≫ Prismatic ≈ scratch** → the gain is *robot* pretraining
  specifically. **Supports** the cross-embodiment claim.
- **OpenVLA ≈ Prismatic ≫ scratch** → the gain is generic vision-language
  features, not robot transfer. **Falsifies** the headline framing.

### 6.1 Why Prismatic is the control that matters

The `scratch` arm alone is a **weak** control. It answers only "does pretraining
help at all," and with a random LM head under attention-only LoRA it can barely
learn the readout — so a win for `pretrained` is almost guaranteed and tells you
nothing about *robot* transfer specifically.

**Prismatic** (`prism-dinosiglip-224px+7b`) is the clean control: it is the
*exact same architecture* as OpenVLA (DINOv2+SigLIP vision + Llama-2-7B) with the
*same vision-language pretraining*, differing **only** in that it was never
robot/action-trained. So the OpenVLA − Prismatic gap isolates precisely the thing
the thesis is about: the robot action prior. Prismatic is "expensive" (needs the
`TRI-ML/prismatic-vlms` package and ≥24 GB bf16), which is why `scratch` exists as
a self-contained lower-bound control that runs in the same env — but only
Prismatic can actually falsify the claim.

### 6.2 Why the scoring decodes through drone stats, not `predict_action`

The offline evaluator (`eval/openvla_evaluator.py`) teacher-forces the val split
and reports, on the 4 real dims only:

- `action_token_accuracy` — top-1 over the 7 action tokens (quantization-free,
  scale-free, so it is the cleanest cross-arm metric)
- `action_l2` / `per_dim_mae` — decoded error in physical units

Both predicted and gold tokens are decoded through the **drone** q01/q99 stats,
**not** OpenVLA's built-in `predict_action` denorm — that uses the Open-X dataset
statistics and would report on the wrong physical scale. Scoring
predicted-vs-gold-decoded (rather than vs raw) makes the quantization floor
identical across arms, so it cancels in any comparison.

---

## 7. Environment & hardware — why the isolation

OpenVLA's `trust_remote_code` files target **transformers 4.40.x**, incompatible
with the main env's transformers 5.x. So Pipeline A runs in an isolated
`.venv-openvla` (`requirements-openvla.txt`), while Pipeline B stays in the main
env. `accelerate` and `peft` are **pinned to the 4.40 era**, not merely floored —
a newer `accelerate` with bnb ≥0.43.2 dispatches a single-device 4-bit model via
`.to()`, which transformers 4.40.1 refuses, crashing the load *after* all 15 GB
of weights have downloaded. Pinning is the fix.

- Pretrained arm: ≥16 GB GPU (4-bit).
- Scratch/Prismatic: ≥24 GB (full bf16 backbone).
- The 6 GB dev laptop is **verify-only**; the lab GPU runs the real ablation.
- **Precision is flag-gated** (`training.precision: bf16 | fp16 | fp32`). Lab GPUs
  default to bf16; Turing T4s (Colab) have no bf16 tensor cores, so they use fp16
  with an automatic GradScaler. One flag, so the lab path is unchanged.

---

## 8. Verification strategy — why CPU tests matter here

Because the real model needs a GPU the dev machine doesn't have, the project is
built so that **everything decidable without the weights is decided on CPU**:

- `tests/test_action_tokenizer.py` — discretization round-trip (< bin width), the
  4↔7-DoF embedding, id-range clipping, all with the **real** OpenVLA tokenizer.
- `tests/test_openvla_dataset.py` — normalization round-trip, prompt/label
  masking, batched shapes, action-token recovery.
- `tests/test_openvla_evaluator.py` — the decode/scoring math (that a one-bin
  offset costs one bin, that corrupting a neutral dim leaves drone error at zero).
- `tests/test_waypoint.py` — the body-frame geometry and the fixed-time index
  logic (nearest-forward-frame, dropped-frame snapping) as pure math.

This isolates the untested surface to exactly one thing: the GPU forward/backward
in the OpenVLA env. When the lab GPU run happens, a failure is almost certainly in
the model-load or dtype path, not in the data or scoring logic — because those are
already proven. The **marginal-loss floor** the trainer prints at startup is the
yardstick every run is read against: an arm settling at its floor learned the
action prior and nothing from the camera, however smooth its curve looks.

**Why not just run it on Colab and skip the CPU tests?** Colab weights re-download
(~15 GB) every session and the free tier idles out; debugging data/scoring bugs
there is slow and expensive. The CPU tests turn most bugs into a two-second local
failure. Colab is used for what only it can do — proving the 4-bit model loads and
learns from pixels — via a capped 600-step probe.

---

## 9. Current status

- **Pipeline B:** end-to-end verified on the dev laptop (smoke train + offline
  eval pass); full UZH-FPV run is a standard GPU job.
- **Pipeline A data/scoring layers:** CPU-verified (all four test suites pass).
- **Pipeline A T4 probe (2026-07-16):** the pretrained arm broke the marginal
  floor on a free Colab T4 — 2.201 vs 2.563 nats over a 600-step / 2400-sample,
  single-epoch, no-repeat run, i.e. a generalization result, not memorization.
  **This proved the pipeline learns from pixels and was the green light to book
  the lab GPU.** It does *not* yet show robot pretraining is the cause — that is
  what the 3-arm ablation exists to determine.
- **Recent lab-GPU prep:** LoRA targets made config-driven and defaulted to
  `all-linear`; converter now emits `timestamps.npy` and the dataset supports a
  fixed-time waypoint; data re-converted locally (backward-compat proven:
  fixed-frame floor still exactly 2.563).
- **Lab GPU secured 2026-09-05 (one 24 GB card, 3090/4090-class).** Config now
  carries a per-arm batch/accum shape so the 4-bit transfer arm and the two bf16
  control arms fit the same card at a **matched effective batch of 16** —
  `resolve_arm_training_cfg()` raises if an override changes that product, since
  an unmatched effective batch would confound every cross-arm number. Fixed-time
  waypoint enabled (floor 2.590). `--max-steps` added so each arm can be
  smoke-tested against the real config before the hours-long run. Procedure:
  `docs/LAB_GPU.md`.

## 10. Open questions for the lab GPU

*(The runbook that answers these is `docs/LAB_GPU.md`.)*

1. Does `all-linear` (~110M trainable) fit each arm at its planned batch shape
   (pretrained 8, scratch 4, prismatic 2)?
2. Run the headline ablation: `pretrained` vs `scratch` vs `prismatic` — does the
   OpenVLA − Prismatic gap actually isolate robot pretraining?
3. Does the fixed-time waypoint lower any arm's floor as predicted?
4. Verify the Prismatic API assumptions (model id, `labels` alignment, dinosiglip
   pixel dict) that are currently written but unvalidated.
```
