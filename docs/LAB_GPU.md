# Lab-GPU runbook — the three-arm ablation on a 24 GB card

Target hardware: **one 24 GB GPU (RTX 3090 / 4090)**. This is the run the whole
project has been built toward; the T4 probe (2026-07-16) only proved the
pretrained arm learns from pixels. See `docs/DESIGN.md` §6 for what the arms mean
and `docs/OPENVLA.md` for the pipeline itself.

Everything decidable without the weights is already decided on CPU (four test
suites, all passing), so a failure here is almost certainly in the **model-load
or dtype path** — not in the data or scoring logic.

---

## 0. Ship the code and the data

The dataset is gitignored (573 MB unpacked), so it travels separately. The
staged tarball at the repo root is current — it carries `timestamps.npy` for all
118 trajectories, which the fixed-time waypoint target requires.

```bash
# from the dev box
scp uzh_fpv.tar.gz  <lab>:~/            # 566 MB
git push                                # then clone/pull on the lab box

# on the lab box, from the repo root
mkdir -p data && tar xzf ~/uzh_fpv.tar.gz -C data
ls data/uzh_fpv/trajectories | wc -l    # expect 118
ls data/uzh_fpv/trajectories/*/timestamps.npy | wc -l   # expect 118 — not 0
```

If that last count is 0 the tarball is stale: re-run
`scripts/convert_uzh_fpv.py` and re-tar, or comment out
`waypoint_horizon_seconds` in `configs/openvla.yaml` to fall back to the
fixed-frame target.

## 1. Environment

OpenVLA's `trust_remote_code` files target transformers 4.40.x, so this needs its
own venv — `accelerate` and `peft` are **pinned**, not floored (see
`docs/DESIGN.md` §7; a newer accelerate crashes the 4-bit load *after* the 15 GB
download).

```bash
python -m venv .venv-openvla && source .venv-openvla/bin/activate
pip install -r requirements-openvla.txt

nvidia-smi                              # confirm 24GB free and the driver/CUDA pair
export HF_HOME=/path/on/a/big/disk      # the weights are ~15 GB
```

## 2. Prove the CPU layers still pass in this env

Two minutes, and it separates "the port broke something" from "the GPU path is
wrong" before any weights download.

```bash
for t in action_tokenizer openvla_dataset openvla_evaluator waypoint; do
  python tests/test_$t.py | tail -1
done
```

## 3. Fetch the weights once, explicitly

```bash
huggingface-cli download openvla/openvla-7b        # ~15 GB
pip install git+https://github.com/TRI-ML/prismatic-vlms   # prismatic arm only
```

## 4. Pre-flight: smoke every arm before committing hours

`--max-steps` caps the run against the **real** config, so the smoke exercises the
exact batch shape and precision the full run will use. Run all three now — an OOM
found in 5 minutes costs nothing; the same OOM found in hour three costs a night.

```bash
python scripts/train_openvla.py --config configs/openvla.yaml --init pretrained --max-steps 20
python scripts/train_openvla.py --config configs/openvla.yaml --init scratch    --max-steps 20
python scripts/train_openvla.py --config configs/openvla.yaml --init prismatic  --max-steps 20
```

Check in the header of each, before the loss lines:

- `precision=torch.bfloat16` (Ampere/Ada have bf16 tensor cores; fp16 is the T4 path)
- `batch_size=N x grad_accum=M -> effective batch 16` — **16 on all three arms**
- `marginal-only loss floor = 2.590 nats` — the yardstick for this run
- `target_mode=waypoint horizon=0.27s`

The `prismatic` smoke is doing double duty: it is the validation of the three
unverified Prismatic API assumptions (`docs/DESIGN.md` §10.4) — the model id in
the TRI-ML registry, that `forward()` accepts `labels` aligned to the text
stream, and the batched dinosiglip `pixel_values` dict. If it produces a finite
loss for 20 steps, all three hold.

**If an arm OOMs:** halve its `batch_size` and double its
`gradient_accumulation_steps` under `training.per_arm` in
`configs/openvla.yaml`. The product must stay 16 — the trainer raises rather than
letting a silent effective-batch mismatch confound the comparison.

## 5. The full runs

~11.8k train / 1.3k val waypoint samples → **736 optimizer steps/epoch, 3,680
over 5 epochs**, identical for every arm by construction. The arms differ in
wall-clock, not in steps: `pretrained` runs 8 samples per forward, `scratch` 4,
`prismatic` 2, and the two bf16 arms are gradient-checkpointed, so expect the
controls to take multiples of the transfer arm's time.

```bash
for arm in pretrained scratch prismatic; do
  python scripts/train_openvla.py --config configs/openvla.yaml --init $arm \
    2>&1 | tee outputs/train_$arm.log
done
```

Run them **one at a time** — two 7B arms will not share a 24 GB card. Under
`nohup`/`tmux` if the session can drop.

## 6. Score and read the ablation

```bash
python scripts/ablate_openvla.py --config configs/openvla.yaml --skip-train
```

That evaluates the latest checkpoint of each arm and prints the comparison table
plus an automatic verdict; it writes `outputs/openvla/ablation_openvla.json`.
(`ablate_openvla.py` can also drive training, but running the arms by hand keeps
one OOM from taking down the other two.)

The verdict turns on `action_token_accuracy`, scored on the **4 real drone dims
only** — the reported *training* loss is diluted ~2× by the three constant
neutral dims plus EOS, so never compare arms on the loss curve alone.

- **OpenVLA ≫ Prismatic ≈ scratch** → the gain is *robot* pretraining. Supports the claim.
- **OpenVLA ≈ Prismatic ≫ scratch** → the gain is generic VL features. Falsifies the framing.

## 7. What to watch while it runs

| Signal | Reading |
|---|---|
| Loss at/above **2.590** | The arm learned the marginal action distribution and nothing from the camera — a null result however smooth the curve looks. |
| Loss clearly below 2.590 | The arm is using the image. (The T4 probe hit 2.201 against the 2.563 fixed-frame floor.) |
| `scratch` sitting at the floor | Expected and uninformative — a random LM head can barely learn the readout. This is why `prismatic` is the control that matters. |
| Step log | A **windowed** mean, reset each log — a plateau shows up immediately, unlike a cumulative mean. |

## 8. Open questions this run answers

1. Does `all-linear` LoRA (~110M trainable) fit at these batch shapes? (§4 answers it.)
2. The headline: does the OpenVLA − Prismatic gap isolate robot pretraining? (§6.)
3. Does the fixed-time waypoint behave as predicted — floor 2.590 vs 2.563, with
   a tighter target range? (Printed at startup in §4.)
4. Do the Prismatic API assumptions hold? (§4's third smoke.)
