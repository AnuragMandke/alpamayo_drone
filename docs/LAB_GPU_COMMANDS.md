# Lab-GPU command sheet — copy-paste, in order

Terse companion to `docs/LAB_GPU.md`, which explains *why* each step exists and
how to read the numbers. This file is only the commands. Steps 0–1 run on the
**dev box**; everything from step 2 runs on the **VM**.

Set these once per shell and the rest pastes verbatim.

```bash
# on the VM
export REPO=~/alpamayo_drone
export HF_HOME=/path/on/a/big/disk/hf      # weights are ~15 GB — not the root disk
```

---

## 0. Dev box → VM: ship the data (the repo comes from git, the data cannot)

`data/uzh_fpv/` is gitignored, so it travels as the tarball. From **Git Bash on
the dev box** (`E:\alpamayo_drone`):

```bash
LAB=user@lab-host                          # <-- your login
scp /e/alpamayo_drone/uzh_fpv.tar.gz $LAB:~/     # 566 MB
```

PowerShell instead of Git Bash — same thing, Windows path:

```powershell
scp E:\alpamayo_drone\uzh_fpv.tar.gz user@lab-host:~/
```

If the VM is reached through a bastion, add `-J user@bastion`. If `scp` stalls
on a flaky link, `rsync -avP --partial` resumes where it stopped:

```bash
rsync -avP --partial /e/alpamayo_drone/uzh_fpv.tar.gz $LAB:~/
```

Then log in:

```bash
ssh $LAB
```

## 1. Clone and unpack

```bash
git clone https://github.com/AnuragMandke/alpamayo_drone.git $REPO
cd $REPO
mkdir -p data && tar xzf ~/uzh_fpv.tar.gz -C data
```

Both counts must print **118**. A 0 on the second means the tarball predates
`timestamps.npy` and the fixed-time target will refuse to run:

```bash
ls data/uzh_fpv/trajectories | wc -l
ls data/uzh_fpv/trajectories/*/timestamps.npy | wc -l
```

## 2. Environment

The pins matter — `accelerate`/`peft` must stay in the transformers 4.40 era, or
the 4-bit load dies *after* downloading all 15 GB.

```bash
cd $REPO
python -m venv .venv-openvla
source .venv-openvla/bin/activate
pip install --upgrade pip
pip install -r requirements-openvla.txt
```

```bash
nvidia-smi                                 # confirm ~24 GB free
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

Expect `True` and `4.40.1`.

## 3. CPU tests in the new env (2 minutes; separates a bad port from a bad GPU path)

```bash
for t in action_tokenizer openvla_dataset openvla_evaluator waypoint arm_config; do
  printf "%-20s " "$t"; python tests/test_$t.py 2>&1 | tail -1
done
```

Five `... TESTS PASSED` lines.

## 4. Weights

```bash
huggingface-cli download openvla/openvla-7b                  # ~15 GB
pip install git+https://github.com/TRI-ML/prismatic-vlms     # prismatic arm only
```

## 5. Smoke all three arms (~5 min each — do not skip)

```bash
python scripts/train_openvla.py --config configs/openvla.yaml --init pretrained --max-steps 20
python scripts/train_openvla.py --config configs/openvla.yaml --init scratch    --max-steps 20
python scripts/train_openvla.py --config configs/openvla.yaml --init prismatic  --max-steps 20
```

Each header must show, before any loss line:

```
[Train] precision=torch.bfloat16 (autocast=on)
[Train] batch_size=N x grad_accum=M -> effective batch 16   <- 16 on ALL THREE
[Train] target_mode=waypoint horizon=0.27s
[Train] marginal-only loss floor = 2.590 nats
```

**On OOM:** halve that arm's `batch_size` and double its
`gradient_accumulation_steps` under `training.per_arm` in
`configs/openvla.yaml`. The product must stay 16 (`1x16` is fine); the trainer
raises if it doesn't.

```bash
nano configs/openvla.yaml        # training.per_arm.<arm>
```

## 6. Full runs — one arm at a time, in tmux

Two 7B arms will not share a 24 GB card. 736 steps/epoch, 3,680 over 5 epochs.

```bash
tmux new -s ablate
cd $REPO && source .venv-openvla/bin/activate
mkdir -p outputs
```

```bash
for arm in pretrained scratch prismatic; do
  python scripts/train_openvla.py --config configs/openvla.yaml --init $arm \
    2>&1 | tee outputs/train_$arm.log
done
```

Detach with `ctrl-b d`; come back with `tmux attach -t ablate`.

Watch from a second shell:

```bash
watch -n 30 nvidia-smi
tail -f $REPO/outputs/train_pretrained.log
```

Loss at or above **2.590** = that arm learned the marginal action distribution
and nothing from the camera, however smooth the curve. Clearly below = it is
using the image.

## 7. Score the ablation

```bash
python scripts/ablate_openvla.py --config configs/openvla.yaml --skip-train \
  2>&1 | tee outputs/ablation.log
```

Prints the per-arm table, the verdict, and writes
`outputs/openvla/ablation_openvla.json`.

## 8. Pull the results back to the dev box

From the **dev box**, not the VM:

```bash
scp $LAB:~/alpamayo_drone/outputs/ablation.log .
scp $LAB:~/alpamayo_drone/outputs/train_*.log .
scp $LAB:~/alpamayo_drone/outputs/openvla/ablation_openvla.json .
```

Adapters are small (~130 MB/arm) if you want the checkpoints too:

```bash
rsync -avP $LAB:~/alpamayo_drone/outputs/openvla/ ./outputs/openvla/
```
