"""
scripts/ablate_openvla.py — run the full cross-embodiment ablation in one command.

For each arm (pretrained / scratch / prismatic) it shells out to the existing,
tested scripts — train_openvla.py then eval_openvla.py — then aggregates each
arm's eval_metrics.json into a comparison table and an automatic verdict on the
headline claim.

    python scripts/ablate_openvla.py --config configs/openvla.yaml

Because the arms can need different environments (OpenVLA env for
pretrained/scratch, prismatic-vlms env for prismatic), each subprocess uses the
current interpreter by default. The typical workflow is to run the arms whose env
is active, then aggregate everything at the end:

    # in the OpenVLA env
    python scripts/ablate_openvla.py --config configs/openvla.yaml --arms pretrained scratch
    # in the prismatic-vlms env
    python scripts/ablate_openvla.py --config configs/openvla.yaml --arms prismatic
    # aggregate all completed arms without re-running anything
    python scripts/ablate_openvla.py --config configs/openvla.yaml --skip-train --skip-eval

Aggregation always runs over the requested arms that have an eval_metrics.json,
so repeated invocations build up the final table.

UNVALIDATED end-to-end (needs weights + lab GPU); the orchestration itself is
env-agnostic. See docs/OPENVLA.md.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
from pathlib import Path

import yaml

ARMS = ["pretrained", "scratch", "prismatic"]
HERE = Path(__file__).resolve().parent
# action_token_accuracy gap (absolute) considered a meaningful separation.
GAP = 0.03
# The comparison metric (higher is better); quantization-free, scale-free.
METRIC = "action_token_accuracy"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/openvla.yaml")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    p.add_argument("--epoch", type=int, default=None,
                   help="Checkpoint epoch to eval (default: latest present)")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter for subprocesses (default: this one)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without running them")
    return p.parse_args()


def arm_dir(cfg, arm):
    return Path(cfg["training"]["output_dir"]) / arm


def find_ckpt(cfg, arm, epoch):
    d = arm_dir(cfg, arm)
    if epoch is not None:
        return d / f"epoch{epoch:03d}"
    ckpts = sorted(d.glob("epoch*"))
    return ckpts[-1] if ckpts else None


def run(cmd, dry_run):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=HERE.parent).returncode


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    py = args.python

    for arm in args.arms:
        print(f"\n{'='*64}\n  ARM: {arm}\n{'='*64}")

        if not args.skip_train:
            rc = run([py, str(HERE / "train_openvla.py"),
                      "--config", args.config, "--init", arm], args.dry_run)
            if rc != 0 and not args.dry_run:
                print(f"[ablate] train ({arm}) failed (rc={rc}); skipping arm")
                continue

        if not args.skip_eval:
            ckpt = find_ckpt(cfg, arm, args.epoch)
            if ckpt is None and not args.dry_run:
                print(f"[ablate] no checkpoint under {arm_dir(cfg, arm)}; "
                      f"skipping eval for {arm}")
                continue
            ckpt = ckpt or (arm_dir(cfg, arm) / "epoch<latest>")
            rc = run([py, str(HERE / "eval_openvla.py"),
                      "--config", args.config, "--init", arm,
                      "--ckpt", str(ckpt)], args.dry_run)
            if rc != 0 and not args.dry_run:
                print(f"[ablate] eval ({arm}) failed (rc={rc})")

    # ---- Aggregate -----------------------------------------------------------
    if args.dry_run:
        return

    results = {}
    for arm in args.arms:
        ckpt = find_ckpt(cfg, arm, args.epoch)
        m = ckpt / "eval_metrics.json" if ckpt else None
        if m and m.exists():
            results[arm] = json.load(open(m))

    if not results:
        print("\n[ablate] no eval_metrics.json found for any requested arm.")
        return

    _print_table(results)
    verdict = _verdict(results)
    print(f"\n[Verdict] {verdict}")

    out = Path(cfg["training"]["output_dir"]) / "ablation_openvla.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "verdict": verdict, "metric": METRIC,
               "gap_threshold": GAP}, open(out, "w"), indent=2)
    print(f"[ablate] Saved -> {out}")


def _print_table(results):
    print(f"\n{'Arm':<12}{'TokenAcc':>10}{'ActionL2':>10}"
          f"{'vx':>8}{'vy':>8}{'vz':>8}{'yaw':>8}{'N':>8}")
    print("-" * 72)
    for arm in ARMS:
        if arm not in results:
            continue
        r = results[arm]
        pdm = r["per_dim_mae"]
        print(f"{arm:<12}{r[METRIC]:>10.4f}{r['action_l2']:>10.4f}"
              f"{pdm['vx']:>8.3f}{pdm['vy']:>8.3f}{pdm['vz']:>8.3f}"
              f"{pdm['yaw_rate']:>8.3f}{r['n_samples']:>8}")


def _verdict(results):
    """Heuristic read of the headline claim from action_token_accuracy gaps.
    The numbers (gaps) are printed; the human makes the call — this is a guide."""
    have = [a for a in ARMS if a in results]
    if not {"pretrained", "scratch", "prismatic"}.issubset(have):
        return (f"need all three arms for a verdict; have {have}. "
                f"Run the missing arm(s), then re-aggregate (--skip-train --skip-eval).")
    acc = {a: results[a][METRIC] for a in ARMS}
    d_pr = acc["pretrained"] - acc["prismatic"]   # robot pretraining gain (vs VL-only)
    d_rs = acc["prismatic"] - acc["scratch"]      # VL pretraining gain (vs random)
    gaps = (f"OpenVLA-Prismatic={d_pr:+.3f}, Prismatic-scratch={d_rs:+.3f} "
            f"(meaningful gap >= {GAP})")
    if d_pr >= GAP and d_rs < GAP:
        return f"SUPPORTS the claim: robot pretraining drives the gain. {gaps}"
    if d_pr < GAP and d_rs >= GAP:
        return f"WEAKENS the claim: gain is generic VL features, not robot transfer. {gaps}"
    if d_pr >= GAP and d_rs >= GAP:
        return f"PARTIAL: VL pretraining helps AND robot pretraining adds on top. {gaps}"
    return f"INCONCLUSIVE: no arm separates clearly. {gaps}"


if __name__ == "__main__":
    main()
