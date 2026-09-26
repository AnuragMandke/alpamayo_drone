"""
scripts/analyze_alpamayo_probe.py — is Alpamayo's gap to the no-vision
baselines real, or noise?

The probe's summary compares means. That is not enough here: the first velocity
run put Alpamayo 4% ahead of constant velocity on the 45-deg-down mount, and
merely changing how a no-vision baseline estimates velocity moves it by about
that much. So this script:

  1. Recomputes several NO-VISION baselines per sample, from the same 1.6 s
     history Alpamayo was given:
       cv       constant 3D velocity (the probe's own baseline)
       cv_2d    constant horizontal velocity, no vertical motion (what a planar
                model could at best do without vision)
       ctrv     constant turn rate and speed, horizontal — the standard
                vehicle-motion baseline, i.e. "keep turning like you are"
  2. Scores Alpamayo against each one PAIRED, per sample.
  3. Puts a 95% CI on the mean paired difference by bootstrapping whole
     TRAJECTORIES, not frames. Frames from one flight are strongly correlated,
     and 200 samples come from only 12 flights; a per-frame bootstrap would give
     a far too narrow interval.

A negative difference means Alpamayo is better. Only an interval that excludes
zero is evidence either way.

USAGE (in .venv-alpamayo or any env with numpy/scipy/torch; no GPU needed)
    python scripts/analyze_alpamayo_probe.py outputs/alpamayo/probe_*.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.openvla_dataset import waypoint_target_index
from probe_alpamayo import EGO_HISTORY_HZ, history_index

DT = 1.0 / EGO_HISTORY_HZ


def baselines(poses, idx, horizon):
    """World-frame displacement at `horizon` for each no-vision baseline."""
    xyz = poses[idx, :3]
    v3 = (xyz[-1] - xyz[-3]) / (2 * DT)
    out = {"cv": v3 * horizon,
           "cv_2d": np.array([v3[0] * horizon, v3[1] * horizon, 0.0])}

    # CTRV: speed and heading from the last two 2-step windows, turn rate from
    # their heading change. Integrated in closed form over the horizon.
    v_prev = (xyz[-3] - xyz[-5]) / (2 * DT)
    speed = np.linalg.norm(v3[:2])
    th1 = np.arctan2(v3[1], v3[0])
    th0 = np.arctan2(v_prev[1], v_prev[0])
    dth = (th1 - th0 + np.pi) % (2 * np.pi) - np.pi
    om = dth / (2 * DT)
    if abs(om) < 1e-3 or speed < 0.3:
        d = speed * horizon * np.array([np.cos(th1), np.sin(th1)])
    else:
        d = (speed / om) * np.array([np.sin(th1 + om * horizon) - np.sin(th1),
                                     np.cos(th1) - np.cos(th1 + om * horizon)])
    out["ctrv"] = np.array([d[0], d[1], 0.0])
    return out


def cluster_bootstrap(diff, groups, reps=5000, seed=0):
    """95% CI of mean(diff), resampling whole groups with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = [diff[groups == g] for g in ug]
    means = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(ug), len(ug))
        means[r] = np.concatenate([by[p] for p in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(path, data_root):
    blob = json.load(open(path))
    summ, rows = blob["summary"], blob["samples"]
    h = summ["horizon_s"]
    root = Path(data_root) / "trajectories"

    cache, recs = {}, []
    for r in rows:
        if r["traj"] not in cache:
            d = root / r["traj"]
            cache[r["traj"]] = (np.load(d / "poses.npy"), np.load(d / "timestamps.npy"))
        poses, ts = cache[r["traj"]]
        t = r["t"]
        idx = history_index(ts, t)
        j = waypoint_target_index(ts, t, h)
        gt = poses[j, :3] - poses[t, :3]
        b = baselines(poses, idx, h)
        rec = {"traj": r["traj"], "mount": r["mount"],
               "alp_3d": r["l2_3d"], "alp_h": r["l2_horiz"]}
        for k, v in b.items():
            rec[f"{k}_3d"] = float(np.linalg.norm(v - gt))
            rec[f"{k}_h"] = float(np.linalg.norm(v[:2] - gt[:2]))
        recs.append(rec)

    print(f"\n=== {path}  (ego-frame={summ['ego_frame']}, images={summ['images']}, "
          f"K={summ['traj_samples']}, horizon={h}s)")
    print("  mean L2 in metres; diff = Alpamayo - baseline (negative = Alpamayo "
          "better), 95% CI by trajectory bootstrap")
    for subset in ["all", "forward", "down45"]:
        rs = [x for x in recs if subset == "all" or x["mount"] == subset]
        if not rs:
            continue
        g = np.array([x["traj"] for x in rs])
        print(f"\n  [{subset}] n={len(rs)} samples from {len(np.unique(g))} trajectories")
        for metric, suf in [("horizontal", "_h"), ("3D", "_3d")]:
            a = np.array([x["alp" + suf] for x in rs])
            print(f"    {metric:<10} Alpamayo {a.mean():.4f}")
            for base in ["cv", "cv_2d", "ctrv"]:
                bvals = np.array([x[base + suf] for x in rs])
                diff = a - bvals
                lo, hi = cluster_bootstrap(diff, g)
                verdict = ("Alpamayo better" if hi < 0 else
                           "Alpamayo worse" if lo > 0 else "no detectable difference")
                print(f"      vs {base:<6} {bvals.mean():.4f}   diff {diff.mean():+.4f} "
                      f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--data-root", default=None,
                   help="Default: data.dataset_root from configs/openvla.yaml")
    args = p.parse_args()
    root = args.data_root
    if root is None:
        import yaml
        root = yaml.safe_load(open("configs/openvla.yaml"))["data"]["dataset_root"]
    for path in args.results:
        analyze(path, root)


if __name__ == "__main__":
    main()
