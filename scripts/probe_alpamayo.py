"""
scripts/probe_alpamayo.py — zero-shot feasibility probe: does a DRIVING VLA
produce anything sensible on drone footage?

WHY THIS IS A PROBE AND NOT A FOURTH ARM
----------------------------------------
Alpamayo cannot drop into the OpenVLA/Prismatic/scratch ablation. It differs in
kind, not just in weights:

  1. OUTPUT. Alpamayo emits a continuous trajectory (64 waypoints / 6.4 s),
     internally acceleration + curvature under a UNICYCLE model. There are no
     discrete action tokens, so the action-token cross-entropy and the 2.590-nat
     marginal floor — the yardstick the whole ablation is read against — simply
     do not apply. The ONLY metric that can compare it to the other three arms
     is `action_l2`, in physical units, on the same val split.

  2. EMBODIMENT. A unicycle is planar: forward speed and yaw rate. A quadrotor
     is not. Whether Alpamayo can express vertical motion AT ALL is an open
     question this probe measures directly (see `pred_abs_z_max_mean` below) —
     if predicted z is ~0 everywhere, the dz component of our target is
     structurally unreachable and any L2 including it is unfair by construction.

  3. INPUT. Alpamayo expects FOUR cameras (cross-left, front-wide,
     cross-right, front-tele) x FOUR frames each (t0-0.3s .. t0) plus 1.6 s of
     egomotion history. UZH-FPV has ONE forward camera. This probe feeds the
     real 4-frame clip from that camera to all four mounts, which is
     off-distribution and is the single biggest caveat on any number below.
     The ego history, at least, is real: poses.npy + timestamps.npy give it to
     us exactly.

  4. ENVIRONMENT. transformers >=4.57.1 / torch >=2.8 against OpenVLA's hard pin
     at 4.40.1. Separate venv, mandatory. See requirements-alpamayo.txt.

So this answers one cheap question — "does a driving prior produce sensible
short-horizon motion on drone video, zero-shot?" — and costs an afternoon
instead of a week. A null result is still a result, and it is the evidence that
would justify (or kill) building a real Alpamayo arm.

USAGE
-----
Stage 1 validates the API against ONE sample and prints every shape. Run it
first; it is the `--max-steps 20` of this pipeline.

    python scripts/probe_alpamayo.py --inspect
    python scripts/probe_alpamayo.py --n-samples 200 --ego-frame velocity
    python scripts/probe_alpamayo.py --n-samples 200 --ego-frame velocity --images blank

EGO FRAME MATTERS MORE THAN ANYTHING ELSE HERE. The first run (--ego-frame
body) scored 0.67 m 3D against 0.26 m for constant velocity — but a no-vision
fake that just goes straight at its current speed scores 0.61 m in that same
frame. The drone flies tilted ~23 deg (median) and sideslips ~28 deg off its
body-x heading; a unicycle can express neither. --ego-frame velocity hands the
model a level virtual car pointing along the direction of travel, and the same
fake drops to ~0.24 m. Only a gap between Alpamayo and constant velocity IN THAT
FRAME says anything about the driving prior. --images blank then says whether
any of it comes from vision.

Inputs follow NVlabs/alpamayo at commit 11a0e01 (test_inference.py, helper.py,
load_physical_aiavdataset.py). An earlier version guessed the API from the model
card and died at AutoProcessor: the checkpoint ships no processor files. Run
--inspect after any upstream bump before trusting a long run.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.openvla_dataset import waypoint_target_index

ALPAMAYO_ID = "nvidia/Alpamayo-R1-10B"

# Input layout, from upstream src/alpamayo_r1/load_physical_aiavdataset.py
# (commit 11a0e01): cameras sorted cross_left, front_wide, cross_right,
# front_tele; 4 frames per camera at t0-0.3s .. t0; flattened camera-major.
N_CAMERAS = 4
FRAMES_PER_CAMERA = 4
FRAME_DT = 0.1

# Ego history: 16 steps at 10 Hz = 1.6 s ending at t0, same source. The prompt
# reserves 48 <|traj_history|> tokens for it, so the count is not negotiable.
EGO_HISTORY_STEPS = 16
EGO_HISTORY_HZ = 10

# Output trajectory: 64 waypoints over 6.4 s => 0.1 s spacing.
TRAJ_DT = 0.1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/openvla.yaml",
                   help="Reused ONLY for dataset_root / train_split / seed, so "
                        "the probe scores the same held-out trajectories the "
                        "other three arms were scored on.")
    p.add_argument("--model-id", default=ALPAMAYO_ID)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--inspect", action="store_true",
                   help="Stage 1: one sample, print every shape, then stop.")
    p.add_argument("--horizon", type=float, default=0.27,
                   help="Match the other arms' fixed-time waypoint target.")
    p.add_argument("--ego-frame", choices=["body", "level", "velocity"],
                   default="body",
                   help="Frame Alpamayo's ego history is expressed in (and its "
                        "prediction read back from). body = the drone's full "
                        "attitude (tilted ~23 deg median, the original probe). "
                        "level = gravity-aligned, heading of body x. velocity = "
                        "gravity-aligned, heading along the direction of travel: "
                        "a virtual car following the drone's horizontal path, "
                        "which removes the sideslip a unicycle cannot express.")
    p.add_argument("--images", choices=["real", "blank"], default="real",
                   help="blank = all-black frames: if L2 barely moves, the "
                        "prediction comes from the ego history, not from vision.")
    p.add_argument("--traj-samples", type=int, default=1,
                   help="Average this many sampled trajectories (the MEAN, not "
                        "best-of-K: picking the one nearest GT would peek).")
    p.add_argument("--out", default=None,
                   help="Default: outputs/alpamayo/probe_<frame>_<images>_k<K>.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data: the same val split the other arms were scored on
# ---------------------------------------------------------------------------

def val_trajectories(root, train_split, seed):
    """Byte-identical split to OpenVLADroneDataset, so the comparison is against
    the same held-out trajectories — not a fresh shuffle that would quietly make
    the numbers incomparable."""
    trajs = sorted((Path(root) / "trajectories").glob("traj_*"))
    if not trajs:
        raise FileNotFoundError(f"No trajectories under {Path(root)/'trajectories'}")
    rng = random.Random(seed)
    idx = list(range(len(trajs)))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_split)
    return [trajs[i] for i in idx[n_train:]]


def history_index(timestamps, t):
    """Frame indices for the EGO_HISTORY_STEPS samples ending at t, picked by
    CLOCK TIME, not frame count — UZH-FPV drops frames, and a fixed frame count
    would give a variable-duration history. None if it runs off the clip start."""
    want = timestamps[t] - np.arange(EGO_HISTORY_STEPS - 1, -1, -1) / EGO_HISTORY_HZ
    if want[0] < timestamps[0]:
        return None
    return [int(np.argmin(np.abs(timestamps - w))) for w in want]


def _yaw_rot(yaw):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("z", yaw)


def _heading_of(R):
    """Heading (rad) of body x projected onto the world horizontal plane."""
    fx = R.apply([1.0, 0.0, 0.0])
    fx = np.atleast_2d(fx)
    return np.arctan2(fx[:, 1], fx[:, 0])


def ego_frame(poses, idx, mode):
    """Rotation world<-F of the frame Alpamayo sees at t0 (= idx[-1]), plus the
    per-step rotations of its ego history expressed in F.

    body     : F = the drone's full attitude. History rotations = R_F^-1 R_i.
    level    : F = yaw-only, heading of body x. Each history step is a level
               "car" with that step's body-x heading.
    velocity : F = yaw-only, heading along horizontal velocity. Each history
               step is a level car pointing where the drone was going — no
               sideslip, which is the only motion a unicycle can express.
    In every mode the last history rotation is the identity, as upstream's
    loader produces for a real car."""
    from scipy.spatial.transform import Rotation
    R_all = Rotation.from_quat(poses[idx, 3:7])
    if mode == "body":
        R_F = R_all[-1]
        return R_F, (R_F.inv() * R_all).as_matrix()
    if mode == "level":
        yaw = _heading_of(R_all)
    else:
        xyz = poses[idx, :3]
        v = np.gradient(xyz, 1.0 / EGO_HISTORY_HZ, axis=0)
        yaw = np.arctan2(v[:, 1], v[:, 0])
        slow = np.linalg.norm(v[:, :2], axis=1) < 0.3   # heading undefined
        yaw[slow] = _heading_of(R_all)[slow]
    R_F = _yaw_rot(yaw[-1])
    return R_F, np.stack([_yaw_rot(y - yaw[-1]).as_matrix() for y in yaw])


def ego_history(poses, idx, R_F):
    """History positions in frame F, origin at t0 (last entry = 0)."""
    xyz = R_F.inv().apply(poses[idx, :3] - poses[idx[-1], :3])
    return xyz.astype(np.float32)


def clip_frames(traj_dir, timestamps, t):
    """uint8 (FRAMES_PER_CAMERA, 3, H, W) at t0-0.3s .. t0, picked by CLOCK
    TIME like ego_history. None if the clip starts too late."""
    want = timestamps[t] - np.arange(FRAMES_PER_CAMERA - 1, -1, -1) * FRAME_DT
    if want[0] < timestamps[0]:
        return None
    idx = [int(np.argmin(np.abs(timestamps - w))) for w in want]
    frames = [np.asarray(Image.open(traj_dir / "images" / f"rgb_{i:03d}.png")
                         .convert("RGB")) for i in idx]
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def constant_velocity(poses, idx, horizon, window=2):
    """The no-vision baseline Alpamayo has to beat: world-frame displacement
    from extrapolating the mean velocity over the last `window` history steps.
    Alpamayo sees this same history, so matching it on forward motion alone
    proves nothing about the driving prior; only beating it does."""
    v = (poses[idx[-1], :3] - poses[idx[-1 - window], :3]) / (window / EGO_HISTORY_HZ)
    return v * horizon


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_alpamayo(model_id, device):
    """Follows NVlabs/alpamayo src/alpamayo_r1/test_inference.py (commit
    11a0e01). The checkpoint ships no processor files: upstream builds one from
    Qwen3-VL-2B-Instruct and swaps in the model's own tokenizer.
    Import errors here are the expected first failure — they mean the model
    package is not installed (pip install git+https://github.com/NVlabs/alpamayo)
    or you are in .venv-openvla instead of .venv-alpamayo."""
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    except ImportError as e:
        raise SystemExit(
            f"Could not import alpamayo_r1 ({e}).\n"
            "  - Are you in .venv-alpamayo? (.venv-openvla pins transformers "
            "4.40.1; Alpamayo needs >=4.57.1 — they cannot share an env.)\n"
            "  - Install: pip install git+https://github.com/NVlabs/alpamayo"
        )
    from alpamayo_r1 import helper

    model = AlpamayoR1.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    return model, processor


def build_inputs(processor, frames, ego_xyz, ego_rot, device):
    # ego_rot: (16,3,3) float32 in the same frame as ego_xyz
    """Four camera mounts from ONE drone camera's 4-frame clip.

    This is the probe's biggest caveat: Alpamayo was trained on a real 4-camera
    rig with genuine cross-view parallax, and we are handing it the same forward
    clip four times. The prompt is upstream's fixed training prompt — there is
    no free-text instruction to set. Any negative result is therefore ambiguous between "the
    driving prior does not transfer" and "we fed it an input it has never seen".
    A positive result, by contrast, is meaningful despite this.
    """
    from alpamayo_r1 import helper

    image_frames = frames[None].expand(N_CAMERAS, *frames.shape)   # (cams, T, 3, H, W)
    messages = helper.create_message(image_frames.flatten(0, 1))

    tokenized = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    return helper.to_device({
        "tokenized_data": tokenized,
        "ego_history_xyz": torch.from_numpy(ego_xyz)[None, None],   # (1,1,16,3)
        "ego_history_rot": torch.from_numpy(ego_rot.astype(np.float32))[None, None],
    }, device)


def predict(model, model_inputs, k=1):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=k,
            max_generation_length=256,
            return_extra=True,
        )
    return pred_xyz, pred_rot, extra


def mean_trajectory(pred_xyz):
    """[B, n_sets, n_samples, T, 3] -> (T, 3), mean over samples."""
    a = pred_xyz.float().cpu().numpy()
    return a[0, 0].mean(axis=0)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    dc, tc = cfg["data"], cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trajs = val_trajectories(dc["dataset_root"], dc["train_split"], tc["seed"])
    print(f"[Probe] {len(trajs)} val trajectories "
          f"(same split, seed {tc['seed']}, as the other three arms)")

    model, processor = load_alpamayo(args.model_id, device)
    print(f"[Probe] loaded {args.model_id}")
    torch.cuda.manual_seed_all(tc["seed"])   # trajectory sampling is stochastic

    # Candidate (traj, frame) samples with BOTH a valid lookahead at the matched
    # horizon and a full 1.6 s history, filtered up front so --n-samples is the
    # number actually scored (the first run silently scored 129 of 200).
    cands = []
    for tp in trajs:
        if not (tp / "poses.npy").exists() or not (tp / "timestamps.npy").exists():
            continue
        ts = np.load(tp / "timestamps.npy")
        for t in range(len(ts)):
            if (waypoint_target_index(ts, t, args.horizon) is not None
                    and history_index(ts, t) is not None):
                cands.append((tp, t))
    if not cands:
        raise SystemExit(
            "No val frames have a valid lookahead and history. Check that "
            "timestamps.npy exists (re-run scripts/convert_uzh_fpv.py).")
    random.Random(tc["seed"]).shuffle(cands)
    n = 1 if args.inspect else min(args.n_samples, len(cands))
    print(f"[Probe] {len(cands)} candidate frames; using {n}  "
          f"(ego-frame={args.ego_frame}, images={args.images}, "
          f"traj-samples={args.traj_samples})")

    from scipy.spatial.transform import Rotation
    rows = []
    k = args.horizon / TRAJ_DT - 1.0            # Alpamayo's grid starts at +0.1 s
    k0, k1 = int(np.floor(k)), int(np.ceil(k))
    w = float(k - k0)

    for i in range(n):
        tp, t = cands[i]
        poses = np.load(tp / "poses.npy")
        ts = np.load(tp / "timestamps.npy")
        idx = history_index(ts, t)
        R_F, ego_rot = ego_frame(poses, idx, args.ego_frame)
        ego_xyz = ego_history(poses, idx, R_F)
        frames = clip_frames(tp, ts, t)
        if args.images == "blank":
            frames = torch.zeros_like(frames)

        mi = build_inputs(processor, frames, ego_xyz, ego_rot, device)
        pred_xyz, pred_rot, extra = predict(model, mi, args.traj_samples)
        traj = mean_trajectory(pred_xyz)                 # (64, 3) in frame F
        p_F = traj[k0] * (1 - w) + traj[k1] * w

        # Everything is scored as a WORLD displacement from t0, so the choice of
        # frame changes what Alpamayo sees but not the yardstick. 3D L2 is
        # frame-invariant; "horizontal" is the gravity plane, which is where a
        # planar driving model's output lives.
        j = waypoint_target_index(ts, t, args.horizon)
        gt_w = poses[j, :3] - poses[t, :3]
        pred_w = R_F.apply(p_F)
        cv_w = constant_velocity(poses, idx, args.horizon)
        R_t = Rotation.from_quat(poses[t, 3:7])
        pred_b, gt_b = R_t.inv().apply(pred_w), R_t.inv().apply(gt_w)

        if args.inspect:
            print("\n================ STAGE 1: shapes ================")
            print(f"  frames (per camera)  : {tuple(frames.shape)} x {N_CAMERAS} cameras")
            print(f"  ego_history_xyz      : {tuple(mi['ego_history_xyz'].shape)}")
            print(f"  ego_history_rot      : {tuple(mi['ego_history_rot'].shape)}")
            tk = mi["tokenized_data"]
            for kk, v in (tk.items() if hasattr(tk, "items") else []):
                if hasattr(v, "shape"):
                    print(f"  tokenized[{kk}]".ljust(24) + f": {tuple(v.shape)}")
            print(f"  pred_xyz             : {tuple(pred_xyz.shape)}")
            print(f"  history in frame F (last 3):\n{np.round(ego_xyz[-3:], 3)}")
            print(f"  first 4 waypoints (F):\n{np.round(traj[:4], 3)}")
            print(f"  pred world @ {args.horizon}s : {np.round(pred_w, 3)}")
            print(f"  GT   world @ {args.horizon}s : {np.round(gt_w, 3)}")
            print(f"  CV   world @ {args.horizon}s : {np.round(cv_w, 3)}")
            cot = extra.get("cot") if isinstance(extra, dict) else None
            if cot is not None:
                print(f"  reasoning trace      : {str(cot[0])[:400]}")
            return

        rows.append({
            "traj": tp.name, "t": int(t),
            "mount": "down45" if "_45_" in tp.name else "forward",
            "l2_3d": float(np.linalg.norm(pred_w - gt_w)),
            "l2_horiz": float(np.linalg.norm(pred_w[:2] - gt_w[:2])),
            "cv_l2_3d": float(np.linalg.norm(cv_w - gt_w)),
            "cv_l2_horiz": float(np.linalg.norm(cv_w[:2] - gt_w[:2])),
            "pred_body": [float(x) for x in pred_b],
            "gt_body": [float(x) for x in gt_b],
            "pred_max_abs_z_F": float(np.abs(traj[:, 2]).max()),
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n}] horizontal L2 = "
                  f"{np.mean([r['l2_horiz'] for r in rows]):.4f} m "
                  f"(const-vel {np.mean([r['cv_l2_horiz'] for r in rows]):.4f} m)",
                  flush=True)

    def summarize(rs):
        P = np.array([r["pred_body"] for r in rs])
        G = np.array([r["gt_body"] for r in rs])
        corr = [float("nan") if P[:, d].std() < 1e-9 or G[:, d].std() < 1e-9
                else float(np.corrcoef(P[:, d], G[:, d])[0, 1]) for d in range(3)]
        m = lambda key: float(np.mean([r[key] for r in rs]))
        return {"n": len(rs), "l2_3d_m": m("l2_3d"), "l2_horiz_m": m("l2_horiz"),
                "const_vel_l2_3d_m": m("cv_l2_3d"),
                "const_vel_l2_horiz_m": m("cv_l2_horiz"),
                "body_corr_xyz": corr}

    res = {
        "model": args.model_id, "horizon_s": args.horizon,
        "ego_frame": args.ego_frame, "images": args.images,
        "traj_samples": args.traj_samples,
        "all": summarize(rows),
        "by_mount": {mt: summarize([r for r in rows if r["mount"] == mt])
                     for mt in ("forward", "down45")
                     if any(r["mount"] == mt for r in rows)},
        "pred_abs_z_max_mean": float(np.mean([r["pred_max_abs_z_F"] for r in rows])),
        "caveats": [
            "single forward camera clip replicated to 4 mounts (off-distribution)",
            "zero-shot, no drone finetuning",
            "8 of 12 val trajectories use the 45-deg-down camera mount",
        ],
    }
    print("\n================ PROBE RESULT ================")
    print(f"  ego-frame={args.ego_frame}  images={args.images}  "
          f"traj-samples={args.traj_samples}  horizon={args.horizon}s")
    print(f"  {'subset':<9}{'n':>5}{'L2 3D':>9}{'CV 3D':>9}"
          f"{'L2 horiz':>10}{'CV horiz':>10}   corr x/y/z (body)")
    for name, r in [("all", res["all"])] + list(res["by_mount"].items()):
        c = r["body_corr_xyz"]
        print(f"  {name:<9}{r['n']:>5}{r['l2_3d_m']:>9.4f}{r['const_vel_l2_3d_m']:>9.4f}"
              f"{r['l2_horiz_m']:>10.4f}{r['const_vel_l2_horiz_m']:>10.4f}"
              f"   {c[0]:+.2f} / {c[1]:+.2f} / {c[2]:+.2f}")
    print(f"  mean max |pred z| in F: {res['pred_abs_z_max_mean']:.4f} m")
    print("  Alpamayo must BEAT const-vel (CV) to show the driving prior adds anything.")

    out = Path(args.out or f"outputs/alpamayo/probe_{args.ego_frame}_"
               f"{args.images}_k{args.traj_samples}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": res, "samples": rows}, open(out, "w"), indent=2)
    print(f"\n[Probe] Saved -> {out}")


if __name__ == "__main__":
    main()
