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
    python scripts/probe_alpamayo.py --n-samples 200

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

from data.openvla_dataset import _body_frame_delta, waypoint_target_index

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
    p.add_argument("--out", default="outputs/alpamayo/probe_alpamayo.json")
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


def ego_history(poses, timestamps, t):
    """(N,3) positions and (N,3,3) rotations over the last EGO_HISTORY_STEPS,
    expressed in the ego frame at time t (so the last entry is the origin /
    identity). Returns (None, None) if the history runs off the start of the clip.

    Frames are picked by CLOCK TIME, not frame count — UZH-FPV drops frames, and
    a fixed frame count would give a variable-duration history. Same reasoning as
    waypoint_horizon_seconds in the main pipeline.
    """
    from scipy.spatial.transform import Rotation

    n = EGO_HISTORY_STEPS
    want = timestamps[t] - np.arange(n - 1, -1, -1) / EGO_HISTORY_HZ
    if want[0] < timestamps[0]:
        return None, None                                   # not enough history
    idx = [int(np.argmin(np.abs(timestamps - w))) for w in want]

    R = Rotation.from_quat(poses[:, 3:7])
    R_t_inv = R[t].inv()
    xyz = np.stack([R_t_inv.apply(poses[i, :3] - poses[t, :3]) for i in idx])
    rot = np.stack([(R_t_inv * R[i]).as_matrix() for i in idx])
    return xyz.astype(np.float32), rot.astype(np.float32)


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


def constant_velocity(ego_xyz, horizon, window=2):
    """The no-vision baseline Alpamayo has to beat. ego_xyz is in the t0 frame
    with the last entry at the origin, so the mean velocity over the last
    `window` steps is -ego_xyz[-1-window] / (window * dt). Alpamayo sees this
    same history, so matching it on forward motion alone proves nothing about
    the driving prior; only beating it does."""
    v = -ego_xyz[-1 - window] / (window / EGO_HISTORY_HZ)
    return v * horizon


def gt_future(poses, timestamps, t, horizons):
    """Body-frame [dx,dy,dz,dyaw] at each horizon in `horizons` (seconds).
    None entries where the clip ends before that horizon."""
    out = []
    for h in horizons:
        j = waypoint_target_index(timestamps, t, h)
        out.append(None if j is None else _body_frame_delta(poses, t, j))
    return out


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
        "ego_history_rot": torch.from_numpy(ego_rot)[None, None],   # (1,1,16,3,3)
    }, device)


def predict(model, model_inputs):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    return pred_xyz, pred_rot, extra


def first_trajectory(pred_xyz):
    """[B, n_sets, n_samples, T, 3] -> (T, 3) for the first sample."""
    a = pred_xyz.float().cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    return a


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

    # Candidate (traj, frame) samples that have a valid lookahead at the matched
    # horizon. Ego-history availability is checked per sample below.
    cands = []
    for tp in trajs:
        if not (tp / "poses.npy").exists() or not (tp / "timestamps.npy").exists():
            continue
        ts = np.load(tp / "timestamps.npy")
        for t in range(len(ts)):
            if waypoint_target_index(ts, t, args.horizon) is not None:
                cands.append((tp, t))
    if not cands:
        raise SystemExit(
            "No val frames have a valid lookahead. Check that timestamps.npy "
            "exists (re-run scripts/convert_uzh_fpv.py).")
    random.Random(tc["seed"]).shuffle(cands)
    n = 1 if args.inspect else min(args.n_samples, len(cands))
    print(f"[Probe] {len(cands)} candidate frames; using {n}")

    rows, z_pred, l2_matched, l2_planar = [], [], [], []
    cv_3d, cv_planar = [], []

    for i in range(n):
        tp, t = cands[i]
        poses = np.load(tp / "poses.npy")
        ts = np.load(tp / "timestamps.npy")
        ego_xyz, ego_rot = ego_history(poses, ts, t)
        if ego_xyz is None:
            continue
        frames = clip_frames(tp, ts, t)
        if frames is None:
            continue

        mi = build_inputs(processor, frames, ego_xyz, ego_rot, device)
        pred_xyz, pred_rot, extra = predict(model, mi)

        if args.inspect:
            print("\n================ STAGE 1: shapes ================")
            print(f"  frames (per camera)  : {tuple(frames.shape)} x {N_CAMERAS} cameras")
            print(f"  ego_history_xyz      : {tuple(mi['ego_history_xyz'].shape)}")
            print(f"  ego_history_rot      : {tuple(mi['ego_history_rot'].shape)}")
            tk = mi["tokenized_data"]
            for k, v in (tk.items() if hasattr(tk, "items") else []):
                if hasattr(v, "shape"):
                    print(f"  tokenized[{k}]".ljust(24) + f": {tuple(v.shape)}")
            print(f"  pred_xyz             : {tuple(pred_xyz.shape)}")
            print(f"  pred_rot             : {tuple(pred_rot.shape)}")
            traj = first_trajectory(pred_xyz)
            print(f"  first trajectory     : {traj.shape}  "
                  f"(expect ~(64, 3) = 6.4s @ {TRAJ_DT}s)")
            print(f"  first 4 waypoints    :\n{np.round(traj[:4], 3)}")
            print(f"  z range over traj    : [{traj[:, 2].min():.3f}, "
                  f"{traj[:, 2].max():.3f}]  <- ~0 means planar/unicycle only")
            gt = gt_future(poses, ts, t, [args.horizon])[0]
            print(f"  GT body-frame @ {args.horizon}s : {np.round(gt, 3)} "
                  f"[dx, dy, dz, dyaw]")
            cot = extra.get("cot") if isinstance(extra, dict) else None
            if cot is not None:
                print(f"  reasoning trace      : {str(cot[0])[:400]}")
            print("\nCHECK BEFORE TRUSTING ANY NUMBER:")
            print("  * Does pred_xyz's frame match our body frame (x forward)?")
            print("    Compare the signs above against the GT row.")
            print("  * Is z genuinely predicted, or pinned at 0 (unicycle)?")
            print("  * Is the trajectory 64 long at 0.1s spacing as documented?")
            return

        traj = first_trajectory(pred_xyz)
        z_pred.append(float(np.abs(traj[:, 2]).max()))

        # Matched horizon: linear interpolation on Alpamayo's 0.1s grid.
        k = args.horizon / TRAJ_DT - 1.0
        k0 = int(np.clip(np.floor(k), 0, len(traj) - 1))
        k1 = int(np.clip(np.ceil(k), 0, len(traj) - 1))
        w = float(k - k0)
        p = traj[k0] * (1 - w) + traj[k1] * w
        g = gt_future(poses, ts, t, [args.horizon])[0]
        if g is not None:
            l2_matched.append(float(np.linalg.norm(p[:3] - g[:3])))
            l2_planar.append(float(np.linalg.norm(p[:2] - g[:2])))
            c = constant_velocity(ego_xyz, args.horizon)
            cv_3d.append(float(np.linalg.norm(c - g[:3])))
            cv_planar.append(float(np.linalg.norm(c[:2] - g[:2])))
            rows.append({"traj": tp.name, "t": t,
                         "pred_xyz": [float(x) for x in p],
                         "gt_dxdydz": [float(x) for x in g[:3]],
                         "gt_dyaw": float(g[3])})

        if (i + 1) % 25 == 0 and l2_planar:
            print(f"  [{i+1}/{n}] running planar L2 = {np.mean(l2_planar):.4f} m "
                  f"(const-vel {np.mean(cv_planar):.4f} m)")

    if not l2_matched:
        raise SystemExit("No scorable samples — every candidate lacked ego history.")

    # Axis-convention diagnostic: if Alpamayo's frame does not match ours, the
    # correlation between predicted and GT forward motion collapses (or flips
    # sign). That is a coordinate bug, NOT a transfer result — check it before
    # concluding anything about the driving prior.
    P = np.array([r["pred_xyz"] for r in rows])
    G = np.array([r["gt_dxdydz"] for r in rows])
    corr = []
    for d in range(3):
        if P[:, d].std() < 1e-9 or G[:, d].std() < 1e-9:
            corr.append(float("nan"))        # constant column (e.g. planar z)
        else:
            corr.append(float(np.corrcoef(P[:, d], G[:, d])[0, 1]))

    res = {
        "model": args.model_id,
        "n_scored": len(l2_matched),
        "horizon_s": args.horizon,
        "l2_3d_m": float(np.mean(l2_matched)),
        "l2_planar_m": float(np.mean(l2_planar)),
        "const_vel_l2_3d_m": float(np.mean(cv_3d)),
        "const_vel_l2_planar_m": float(np.mean(cv_planar)),
        "pred_abs_z_max_mean": float(np.mean(z_pred)),
        "per_axis_corr_pred_vs_gt": {"x": corr[0], "y": corr[1], "z": corr[2]},
        "caveats": [
            "single forward camera clip replicated to 4 mounts (off-distribution)",
            "zero-shot, no drone finetuning",
            "unicycle output model may not express vertical motion",
        ],
    }
    print("\n================ PROBE RESULT ================")
    print(f"  scored samples        : {res['n_scored']}")
    print(f"  L2 @ {args.horizon}s (3D)      : {res['l2_3d_m']:.4f} m")
    print(f"  L2 @ {args.horizon}s (planar)  : {res['l2_planar_m']:.4f} m")
    print(f"  const-vel baseline    : {res['const_vel_l2_planar_m']:.4f} m planar, "
          f"{res['const_vel_l2_3d_m']:.4f} m 3D   <- Alpamayo must BEAT this")
    print(f"  mean max |pred z|     : {res['pred_abs_z_max_mean']:.4f} m"
          f"   <- ~0 => planar only, dz unreachable")
    print(f"  corr(pred, gt) x/y/z  : {corr[0]:+.3f} / {corr[1]:+.3f} / {corr[2]:+.3f}")
    print("  NOTE: a near-zero or negative x-correlation means the coordinate "
          "frames disagree — fix that before reading L2 as a transfer result.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": res, "samples": rows[:50]}, open(out, "w"), indent=2)
    print(f"\n[Probe] Saved -> {out}")


if __name__ == "__main__":
    main()
