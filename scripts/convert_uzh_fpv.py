"""
scripts/convert_uzh_fpv.py — Convert UZH-FPV drone racing sequences to the
AirSimDroneDataset trajectory format.

Input (one UZH-FPV sequence folder):
    <sequence>/
        img/             # grayscale frames ~30 Hz, frame_XXXXXXXXXX.png
        groundtruth.txt  # ~100 Hz poses: timestamp x y z qx qy qz qw
        events.txt, imu/ ...   (ignored)

Output (per chunk):
    <dst>/trajectories/
        traj_{sequence_name}_t{i:03d}/
            images/          # rgb_000.png ... (224x224 RGB)
            actions.npy      # (T, 4) float32  [vx, vy, vz, yaw_rate]
            instructions.txt # one natural-language goal per line

Note: chunk names follow {sequence_name}_t{i:03d}; the directory gets a
"traj_" prefix because AirSimDroneDataset discovers trajectories via the
glob "traj_*".

Usage:
    python scripts/convert_uzh_fpv.py \
        --src data/raw/uzh_fpv/ \
        --dst data/airsim \
        --chunk_size 120 \
        --max_sequences 999 \
        --dry_run

--src accepts either a single sequence folder or a parent directory
containing multiple sequence folders.

Standalone: depends only on numpy, scipy, Pillow, tqdm (no project imports).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
VEL_CLIP = 5.0            # m/s for vx, vy, vz
YAW_RATE_CLIP = 3.14      # rad/s
MAX_ALIGN_DIST_S = 0.05   # max image-to-GT timestamp distance
MAX_BAD_FRAC = 0.10       # skip chunk if more frames than this misalign

INSTRUCTION_TEMPLATES = [
    ("indoor_forward", [
        "Navigate forward through the indoor corridor.",
        "Fly forward quickly through the building.",
        "Move ahead through the indoor space.",
    ]),
    ("indoor_45", [
        "Turn and navigate through the indoor space.",
        "Execute a banked turn through the corridor.",
    ]),
    ("outdoor_forward", [
        "Navigate forward in the outdoor environment.",
        "Fly forward through the outdoor course.",
    ]),
    ("outdoor_45", [
        "Execute a turning maneuver outdoors.",
        "Navigate the outdoor course with a banked turn.",
    ]),
]
DEFAULT_INSTRUCTIONS = [
    "Navigate to the goal.",
    "Fly to the target position.",
]


def instructions_for_sequence(sequence_name: str) -> list:
    for substring, instructions in INSTRUCTION_TEMPLATES:
        if substring in sequence_name:
            return instructions
    return DEFAULT_INSTRUCTIONS


# ------------------------------------------------------------------
# Groundtruth parsing and action derivation
# ------------------------------------------------------------------

def parse_groundtruth(gt_path: Path):
    """Parse groundtruth.txt → (timestamps (T,), positions (T,3), quats (T,4) xyzw)."""
    try:
        rows = []
        with open(gt_path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    print(f"  [Warn] {gt_path}:{line_no}: expected 8 columns, "
                          f"got {len(parts)} — skipping line")
                    continue
                rows.append([float(v) for v in parts[:8]])
    except OSError as e:
        raise RuntimeError(f"Failed to read groundtruth file {gt_path}: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Malformed numeric value in {gt_path}: {e}") from e

    if len(rows) < 2:
        raise RuntimeError(
            f"{gt_path} contains {len(rows)} pose rows; need at least 2 "
            "to derive velocities."
        )

    data = np.asarray(rows, dtype=np.float64)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quats = data[:, 4:8]  # xyzw

    # Drop non-increasing timestamps (would give dt <= 0)
    keep = np.concatenate([[True], np.diff(timestamps) > 0])
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"  [Warn] Dropped {n_dropped} non-increasing timestamp rows "
              f"from {gt_path.name}")
        timestamps, positions, quats = timestamps[keep], positions[keep], quats[keep]

    return timestamps, positions, quats


def derive_actions(timestamps, positions, quats):
    """
    Derive body-frame velocity actions from world-frame poses.

    Returns (T-1, 4) float32: [vx, vy, vz, yaw_rate].
    """
    dt = np.diff(timestamps)                      # (T-1,)
    vel_world = np.diff(positions, axis=0) / dt[:, None]  # (T-1, 3)

    rotations = Rotation.from_quat(quats)         # xyzw convention

    n = len(dt)
    actions = np.zeros((n, 4), dtype=np.float32)
    rot_mats = rotations.as_matrix()              # (T, 3, 3)
    for i in range(n):
        R = rot_mats[i]
        actions[i, :3] = R.T @ vel_world[i]
        delta_r = rotations[i].inv() * rotations[i + 1]
        actions[i, 3] = delta_r.as_euler("zyx")[0] / dt[i]

    actions[:, :3] = np.clip(actions[:, :3], -VEL_CLIP, VEL_CLIP)
    actions[:, 3] = np.clip(actions[:, 3], -YAW_RATE_CLIP, YAW_RATE_CLIP)
    return actions


# ------------------------------------------------------------------
# Image discovery and timestamp alignment
# ------------------------------------------------------------------

def discover_images(img_dir: Path):
    """
    Find frame_*.png images and parse timestamps from filenames.

    Returns (image_paths sorted by timestamp, timestamps (N,) float64 in the
    raw filename unit — rescaled later against the groundtruth clock).
    """
    try:
        files = sorted(img_dir.glob("frame_*.png"))
    except OSError as e:
        raise RuntimeError(f"Failed to list images in {img_dir}: {e}") from e

    paths, stamps = [], []
    pattern = re.compile(r"frame_(\d+(?:\.\d+)?)")
    for f in files:
        m = pattern.search(f.stem)
        if not m:
            print(f"  [Warn] Cannot parse timestamp from {f.name} — skipping")
            continue
        paths.append(f)
        stamps.append(float(m.group(1)))

    if not paths:
        raise RuntimeError(f"No parsable frame_*.png images found in {img_dir}")

    order = np.argsort(stamps)
    paths = [paths[i] for i in order]
    stamps = np.asarray(stamps, dtype=np.float64)[order]
    return paths, stamps


def rescale_image_timestamps(img_ts, gt_ts):
    """
    Filename timestamps may be in seconds, milli/micro/nanoseconds, or use a
    different epoch than groundtruth. Pick the unit scale that best matches
    the groundtruth clock, then remove any constant offset if the spans
    match but the epochs differ.
    """
    gt_med = np.median(gt_ts)
    best_scale, best_err = 1.0, np.inf
    for scale in (1.0, 1e-3, 1e-6, 1e-9):
        scaled_med = np.median(img_ts) * scale
        err = abs(scaled_med - gt_med)
        if err < best_err:
            best_scale, best_err = scale, err
    scaled = img_ts * best_scale

    # Different epoch (e.g. frame index counter vs. absolute time): align
    # midpoints if the scaled timestamps don't overlap the GT range at all.
    if scaled.max() < gt_ts.min() or scaled.min() > gt_ts.max():
        scaled = scaled - np.median(scaled) + gt_med
        print("  [Warn] Image and groundtruth clocks share no overlap; "
              "aligned by midpoint — verify alignment quality below.")
    return scaled


def align_images_to_gt(img_ts, gt_ts):
    """
    Nearest-neighbour alignment via np.searchsorted.

    Returns (gt_indices (N,) int, distances (N,) float seconds).
    """
    idx = np.searchsorted(gt_ts, img_ts)
    idx = np.clip(idx, 1, len(gt_ts) - 1)
    left, right = gt_ts[idx - 1], gt_ts[idx]
    use_left = (img_ts - left) <= (right - img_ts)
    nearest = np.where(use_left, idx - 1, idx)
    dist = np.abs(gt_ts[nearest] - img_ts)
    return nearest, dist


# ------------------------------------------------------------------
# Sequence conversion
# ------------------------------------------------------------------

def convert_sequence(seq_dir: Path, dst_root: Path, chunk_size: int,
                     dry_run: bool):
    """
    Convert one UZH-FPV sequence into chunked trajectories.

    Returns (chunks_written, frames_written).
    """
    sequence_name = seq_dir.name
    print(f"\n[Convert] Sequence: {sequence_name}")

    gt_ts, positions, quats = parse_groundtruth(seq_dir / "groundtruth.txt")
    actions_full = derive_actions(gt_ts, positions, quats)   # (T-1, 4)

    img_paths, img_ts_raw = discover_images(seq_dir / "img")
    img_ts = rescale_image_timestamps(img_ts_raw, gt_ts)
    gt_idx, align_dist = align_images_to_gt(img_ts, gt_ts)

    # Action for frame i = body-frame velocity at its nearest GT pose.
    # actions_full has T-1 rows, so clamp the last pose index.
    action_idx = np.minimum(gt_idx, len(actions_full) - 1)

    n_frames = len(img_paths)
    min_chunk = chunk_size // 2
    instructions = instructions_for_sequence(sequence_name)

    chunks_written = 0
    frames_written = 0
    chunk_i = 0

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        length = end - start
        if length < min_chunk:
            print(f"  [Skip] Final chunk has {length} frames "
                  f"(< {min_chunk}) — discarded")
            break

        chunk_name = f"{sequence_name}_t{chunk_i:03d}"
        chunk_i += 1

        bad = align_dist[start:end] > MAX_ALIGN_DIST_S
        bad_frac = bad.mean()
        if bad_frac > MAX_BAD_FRAC:
            print(f"  [Skip] {chunk_name}: {bad_frac:.1%} of frames have "
                  f"nearest-GT distance > {MAX_ALIGN_DIST_S}s")
            continue

        chunk_actions = actions_full[action_idx[start:end]].astype(np.float32)
        traj_dir = dst_root / f"traj_{chunk_name}"

        if dry_run:
            print(f"  [DryRun] Would write {traj_dir}  "
                  f"({length} frames, actions {chunk_actions.shape})")
            chunks_written += 1
            frames_written += length
            continue

        img_out = traj_dir / "images"
        try:
            img_out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"  [Error] Cannot create {img_out}: {e} — skipping chunk")
            continue

        try:
            for t, frame_path in enumerate(
                tqdm(img_paths[start:end], desc=f"  {chunk_name}",
                     unit="img", leave=False)
            ):
                img = Image.open(frame_path).convert("RGB")
                img = img.resize(IMAGE_SIZE, Image.BILINEAR)
                img.save(img_out / f"rgb_{t:03d}.png")

            np.save(traj_dir / "actions.npy", chunk_actions)
            (traj_dir / "instructions.txt").write_text(
                "\n".join(instructions), encoding="utf-8"
            )
        except OSError as e:
            print(f"  [Error] Failed writing chunk {chunk_name}: {e} "
                  f"— skipping chunk")
            continue

        chunks_written += 1
        frames_written += length
        print(f"  [OK] {traj_dir.name}: {length} frames, "
              f"actions {chunk_actions.shape}")

    return chunks_written, frames_written


# ------------------------------------------------------------------
# Sequence discovery
# ------------------------------------------------------------------

def is_sequence_dir(path: Path) -> bool:
    return (path / "groundtruth.txt").is_file() and (path / "img").is_dir()


def find_sequences(src: Path) -> list:
    """--src is either one sequence folder or a parent of sequence folders."""
    if not src.exists():
        raise RuntimeError(f"--src path does not exist: {src}")
    if is_sequence_dir(src):
        return [src]
    try:
        candidates = sorted(p for p in src.iterdir() if p.is_dir())
    except OSError as e:
        raise RuntimeError(f"Failed to list {src}: {e}") from e
    sequences = [p for p in candidates if is_sequence_dir(p)]
    if not sequences:
        raise RuntimeError(
            f"No UZH-FPV sequences found under {src}. A sequence folder "
            "must contain groundtruth.txt and an img/ directory."
        )
    return sequences


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert UZH-FPV sequences to AirSimDroneDataset format."
    )
    p.add_argument("--src", required=True,
                   help="UZH-FPV sequence folder or parent directory")
    p.add_argument("--dst", default="data/airsim",
                   help="Output dataset root (trajectories/ created inside)")
    p.add_argument("--chunk_size", type=int, default=120,
                   help="Frames per trajectory chunk")
    p.add_argument("--max_sequences", type=int, default=999,
                   help="Maximum number of sequences to convert")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be written without writing")
    return p.parse_args()


def main():
    args = parse_args()

    if args.chunk_size < 2:
        print(f"[Error] --chunk_size must be >= 2, got {args.chunk_size}")
        sys.exit(1)

    try:
        sequences = find_sequences(Path(args.src))
    except RuntimeError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    sequences = sequences[: args.max_sequences]
    dst_root = Path(args.dst) / "trajectories"

    if not args.dry_run:
        try:
            dst_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[Error] Cannot create output directory {dst_root}: {e}")
            sys.exit(1)

    n_processed = 0
    total_chunks = 0
    total_frames = 0

    for seq_dir in sequences:
        try:
            chunks, frames = convert_sequence(
                seq_dir, dst_root, args.chunk_size, args.dry_run
            )
        except RuntimeError as e:
            print(f"[Error] Skipping sequence {seq_dir.name}: {e}")
            continue
        n_processed += 1
        total_chunks += chunks
        total_frames += frames

    print()
    print(f"[Convert] Sequences processed : {n_processed}")
    print(f"[Convert] Chunks written       : {total_chunks}")
    print(f"[Convert] Total frames         : {total_frames}")
    print(f"[Convert] Output               : {dst_root}/")
    if args.dry_run:
        print("[Convert] (dry run — nothing was written)")


if __name__ == "__main__":
    main()
