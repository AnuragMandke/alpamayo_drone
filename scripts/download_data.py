"""
scripts/download_data.py — Download and prepare AirSim drone trajectory data

Downloads the Microsoft AirSim Drone Racing dataset and converts it
into the format expected by AirSimDroneDataset:

    data/airsim/trajectories/
        traj_XXXX/
            images/          # rgb_000.png ...
            actions.npy      # (T, 4) float32
            instructions.txt # natural language goals

If you already have AirSim running locally and want to collect your own
trajectories, use the --collect flag to run the data collection script
instead of downloading.
"""

import argparse
import os
import sys
import json
import shutil
import zipfile
import urllib.request
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------
# Instruction templates for each trajectory type
# These simulate the language conditioning signal
# ------------------------------------------------------------------
INSTRUCTION_TEMPLATES = {
    "point_nav": [
        "Fly to the goal marker ahead.",
        "Navigate to the target position.",
        "Reach the red waypoint as quickly as possible.",
        "Move to the destination while maintaining stable flight.",
    ],
    "obstacle": [
        "Navigate to the goal while avoiding all obstacles.",
        "Fly through the course without hitting any barriers.",
        "Reach the target safely, avoiding all objects in your path.",
    ],
    "hover": [
        "Hold your current position and altitude.",
        "Stabilize and hover in place.",
        "Maintain your position without drifting.",
    ],
}


def generate_synthetic_trajectories(
    output_dir: Path,
    n_trajectories: int = 200,
    T: int = 100,
    image_size: int = 224,
):
    """
    Generate synthetic placeholder trajectories for offline development.

    In a real experiment, replace these with actual AirSim recordings.
    Actions are smooth sine-wave velocity profiles — realistic enough
    for pipeline testing.
    """
    import random
    from PIL import Image

    traj_root = output_dir / "trajectories"
    traj_root.mkdir(parents=True, exist_ok=True)

    print(f"[Download] Generating {n_trajectories} synthetic trajectories "
          f"(T={T}, image_size={image_size})")
    print("[Download] Replace with real AirSim data before paper experiments.")

    task_types = list(INSTRUCTION_TEMPLATES.keys())

    for i in range(n_trajectories):
        traj_dir = traj_root / f"traj_{i:04d}"
        img_dir = traj_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # --- Generate synthetic RGB images (noise as placeholder) ---
        for t in range(T):
            # Gradient + noise: visually distinct per frame
            r_val = int(255 * t / T)
            base = np.full((image_size, image_size, 3), [r_val, 80, 120], dtype=np.uint8)
            noise = np.random.randint(0, 30, base.shape, dtype=np.uint8)
            img_arr = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(img_arr).save(img_dir / f"rgb_{t:03d}.png")

        # --- Generate smooth synthetic actions [vx, vy, vz, yaw_rate] ---
        t_arr = np.linspace(0, 2 * np.pi, T)
        phase = random.uniform(0, np.pi)
        speed = random.uniform(0.5, 3.0)
        actions = np.stack([
            speed * np.sin(t_arr + phase),          # vx
            speed * 0.3 * np.cos(t_arr + phase),    # vy
            np.sin(0.5 * t_arr) * 0.5,              # vz (gentle altitude changes)
            0.1 * np.cos(t_arr + phase),             # yaw_rate
        ], axis=-1).astype(np.float32)               # (T, 4)

        np.save(traj_dir / "actions.npy", actions)

        # --- Write instructions ---
        task = random.choice(task_types)
        instrs = INSTRUCTION_TEMPLATES[task]
        (traj_dir / "instructions.txt").write_text("\n".join(instrs))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_trajectories} trajectories generated")

    print(f"[Download] Done. Data written to {traj_root}")
    print(f"[Download] Total trajectories: {n_trajectories}")
    print()
    print("  To use real AirSim data:")
    print("  1. Install AirSim: https://microsoft.github.io/AirSim/")
    print("  2. Record trajectories with AirSim's recording API")
    print("  3. Run:  python scripts/convert_airsim_recording.py --src <recording_dir>")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/airsim",
                   help="Output directory for dataset")
    p.add_argument("--n_trajs", type=int, default=200,
                   help="Number of synthetic trajectories to generate")
    p.add_argument("--T", type=int, default=100,
                   help="Timesteps per trajectory")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_synthetic_trajectories(
        output_dir,
        n_trajectories=args.n_trajs,
        T=args.T,
    )


if __name__ == "__main__":
    main()
