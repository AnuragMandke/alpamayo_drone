"""
CPU test for body-frame waypoint target derivation (no data/model needed).

Run: python tests/test_waypoint.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation

from data.openvla_dataset import build_waypoint_targets, _single_waypoint

IDENT = [0.0, 0.0, 0.0, 1.0]   # qx,qy,qz,qw


def test_straight_line_identity_orientation():
    # Move along world +x at unit speed, no rotation: body-frame displacement
    # over `horizon` frames is purely forward, no heading change.
    T, H = 12, 5
    poses = np.zeros((T, 7), dtype=np.float64)
    poses[:, 0] = np.arange(T)        # x = 0,1,2,...
    poses[:, 3:7] = IDENT
    wp = _single_waypoint(poses, 0, H)
    assert np.allclose(wp[:3], [H, 0, 0], atol=1e-5), wp
    assert abs(wp[3]) < 1e-6, wp
    print(f"[ok] straight line: dx={wp[0]:.3f} (=horizon), dy=dz=dyaw=0")


def test_body_frame_rotates_displacement():
    # Drone yawed +90deg (facing world +y); a world +x displacement is to its
    # right, so in the body frame it should appear as -y (forward=+x_body=world+y).
    T, H = 8, 3
    poses = np.zeros((T, 7), dtype=np.float64)
    poses[:, 0] = np.arange(T)                       # move along world +x
    poses[:, 3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    wp = _single_waypoint(poses, 0, H)
    assert np.allclose(wp[:2], [0.0, -H], atol=1e-5), wp
    print(f"[ok] yawed frame: world +x disp -> body {np.round(wp[:2],3)}")


def test_pure_yaw_change():
    # Stationary position, heading rotates by +0.5 rad over the horizon.
    T, H = 8, 4
    poses = np.zeros((T, 7), dtype=np.float64)
    for t in range(T):
        poses[t, 3:7] = Rotation.from_euler("z", 0.5 * (t / H)).as_quat()
    wp = _single_waypoint(poses, 0, H)
    assert np.allclose(wp[:3], 0.0, atol=1e-6), wp
    assert abs(wp[3] - 0.5) < 1e-5, wp
    print(f"[ok] pure yaw: dyaw={wp[3]:.4f} (~0.5), no translation")


def test_batch_matches_single_and_shape():
    rng = np.random.default_rng(0)
    T, H = 20, 6
    poses = np.zeros((T, 7), dtype=np.float64)
    poses[:, :3] = np.cumsum(rng.normal(size=(T, 3)), axis=0)
    quats = Rotation.from_euler("zyx", rng.normal(scale=0.3, size=(T, 3))).as_quat()
    poses[:, 3:7] = quats
    full = build_waypoint_targets(poses, H)
    assert full.shape == (T - H, 4), full.shape
    for t in (0, 5, T - H - 1):
        assert np.allclose(full[t], _single_waypoint(poses, t, H), atol=1e-5)
    print(f"[ok] build_waypoint_targets shape {full.shape}; matches _single_waypoint")


if __name__ == "__main__":
    test_straight_line_identity_orientation()
    test_body_frame_rotates_displacement()
    test_pure_yaw_change()
    test_batch_matches_single_and_shape()
    print("\nALL WAYPOINT TESTS PASSED")
