"""
CPU test for body-frame waypoint target derivation (no data/model needed).

Run: python tests/test_waypoint.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation

from data.openvla_dataset import (
    build_waypoint_targets, _single_waypoint, _body_frame_delta,
    waypoint_target_index, waypoint_pairs,
)

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


def test_time_index_picks_nearest_forward_frame():
    # Uniform 0.1s frames; a 0.25s horizon from t=0 lands between frames 2 (0.2s)
    # and 3 (0.3s) — 0.25 is equidistant, and the tie-break prefers the earlier.
    ts = np.arange(10) * 0.1
    assert waypoint_target_index(ts, 0, 0.25) == 2, "tie -> earlier frame"
    assert waypoint_target_index(ts, 0, 0.31) == 3, "0.31s -> frame 3 (0.3s)"
    # Past the end of the trajectory -> no valid lookahead.
    assert waypoint_target_index(ts, 8, 0.5) is None
    print("[ok] time index: nearest forward frame, None past the end")


def test_time_index_survives_dropped_frames():
    # A gap between 0.2s and 0.6s (a dropped stretch): a 0.3s horizon from t=0
    # (target 0.3s) must snap to the nearest ACTUAL frame, not a fixed +N offset.
    ts = np.array([0.0, 0.1, 0.2, 0.6, 0.7, 0.8])
    #                0    1    2    3    4    5
    j = waypoint_target_index(ts, 0, 0.3)   # target 0.3s; nearest is 0.2 (frame 2)
    assert j == 2, j
    j2 = waypoint_target_index(ts, 2, 0.3)  # from 0.2s, target 0.5s; nearest 0.6 (3)
    assert j2 == 3, j2
    print("[ok] time index: snaps across a dropped-frame gap")


def test_time_based_targets_match_geometry_and_pairs():
    rng = np.random.default_rng(1)
    T = 30
    poses = np.zeros((T, 7), dtype=np.float64)
    poses[:, :3] = np.cumsum(rng.normal(size=(T, 3)), axis=0)
    poses[:, 3:7] = Rotation.from_euler(
        "zyx", rng.normal(scale=0.3, size=(T, 3))).as_quat()
    # Irregular timestamps (variable dt), the realistic case.
    ts = np.cumsum(rng.uniform(0.02, 0.08, size=T))
    horizon_s = 0.15

    pairs = waypoint_pairs(ts, horizon_s)
    assert pairs and pairs[0][0] == 0
    targets = build_waypoint_targets(poses, 0, ts, horizon_s)
    assert targets.shape == (len(pairs), 4), (targets.shape, len(pairs))
    for row, (t, t_tgt) in zip(targets, pairs):
        assert np.allclose(row, _body_frame_delta(poses, t, t_tgt), atol=1e-5)
    # Every target frame is strictly ahead of its source.
    assert all(t_tgt > t for t, t_tgt in pairs)
    print(f"[ok] time-based targets: {len(pairs)} pairs, all forward, match geometry")


if __name__ == "__main__":
    test_straight_line_identity_orientation()
    test_body_frame_rotates_displacement()
    test_pure_yaw_change()
    test_batch_matches_single_and_shape()
    test_time_index_picks_nearest_forward_frame()
    test_time_index_survives_dropped_frames()
    test_time_based_targets_match_geometry_and_pairs()
    print("\nALL WAYPOINT TESTS PASSED")
