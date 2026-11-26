"""Dense reward from forward kinematics, for ``rlt-local online --reward-plugin
armnet_rlt.dense_reward:shape``.

    r_t = -alpha * ||p_t - b|| - gamma * 1[trip] + R * 1[pressed]

p_t is the gripper tip (``gripper_frame_link`` of so101_new_calib.urdf) from the joint
state in degrees, b the green button, linear in the commanded rail position with
per-cell constants fitted to the tip position at the end of successful episodes
(2,618 successes of the ID and OOD rollout datasets; median residual 1.2-1.4 cm).
The plugin adds the distance term to every transition, using the joint state the
transition ends in; the trip and press terms are the algorithm's own reward.
``reward`` computes all three terms for a world model's predicted joint states.

Environment:
  DENSE_ALPHA   weight of the distance term, per metre (0.2)
"""

from __future__ import annotations

import os

import numpy as np
import torch

# (xyz, rpy) of the shoulder_pan ... wrist_roll joint origins; every axis is z.
CHAIN = (
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
)
TIP = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))

# Button position in the robot base frame (m): offset + rail * slope.
BUTTON = {
    1: ((0.2128, -0.0617, -0.0140), (0.0548, 0.2232, 0.0306)),
    3: ((0.2412, -0.1014, 0.0511), (0.0380, 0.2793, -0.0244)),
}


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin(xyz, rpy) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rpy(*rpy)
    m[:3, 3] = xyz
    return m


_ORIGINS = [_origin(*joint) for joint in CHAIN]
_TIP = _origin(*TIP)


def tip_position(joints_deg) -> np.ndarray:
    """Gripper tip (m) for joint states in degrees, shape ``(..., >=5)`` -> ``(..., 3)``."""
    q = np.radians(np.asarray(joints_deg, dtype=np.float64)[..., :5])
    pose = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for index, origin in enumerate(_ORIGINS):
        c, s = np.cos(q[..., index]), np.sin(q[..., index])
        rot = np.zeros(q.shape[:-1] + (4, 4))
        rot[..., 0, 0], rot[..., 0, 1], rot[..., 1, 0], rot[..., 1, 1] = c, -s, s, c
        rot[..., 2, 2] = rot[..., 3, 3] = 1.0
        pose = pose @ origin @ rot
    return (pose @ _TIP)[..., :3, 3]


def cell_number(cell: str | int) -> int:
    """3 for ``cell-03`` or 3."""
    if isinstance(cell, (int, np.integer)):
        return int(cell)
    digits = "".join(ch for ch in str(cell) if ch.isdigit())
    if not digits:
        raise ValueError(f"no cell number in {cell!r}")
    return int(digits)


def button_position(rail: float, cell: str | int) -> np.ndarray:
    """Green button (m) at commanded rail position ``rail``."""
    number = cell_number(cell)
    if number not in BUTTON:
        raise KeyError(f"no button calibration for cell {number}")
    offset, slope = BUTTON[number]
    return np.asarray(offset) + float(rail) * np.asarray(slope)


def distance(joints_deg, rail: float, cell: str | int) -> np.ndarray:
    """Tip-to-button distance (m) for joint states ``(..., >=5)``."""
    return np.linalg.norm(tip_position(joints_deg) - button_position(rail, cell), axis=-1)


def reward(
    joints_deg,
    rail: float,
    cell: str | int,
    *,
    pressed=False,
    tripped=False,
    alpha: float | None = None,
    gamma: float = 1.0,
    press_reward: float = 1.0,
) -> np.ndarray:
    """All three terms, per step, for joint states ``(..., >=5)``."""
    alpha = float(os.environ.get("DENSE_ALPHA", 0.2)) if alpha is None else alpha
    return (
        -alpha * distance(joints_deg, rail, cell)
        - gamma * np.asarray(tripped, dtype=np.float64)
        + press_reward * np.asarray(pressed, dtype=np.float64)
    )


def shape(episode: list[dict[str, torch.Tensor]], frames=None, scene: dict | None = None) -> None:
    """Adds ``-alpha * distance`` of each transition's end state to its reward."""
    if not episode:
        return
    if not scene or scene.get("rail") is None or scene.get("cell") is None:
        print("[dense-reward] no rail or cell for this episode; rewards unchanged", flush=True)
        return
    alpha = float(os.environ.get("DENSE_ALPHA", 0.2))
    joints = torch.stack([fields["next_proprioception"] for fields in episode]).numpy()
    gaps = distance(joints, float(scene["rail"]), scene["cell"])
    for fields, gap in zip(episode, gaps):
        fields["reward"] = fields["reward"] + torch.tensor(-alpha * float(gap), dtype=torch.float32)
