"""A small motion library and MDP of the mjlab G1 on the CPU, built from generated clips at 50 Hz.

Clips (rows): walk (0-5) upright at 0.76 m walking along x while turning; low (6-9) upright at 0.40 m; tilted (10-13)
at 0.76 m rolled 90 degrees about x; lying (14-17) at 0.15 m pitched 90 degrees about y.
"""

import math

import torch

from blinkmimic2.dr import XiLayout
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import MDP
from blinkmimic2.motion import CONTROL_DT, MotionLib
from blinkmimic2.robot import get_robot

ROBOT = get_robot("mjlab_g1")
CLIPS = {"walk": 6, "low": 4, "tilted": 4, "lying": 4}


def _clip(n: int, z: float, axis: int | None, j: torch.Tensor, walk: bool):
    t = torch.arange(n, dtype=torch.float32)
    p = torch.stack((0.5 * 0.02 * t if walk else torch.zeros(n), torch.zeros(n), torch.full((n,), z)), -1)
    quat = torch.zeros(n, 4)
    if walk:  # yaw 0.3 rad/s
        quat[:, 0], quat[:, 3] = torch.cos(0.15 * 0.02 * t), torch.sin(0.15 * 0.02 * t)
    else:
        quat[:, 0] = 1.0
        if axis is not None:
            quat[:, 0], quat[:, 1 + axis] = math.cos(math.pi / 4), math.sin(math.pi / 4)
    js = j + 0.05 * torch.sin(t).unsqueeze(-1)
    return p, quat, js


def library(device: str = "cpu"):
    """(Kinematics, MotionLib, MDP) of the clips above."""
    kin = Kinematics(ROBOT, device)
    j = torch.tensor(ROBOT.default_joint_pos_list())
    spec = {"walk": (0.76, None, True), "low": (0.40, None, False), "tilted": (0.76, 0, False), "lying": (0.15, 1, False)}
    clips = [_clip(CLIPS[k], z, axis, j, walk) for k, (z, axis, walk) in spec.items()]
    motion = MotionLib.from_clips(list(spec), *zip(*clips), kin, device, CONTROL_DT)
    mdp = MDP(kin, motion, j.to(device), XiLayout(ROBOT, device).dim, ROBOT.sensing)
    return kin, motion, mdp
