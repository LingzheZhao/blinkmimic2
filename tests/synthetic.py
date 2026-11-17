"""Small motion libraries and MDPs of the mjlab G1 on the CPU, built from generated clips at 50 Hz.

``library``, clips (rows): walk (0-5) upright at 0.76 m walking along x while turning; low (6-9) upright at 0.40 m;
tilted (10-13) at 0.76 m rolled 90 degrees about x; lying (14-17) at 0.15 m pitched 90 degrees about y.
``getup_library``: the clips of GETUP.
"""

import math

import torch

from blinkmimic2.dr import XiLayout
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import MDP
from blinkmimic2.motion import MotionLib
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
    motion = MotionLib.from_clips(list(spec), *zip(*clips), kin, device)
    mdp = MDP(kin, motion, j.to(device), XiLayout(ROBOT, device).dim, ROBOT.sensing)
    return kin, motion, mdp


C45 = math.cos(math.pi / 4)
UP, PITCHED, ROLLED = (1.0, 0.0, 0.0, 0.0), (C45, 0.0, C45, 0.0), (C45, C45, 0.0, 0.0)
# Clips of the recovery tests as runs of (rows, pelvis height, root quaternion wxyz). Pitched 90 degrees about y or
# rolled 90 degrees about x, the pelvis up axis is horizontal (world z 0). Rows: getup_a 0-9, getup_b 10-16, excluded
# 17-29, walk 30-41 (along x at 0.5 m/s).
GETUP = {"getup_a": [(3, 0.15, PITCHED), (1, 0.76, ROLLED), (6, 0.76, UP)],
         "getup_b": [(2, 0.76, UP), (3, 0.15, PITCHED), (2, 0.76, UP)],
         "excluded": [(3, 0.15, PITCHED), (10, 0.50, UP)],
         "walk": [(12, 0.76, UP)]}


def getup_library(cls=MotionLib, device: str = "cpu"):
    """(Kinematics, motion library of class ``cls``, MDP) of the GETUP clips."""
    kin = Kinematics(ROBOT, device)
    j = torch.tensor(ROBOT.default_joint_pos_list())
    ps, qs, js = [], [], []
    for name, runs in GETUP.items():
        z = torch.cat([torch.full((n,), h) for n, h, _ in runs])
        t = torch.arange(len(z), dtype=torch.float32)
        x = 0.5 * 0.02 * t if name == "walk" else torch.zeros(len(z))
        ps.append(torch.stack((x, torch.zeros(len(z)), z), -1))
        qs.append(torch.cat([torch.tensor(q).expand(n, 4) for n, _, q in runs]))
        js.append(j + 0.05 * torch.sin(t).unsqueeze(-1))
    motion = cls.from_clips(list(GETUP), ps, qs, js, kin, device)
    return kin, motion, MDP(kin, motion, j.to(device), XiLayout(ROBOT, device).dim, ROBOT.sensing)
