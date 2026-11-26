"""Differentiable forward kinematics of the robot in torch, built from its MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import RobotSpec
from .rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat


class Kinematics:
    def __init__(self, robot: RobotSpec, device: str):
        m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.device = device
        tracked = robot.tracked_bodies
        jname_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
        # moving bodies in tree order (mujoco orders parents before children)
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] == 0:
                continue  # fixed links are merged visually only; none are tracked
            jid = m.body_jntadr[b]
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                while pb not in mj_to_local:  # skip fixed parents
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in joint_names order)
        lo, hi = np.zeros(robot.num_joints), np.zeros(robot.num_joints)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,J) in joint_names order.
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + mv3(Rp, self.off_p[i])
            R_off = mm3(Rp, self.off_R[i])
            Ri = mm3(R_off, axis_angle_to_mat(self.axes[i], q[..., j]))
            axis_w = mv3(R_off, self.axes[i])
            wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )


def foot_capsules(robot: RobotSpec, bodies: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Difference from the release code (train.py --sole_height_reward; the release has no such function): the end
    points of the collision capsules of each body in ``bodies``, in that body's frame, from the compiled MJCF, with the
    radius of the capsule each end point belongs to: ``(ends (len(bodies), E, 3), radius (len(bodies), E))``, E = twice
    the largest number of capsules of one body (a body with fewer repeats its last end point). The lowest point of a
    capsule lies one radius below its lower end point, so a body's lowest capsule point is the least (end point height
    - its radius) (``mdp.sole_height``). A body without collision capsules is refused."""
    m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    all_ends, all_radii = [], []
    for name in bodies:
        b = m.body(name).id
        ends, radii = [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue  # a capsule that collides with nothing is not collision geometry
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[g])
            half = rot.reshape(3, 3)[:, 2] * m.geom_size[g, 1]  # the capsule's axis is its frame's z, half-length size[1]
            for sign in (1.0, -1.0):
                ends.append(m.geom_pos[g] + sign * half)
                radii.append(float(m.geom_size[g, 0]))
        if not ends:
            raise ValueError(f"{robot.mjcf_path}: body {name} has no collision capsules")
        all_ends.append(ends), all_radii.append(radii)
    E = max(len(e) for e in all_ends)
    all_ends = [e + [e[-1]] * (E - len(e)) for e in all_ends]
    all_radii = [r + [r[-1]] * (E - len(r)) for r in all_radii]
    return (torch.tensor(np.array(all_ends), dtype=torch.float32, device=device),
            torch.tensor(np.array(all_radii), dtype=torch.float32, device=device))  # fmt: skip
