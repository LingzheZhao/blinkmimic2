"""Comparison of blinkmimic2.rigid_body with MuJoCo C on the trainer's model (CPU).

For random states (base pose and velocities, joints over their ranges, joint speeds, torso centre-of-mass offsets in
the trainer's range, joint targets far enough to saturate actuators) and for library states (rows of the LAFAN1
clips, every get-up clip's lying rows among them, with random velocities) it compares, per state:

  M        mj_fullM (armature included)                qfrc_bias    gravity + Coriolis / centrifugal
  jacp/jacr  mj_jacBody of every robot body            qfrc_actuator  the position actuators with forcerange
  qfrc_passive                                          physics_substep  one implicitfast mj_step from the same state,
                                                                       fed MuJoCo's own qfrc_constraint

and the Jacobian-free (tree) path's mass matrix, bias force minus link wrenches and physics substep. It writes the
largest absolute and relative difference of each (relative: per state, the largest absolute entry difference over the
largest absolute entry of MuJoCo's value), in float64 and float32.

  python tools/rigid_body_comparison.py --out comparison.json [--motion_dir DIR] [--n_random 200] [--n_library 400]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mujoco
import numpy as np
import torch

from blinkmimic2.rigid_body import RigidBody, from_mujoco, trainer_model
from blinkmimic2.robot import get_robot

GETUP = "fallAndGetUp"


def random_states(m, rng, n, com_range):
    qpos = np.zeros((n, m.nq))
    qvel = rng.normal(size=(n, m.nv))
    qpos[:, :3] = rng.uniform([-1, -1, 0.1], [1, 1, 1.2], size=(n, 3))
    quat = rng.normal(size=(n, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    qpos[:, 7:] = rng.uniform(lo, hi, size=(n, m.nu))
    qvel[:, :3] *= 1.0
    qvel[:, 3:6] *= 3.0
    qvel[:, 6:] *= 5.0
    com = np.stack([rng.uniform(a, b, size=n) for a, b in com_range], -1)
    return qpos, qvel, com


def library_states(m, rng, files, n, com_range, load_motion):
    """Rows of the clips (half of them from the get-up clips' rows with the pelvis below 0.5 m or tilted past 60 deg),
    with random velocities of the training reset's size."""
    rows = []
    for f in files:
        p, quat, j, _ = load_motion(f)
        low = (p[:, 2] < 0.5) | (1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2) < 0.5)
        rows.append((os.path.basename(f), p, quat, j, low))
    getup = [r for r in rows if r[0].startswith(GETUP)]
    out = []
    for k in range(n):
        if k % 2 == 0:
            name, p, quat, j, low = getup[rng.integers(len(getup))]
            idx = np.flatnonzero(low)
        else:
            name, p, quat, j, low = rows[rng.integers(len(rows))]
            idx = np.arange(len(p))
        i = rng.choice(idx)
        out.append(np.concatenate((p[i], quat[i], j[i])))
    qpos = np.array(out)
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    qvel = rng.normal(size=(n, m.nv)) * np.r_[[0.5] * 3, [0.6] * 3, [2.0] * (m.nv - 6)]
    com = np.stack([rng.uniform(a, b, size=n) for a, b in com_range], -1)
    return qpos, qvel, com


def mujoco_reference(m, qpos, qvel, ctrl, com, com_body):
    """MuJoCo's quantities per state; the torso ipos is offset for that state and restored."""
    d = mujoco.MjData(m)
    ipos0 = m.body_ipos[com_body].copy()
    nb = [b for b in range(m.nbody) if m.body_jntnum[b] > 0]
    out = {k: [] for k in ("M", "bias", "jacp", "jacr", "act", "passive", "constraint", "qpos1", "qvel1", "ncon", "nefc")}
    for s in range(len(qpos)):
        m.body_ipos[com_body] = ipos0 + com[s]
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[s], qvel[s], ctrl[s]
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        jp, jr = [], []
        for b in nb:
            a, r = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, a, r, b)
            jp.append(a.T), jr.append(r.T)
        out["M"].append(M), out["bias"].append(d.qfrc_bias.copy()), out["jacp"].append(np.stack(jp)), out["jacr"].append(np.stack(jr))
        out["act"].append(d.qfrc_actuator.copy()), out["passive"].append(d.qfrc_passive.copy())
        out["constraint"].append(d.qfrc_constraint.copy()), out["ncon"].append(d.ncon), out["nefc"].append(d.nefc)
        mujoco.mj_step(m, d)
        out["qpos1"].append(d.qpos.copy()), out["qvel1"].append(d.qvel.copy())
    m.body_ipos[com_body] = ipos0
    return {k: np.array(v) for k, v in out.items()}


def ctrl_for(m, rng, qpos, act_joint):
    """Joint targets around the state's joints, wide enough that some actuators hit their force range."""
    n = len(qpos)
    tgt = qpos[:, 7:] + rng.normal(size=(n, m.nu)) * 0.6
    return tgt[:, act_joint]  # ctrl in actuator order


def compare(rb: RigidBody, m, qpos, qvel, ctrl_j, com, ref, dtype):
    T = lambda x: torch.tensor(x, dtype=dtype)  # noqa: E731
    rb2 = RigidBody(m, rb.joint_names, dtype=dtype)
    p, R, q, nu = from_mujoco(T(qpos), T(qvel))
    co = T(com)
    kin = rb2.kinematics(p, R, q, co)
    Jp, Jr = rb2.jacobians(kin)
    M = rb2.mass_matrix(kin, Jp, Jr)
    h = rb2.bias(kin, Jp, Jr, nu)
    fa, _ = rb2.actuator(q, nu[:, 6:], T(ctrl_j))
    act = torch.cat((torch.zeros_like(nu[:, :6]), fa), -1)
    passive = torch.cat((torch.zeros_like(nu[:, :6]), -rb2.damping[6:] * nu[:, 6:]), -1)
    p1, R1, q1, nu1 = rb2.substep(p, R, q, nu, T(ctrl_j), m.opt.timestep, tau_ext=T(ref["constraint"]), com_offset=co)
    P1, Rm1, Q1, NU1 = from_mujoco(T(ref["qpos1"]), T(ref["qvel1"]))
    rot_err = torch.linalg.matrix_norm(R1 - Rm1, ord=2)
    # the Jacobian-free (tree) path: mass matrix, bias minus link wrenches, and its steps with both solves
    kin_t = rb2.kinematics(p, R, q, co)
    M_t = rb2.mass_matrix_tree(kin_t)
    links = list(range(rb2.nb))
    g = torch.Generator().manual_seed(0)
    Wr = (torch.randn(len(p), rb2.nb, 6, generator=g, dtype=torch.float64) * 20).to(dtype)
    hb_t = rb2.bias_tree(kin_t, nu, links, Wr)
    hb_d = h - rb2.wrench_force(Jp, Jr, links, Wr)
    tree = {}
    for solve in ("cholesky", "ltdl", "columns"):
        out = rb2.substep_tree(p, R, q, nu, T(ctrl_j), m.opt.timestep, tau_ext=T(ref["constraint"]), com_offset=co, solve=solve)
        tree[solve] = out
    # sensitivity: the same physics substep with kd kept in the implicit term for force-clamped actuators (what MuJoCo does not do)
    act_fn = rb2.actuator
    rb2.actuator = lambda q_, qd_, t_: (act_fn(q_, qd_, t_)[0], rb2.kd.expand(q_.shape))
    nu1_kept = rb2.substep(p, R, q, nu, T(ctrl_j), m.opt.timestep, tau_ext=T(ref["constraint"]), com_offset=co)[3]
    rb2.actuator = act_fn

    def err(mine, theirs):
        mine, theirs = mine.double().numpy(), np.asarray(theirs)
        ax = tuple(range(1, theirs.ndim))
        a = np.abs(mine - theirs).max(axis=ax)
        s = np.abs(theirs).max(axis=ax)
        rel = a / np.maximum(s, 1e-12)
        return {"max_abs": float(a.max()), "max_rel": float(rel.max()), "median_rel": float(np.median(rel)),
                "scale_median": float(np.median(s))}  # fmt: skip

    return {
        "M": err(M, ref["M"]), "qfrc_bias": err(h, ref["bias"]), "jacp": err(Jp, ref["jacp"]), "jacr": err(Jr, ref["jacr"]),
        "qfrc_actuator": err(act, ref["act"]), "qfrc_passive": err(passive, ref["passive"]),
        "physics_substep_qvel": err(nu1, ref["qvel1"]), "physics_substep_pos": err(p1, ref["qpos1"][:, :3]), "physics_substep_joint_pos": err(q1, ref["qpos1"][:, 7:]),
        "physics_substep_rot_max_abs": float(rot_err.max()), "physics_substep_qvel_if_clamped_kd_kept": err(nu1_kept, ref["qvel1"]),
        "tree_M": err(M_t, ref["M"]), "tree_bias_minus_wrench_vs_dense": err(hb_t, hb_d.double().numpy()),
        "tree_cholesky_physics_substep_qvel": err(tree["cholesky"][3], ref["qvel1"]), "tree_ltdl_physics_substep_qvel": err(tree["ltdl"][3], ref["qvel1"]), "tree_columns_physics_substep_qvel": err(tree["columns"][3], ref["qvel1"]),
    }  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--motion_dir", default=None)
    ap.add_argument("--n_random", type=int, default=200)
    ap.add_argument("--n_library", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    robot = get_robot("mjlab_g1")
    m = trainer_model(robot)
    rb = RigidBody(m, robot.joint_names)
    rng = np.random.default_rng(a.seed)
    com_body = m.body("robot/" + robot.com_body).id
    act_joint = np.array([m.jnt_dofadr[m.actuator_trnid[k, 0]] - 6 for k in range(m.nu)])
    sets = {"random": random_states(m, rng, a.n_random, robot.com_range)}
    if a.motion_dir:
        files = sorted(glob.glob(os.path.join(a.motion_dir, robot.motion_glob)))
        sets["library"] = library_states(m, rng, files, a.n_library, robot.com_range, robot.load_motion)
    report = {"model": {"nv": m.nv, "nbody": rb.nb, "integrator": int(m.opt.integrator), "timestep": m.opt.timestep,
                        "mujoco": mujoco.__version__, "total_mass": rb.total_mass}}  # fmt: skip
    for name, (qpos, qvel, com) in sets.items():
        ctrl = ctrl_for(m, rng, qpos, act_joint)
        ref = mujoco_reference(m, qpos, qvel, ctrl, com, com_body)
        ctrl_j = np.zeros((len(qpos), m.nu))
        ctrl_j[:, act_joint] = ctrl
        sat = np.abs(ref["act"][:, 6:]) >= (np.abs(rb.f_hi.numpy()) - 1e-9)
        info = {"states": len(qpos), "states_with_contact": int((ref["ncon"] > 0).sum()),
                "states_with_constraint_rows": int((ref["nefc"] > 0).sum()), "saturated_actuator_share": float(sat.mean()),
                "pelvis_below_0.5": int((qpos[:, 2] < 0.5).sum())}  # fmt: skip
        report[name] = {"info": info, "float64": compare(rb, m, qpos, qvel, ctrl_j, com, ref, torch.float64),
                        "float32": compare(rb, m, qpos, qvel, ctrl_j, com, ref, torch.float32)}  # fmt: skip
        print(name, json.dumps(report[name], indent=1), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
