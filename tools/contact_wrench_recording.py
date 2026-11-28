"""Record controls of the training simulator's model with per-link contact wrenches, stepped by MuJoCo C (CPU only):
the data of the rigid-body step comparison (tools/rigid_body_step_comparison.py).

The physics is the training simulator's model (``rigid_body.trainer_model``: mjlab's compiled scene with ``make_sim``'s
options: implicitfast, pyramidal cone, Newton with the robot's iterations), stepped by MuJoCo C in float64 with the
solver tolerance raised to MuJoCo Warp's float32 floor (1e-6). Each trial runs a checkpoint's averaged policy
closed-loop under the training conditions of ``TrackingSim`` with
randomization: per trial a friction drawn from ``dr.friction_range`` on every robot geom, a torso centre-of-mass offset
from ``com_range`` and joint encoder offsets from ``dr.joint_offset`` (the actuator target moves by the offset, the
observation reads q - offset), the reset noise of ``placed_state``, observation noise (``MDP.add_noise``) and action
noise 0.1 N(0, 1) clamped to +-20 (``networks.collect_actions``). The command starts at the trial's row and advances one
row per control (held at the clip end); only a bad state (a MuJoCo warning) ends a trial. The policy runs on the CPU
in float32 (the training ran its MLPs in bf16 autocast on the card).

Per control (with --store) it stores: qpos and qvel at the control boundary and after each of the 4 physics substeps
(float64), the action, the action of the preceding control (``a_prev``) and the joint targets, xi in ``XiLayout`` order,
and per physics substep, from the data of the state the physics substep starts from (mj_forward before mj_step): the
floor's and the robot-robot contacts' wrenches on every robot body (their sum is the body's net contact wrench; force in
the world frame and moment about the body origin, from mj_contactForce with the contact-frame convention of
mj_rnePostConstraint: +f on geom2's body, -f on geom1's), the generalized force of the robot-robot contacts, that of the
joint-limit solver constraints and MuJoCo's qfrc_constraint. The stored key ``identity_max`` is the largest
|qfrc_constraint - J^T W_floor - tau_self - tau_limit| of the control: the split of the constraint force into these
three parts is exact when it is at rounding level.

Without --store only the per-body floor-contact statistics are written (contact physics substeps, summed force).

  python tools/contact_wrench_recording.py --run_dir RUN --ckpt 7000 --motion_dir DIR --clips fallAndGetUp1_subject1 ... \
      --stride 100 --controls 150 --seed 0 --store --out recording.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import mujoco
import numpy as np
import torch

from blinkmimic2.dr import XiLayout
from blinkmimic2.env import DECIMATION, placed_state
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import MDP
from blinkmimic2.motion import MotionLib
from blinkmimic2.networks import Policy, policy_kwargs
from blinkmimic2.rigid_body import trainer_model
from blinkmimic2.robot import get_robot
from blinkmimic2.rotations import mat_to_quat

WARP_TOLERANCE_FLOOR = 1e-6
BAD = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC,
       mujoco.mjtWarning.mjWARN_BADCTRL)  # fmt: skip


class Bodies:
    """Robot bodies of the model (the moving bodies, in model order) and the floor geoms."""

    def __init__(self, m, prefix="robot/"):
        self.ids = [b for b in range(m.nbody) if m.body_jntnum[b] > 0]
        self.local = {b: i for i, b in enumerate(self.ids)}
        self.names = [m.body(b).name.removeprefix(prefix) for b in self.ids]
        root = self.ids[0]
        robot = set()
        for b in range(m.nbody):
            k = b
            while k != 0 and k != root:
                k = m.body_parentid[k]
            if k == root:
                robot.add(b)
        self.floor_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] not in robot}
        self.act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])


def substep_loads(m, d, B: Bodies, f6, jp, jr):
    """After mj_forward: floor and robot-robot contact wrenches per robot body (nb, 6) each, tau_self (nv), tau_limit
    (nv), qfrc_constraint (nv)."""
    nb = len(B.ids)
    Wf = np.zeros((nb, 6))
    Ws = np.zeros((nb, 6))
    for i in range(d.ncon):
        c = d.contact[i]
        if c.efc_address < 0:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fr = c.frame.reshape(3, 3)
        F, Tq = fr.T @ f6[:3], fr.T @ f6[3:]
        g = (c.geom[0], c.geom[1])
        floor = g[0] in B.floor_geoms or g[1] in B.floor_geoms
        for k, s in ((0, -1.0), (1, 1.0)):
            b = m.geom_bodyid[g[k]]
            if b not in B.local:
                continue
            j = B.local[b]
            W = Wf if floor else Ws
            W[j, :3] += s * F
            W[j, 3:] += s * (np.cross(c.pos - d.xpos[b], F) + Tq)
    tf = np.zeros(m.nv)
    ts = np.zeros(m.nv)
    for j, b in enumerate(B.ids):
        if not (Wf[j].any() or Ws[j].any()):
            continue
        mujoco.mj_jacBody(m, d, jp, jr, b)
        tf += jp.T @ Wf[j, :3] + jr.T @ Wf[j, 3:]
        ts += jp.T @ Ws[j, :3] + jr.T @ Ws[j, 3:]
    tl = np.zeros(m.nv)
    nefc = d.nefc
    if nefc:
        if mujoco.mj_isSparse(m):
            raise NotImplementedError("sparse constraint Jacobian")
        J = d.efc_J[: nefc * m.nv].reshape(nefc, m.nv)
        lim = d.efc_type[:nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
        tl = J[lim].T @ d.efc_force[:nefc][lim]
        contact = np.isin(d.efc_type[:nefc], (mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
                                              mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS))
        other = ~lim & ~contact
        if other.any():
            raise RuntimeError(f"unexpected constraint types {set(d.efc_type[:nefc][other].tolist())}")
    return Wf, Ws, tf, ts, tl, d.qfrc_constraint.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", type=int, required=True)
    ap.add_argument("--motion_dir", required=True)
    ap.add_argument("--clips", nargs="+", required=True, help="clip names (file stems); 'all' for every clip")
    ap.add_argument("--stride", type=int, required=True, help="rows between trial starts in a clip")
    ap.add_argument("--controls", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        ap.error(f"{a.out} exists")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    run = json.load(open(os.path.join(a.run_dir, "args.json")))
    ck = torch.load(os.path.join(a.run_dir, f"ckpt_{a.ckpt:06d}.pt"), map_location="cpu", weights_only=False)
    robot = get_robot(run["robot"])
    dev = "cpu"
    kin = Kinematics(robot, dev)
    motion = MotionLib.load(robot, kin, dev, a.motion_dir, "", None, run["motion_store"])
    layout = XiLayout(robot, dev)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list()), layout.dim, robot.sensing)
    ar = ck["args"]
    widths = lambda k: tuple(int(x) for x in ar[k].split(","))  # noqa: E731
    K = ar["goal_frames"]
    pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=ar["hist"], enc_hidden=widths("enc_hidden"),
                dec_hidden=widths("dec_hidden"), **policy_kwargs(ar, robot))  # fmt: skip
    pi.load_state_dict(ck["pi"])
    pi.eval()

    names = list(motion.names)
    clips = names if a.clips == ["all"] else a.clips
    starts, clip_of = [], []
    for cn in clips:
        c = names.index(cn)
        s0, e = int(motion.clip_start[c]), int(motion.clip_end[c])
        for f in range(s0, e - 1, a.stride):
            starts.append(f), clip_of.append(c)
    T = len(starts)
    m0 = trainer_model(robot)
    tol_compiled = float(m0.opt.tolerance)
    m0.opt.tolerance = max(tol_compiled, WARP_TOLERANCE_FLOOR)
    B = Bodies(m0)
    nb, nv, J = len(B.ids), m0.nv, robot.num_joints
    robot_geoms = [g for g in range(m0.ngeom) if g not in B.floor_geoms]
    com_body = m0.body("robot/" + robot.com_body).id
    c = robot.dr
    friction = rng.uniform(*c.friction_range, size=T)
    com = np.stack([rng.uniform(lo, hi, size=T) for lo, hi in robot.com_range], -1)
    half = np.array([robot.dr.joint_offset.get(".*", 0.0)] * J) if list(c.joint_offset) == [".*"] else None
    if half is None:
        raise NotImplementedError("joint_offset patterns other than '.*'")
    joff = rng.uniform(-1, 1, size=(T, J)) * half
    xi = torch.zeros(T, layout.dim)
    xi[:, layout.slices["friction"]] = torch.tensor(friction, dtype=torch.float32).unsqueeze(-1)
    xi[:, layout.slices["joint_offset"]] = torch.tensor(joff, dtype=torch.float32)
    xi[:, layout.slices["com"]] = torch.tensor(com, dtype=torch.float32)
    models, datas = [], []
    for t in range(T):
        m = copy.deepcopy(m0)
        m.geom_friction[robot_geoms, 0] = friction[t]
        m.body_ipos[com_body] = m0.body_ipos[com_body] + com[t]
        models.append(m), datas.append(mujoco.MjData(m))
    # placement: reference state of the start rows with the training reset noise, joints clamped
    f = torch.tensor(starts)
    ref = tuple(x.clone().float() for x in motion.ref_state(f))
    p, R, v, w, q, qd = placed_state(ref, robot.dr, True)
    q = torch.clamp(q, kin.q_lo, kin.q_hi)
    quat = mat_to_quat(R)
    wl = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
    default_q = torch.tensor(robot.default_joint_pos_list())
    alpha = torch.tensor(robot.action_scale_list())
    for t in range(T):
        d = datas[t]
        mujoco.mj_resetData(models[t], d)
        d.qpos[:] = np.concatenate((p[t].numpy(), quat[t].numpy(), q[t].numpy())).astype(np.float64)
        d.qvel[:] = np.concatenate((v[t].numpy(), wl[t].numpy(), qd[t].numpy())).astype(np.float64)
        d.ctrl[:] = d.qpos[7:][B.act_dof - 6]
        mujoco.mj_forward(models[t], d)
    alive = np.ones(T, dtype=bool)
    a_prev = torch.zeros(T, J)
    hist = None
    f6, jp, jr = np.zeros(6), np.zeros((3, nv)), np.zeros((3, nv))
    rec = {k: [] for k in ("trial", "row", "control", "qpos", "qvel", "action", "a_prev", "target", "xi", "wrench_floor",
                           "wrench_self", "tau_floor", "tau_self", "tau_limit", "qfrc_constraint", "identity_max", "ncon")}  # fmt: skip
    stat_n = np.zeros((len(names), nb))  # floor-contact physics substeps per clip and body
    stat_f = np.zeros((len(names), nb))  # summed floor normal force (world z) per clip and body, N x physics substep
    stat_fn = np.zeros((len(names), nb))  # summed |force|
    stat_sub = np.zeros(len(names))  # substeps per clip

    def state_tensor():
        qp = np.stack([d.qpos for d in datas])
        qv = np.stack([d.qvel for d in datas])
        qp_t, qv_t = torch.tensor(qp, dtype=torch.float32), torch.tensor(qv, dtype=torch.float32)
        from blinkmimic2.rotations import quat_to_mat

        Rm = quat_to_mat(qp_t[:, 3:7])
        return (qp_t[:, :3], Rm, qv_t[:, :3], (Rm @ qv_t[:, 3:6].unsqueeze(-1)).squeeze(-1), qp_t[:, 7:], qv_t[:, 6:]), qp, qv

    row = f.clone()
    t0 = time.time()
    for ctl in range(a.controls):
        S, qp0, qv0 = state_tensor()
        with torch.no_grad():
            o = mdp.add_noise(mdp.obs(S, a_prev, xi, row))
            hist = o.unsqueeze(1).repeat(1, pi.hist, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            act = pi(hist, motion.goal(row, K))
            act = torch.clamp(act + 0.1 * torch.randn_like(act), -20, 20)
        target = (default_q + xi[:, layout.slices["joint_offset"]] + alpha * act).double().numpy()
        for t in range(T):
            if not alive[t]:
                continue
            m, d = models[t], datas[t]
            d.ctrl[:] = target[t][B.act_dof - 6]
            warn0 = sum(d.warning[w].number for w in BAD)
            QP, QV = [d.qpos.copy()], [d.qvel.copy()]
            WF, WS, TF, TS, TL, QC, ncon = [], [], [], [], [], [], []
            ident = 0.0
            for _ in range(DECIMATION):
                mujoco.mj_forward(m, d)
                Wf, Ws, tf, ts, tl, qc = substep_loads(m, d, B, f6, jp, jr)
                ident = max(ident, float(np.abs(qc - tf - ts - tl).max()))
                fz = Wf[:, 2]
                on = np.abs(Wf[:, :3]).sum(-1) > 0
                stat_n[clip_of[t]] += on
                stat_f[clip_of[t]] += fz
                stat_fn[clip_of[t]] += np.linalg.norm(Wf[:, :3], axis=-1)
                stat_sub[clip_of[t]] += 1
                WF.append(Wf), WS.append(Ws), TF.append(tf), TS.append(ts), TL.append(tl), QC.append(qc), ncon.append(d.ncon)
                mujoco.mj_step(m, d)
                QP.append(d.qpos.copy()), QV.append(d.qvel.copy())
            bad = sum(d.warning[w].number for w in BAD) > warn0 or not np.isfinite(d.qpos).all()
            if bad:
                alive[t] = False
                continue
            if a.store:
                for k, val in (("trial", t), ("row", int(row[t])), ("control", ctl), ("qpos", np.stack(QP)), ("qvel", np.stack(QV)),
                               ("action", act[t].numpy()), ("a_prev", a_prev[t].numpy()), ("target", target[t]),
                               ("xi", xi[t].numpy()), ("wrench_floor", np.stack(WF).astype(np.float32)),
                               ("wrench_self", np.stack(WS).astype(np.float32)),
                               ("tau_floor", np.stack(TF).astype(np.float32)), ("tau_self", np.stack(TS).astype(np.float32)),
                               ("tau_limit", np.stack(TL).astype(np.float32)), ("qfrc_constraint", np.stack(QC)),
                               ("identity_max", ident), ("ncon", np.array(ncon))):  # fmt: skip
                    rec[k].append(val)
        a_prev = act
        last = motion.clip_end[motion.clip_id[row]]
        row = torch.minimum(row + 1, last)
        if ctl % 25 == 0:
            print(f"control {ctl}: {alive.sum()} / {T} trials running, {time.time() - t0:.0f} s", flush=True)
    out = {"body_names": np.array(B.names), "clip_names": np.array(names), "stat_contact_substeps": stat_n,
           "stat_force_z": stat_f, "stat_force_norm": stat_fn, "stat_substeps": stat_sub,
           "trial_start_row": np.array(starts), "trial_clip": np.array(clip_of), "trial_running_at_end": alive,
           "friction": friction, "com_offset": com, "joint_offset": joff}  # fmt: skip
    if a.store:
        out |= {k: np.array(v) for k, v in rec.items()}
    meta = {"run_dir": a.run_dir, "ckpt": a.ckpt, "iteration": ck.get("iteration"), "clips": clips, "stride": a.stride,
            "controls": a.controls, "seed": a.seed, "trials": T, "mujoco": mujoco.__version__,
            "tolerance": {"compiled": tol_compiled, "stepped": max(tol_compiled, WARP_TOLERANCE_FLOOR)},
            "options": {"integrator": int(m0.opt.integrator), "cone": int(m0.opt.cone), "solver": int(m0.opt.solver),
                        "iterations": int(m0.opt.iterations), "ls_iterations": int(m0.opt.ls_iterations),
                        "timestep": float(m0.opt.timestep)},
            "xi_slices": {k: [s.start, s.stop] for k, s in layout.slices.items()}, "wall_s": time.time() - t0}  # fmt: skip
    np.savez_compressed(a.out, **out)
    json.dump(meta, open(a.out.removesuffix(".npz") + ".json", "w"), indent=1)
    if a.store and rec["identity_max"]:
        print("identity_max over controls:", float(np.max(rec["identity_max"])))
    print(f"wrote {a.out}: {len(rec['trial'])} controls stored, {T} trials, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
