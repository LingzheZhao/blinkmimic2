"""Roll the policy of one checkpoint out from placed starts as tools/rollouts.py does, with the physics stepped by
MuJoCo C (the mujoco package) instead of MuJoCo Warp (ROLLOUTS.md, "Rollouts on MuJoCo C").

    python tools/mujoco_rollouts.py --run_dir logs/run_s1 --ckpt 6500 --scene scene.json --termination nonfinite \\
        --controls 1000 --threads 16 --out rollouts/run_s1_6500_mujoco
    python tools/mujoco_rollouts.py ... --threads 16 --checks rollouts/run_s1_6500 --out checks/run_s1_6500

The loop, the scene, the policy, its precision, the observation, the action map and the record are those of
tools/rollouts.py. The evaluation-mode TrackingSim is built as that tool builds it; its compiled MjModel, with the
entries mjlab wrote into MuJoCo Warp's model after compilation and the solver tolerance MuJoCo Warp steps with, is the
model MuJoCo C steps, one MjData per trial for the whole rollout. With --checks RECORD the program compares the two engines from the same states (a record of
tools/rollouts.py on the same scene gives the start states) and writes checks.json instead of rollouts.
"""

import argparse
import concurrent.futures
import copy
import dataclasses
import importlib.metadata
import json
import os
import time

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

import blinkmimic2.networks as networks
import blinkmimic2.rotations as rotations
import rollouts
from blinkmimic2.dr import XiLayout
from blinkmimic2.env import DECIMATION, PHYSICS_DT, TrackingSim, make_sim, placed_state
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import MDP, flatten_state
from blinkmimic2.motion import CONTROL_DT, MotionLib
from blinkmimic2.networks import Policy, policy_kwargs
from blinkmimic2.robot import get_robot
from blinkmimic2.rotations import mat_to_quat, quat_to_mat

PREFIX = "robot/"  # the entity name of make_scene_cfg's robot, which mjlab puts before its element names
W = mujoco.mjtWarning
# MuJoCo C's warnings of a bad position, velocity or acceleration (after which it resets the data to the model's
# initial state: mj_checkPos, mj_checkVel, mj_checkAcc) and of a bad ctrl; the state after a control in which one of
# them rose is recorded as not finite
BAD_STATE = (W.mjWARN_BADQPOS, W.mjWARN_BADQVEL, W.mjWARN_BADQACC, W.mjWARN_BADCTRL)
FULL = (W.mjWARN_CONTACTFULL, W.mjWARN_CNSTRFULL)  # contacts, constraint rows dropped for want of arena memory
STATE_GROUPS = ("root_position", "root_quaternion", "root_linear_velocity", "root_angular_velocity",
                "joint_positions", "joint_velocities")  # fmt: skip
LIFT_M = 2.0  # height added to the placed roots for the step without floor contact


def trainer_model(mj_model, warp_fields: dict):
    """A copy of ``mj_model`` (the MjModel the training simulator was built from) holding the entries that were
    written after compilation into the fields ``warp_fields`` {name: [worlds, ...] tensor of MuJoCo Warp's model}:
    an entry whose value differs from the float32 rounding of the compiled value takes the written value; the others
    keep the compiled float64 value. A field whose worlds differ, or whose shape or name is not the model's, raises.
    Returns the model and {name: {"entries": count, "values": the distinct written values}}."""
    model, written = copy.deepcopy(mj_model), {}
    for name, field in sorted(warp_fields.items()):
        if not hasattr(mj_model, name):
            raise ValueError(f"the training simulator's model field {name} is not a field of the MjModel")
        base, worlds = getattr(mj_model, name), field.detach().cpu().numpy()
        if worlds.ndim != base.ndim + 1 or worlds.shape[1:] != base.shape or len(worlds) == 0:
            raise ValueError(f"{name}: shape {worlds.shape} of the training simulator, {base.shape} of the MjModel")
        if not (worlds == worlds[:1]).all():
            raise ValueError(f"{name} differs between the worlds of the training simulator; one MjModel cannot hold it")
        value = worlds[0]
        changed = value != base.astype(value.dtype)
        getattr(model, name)[changed] = value[changed]
        written[name] = {"entries": int(changed.sum()), "values": sorted({float(x) for x in value[changed]})}
    return model, written


# The options of MuJoCo Warp's model that may differ from the MjModel's it was built from, and why: put_model raises
# the main solver tolerance to at least 1e-6 for float32 (mujoco_warp io.py). Another difference raises.
WRITTEN_OPTIONS = {"tolerance": "MuJoCo Warp's put_model raises it to at least 1e-6, its float32 floor"}


def _option(model, name: str) -> np.ndarray:
    """The MjModel's option ``name`` as numbers: a number or vector as float64, an enumeration or flags as an int."""
    x = getattr(model.opt, name)
    return np.asarray(x, np.float64) if isinstance(x, (float, np.ndarray)) else np.asarray(int(x))


def _one_world(name: str, value, compiled: np.ndarray) -> np.ndarray:
    """The one value of the training simulator's option ``value``: a per-world option [worlds, ...] must hold one
    value over the worlds; an integer option is a single value."""
    w = np.asarray(value)
    if w.ndim == compiled.ndim and w.dtype.kind in "iub":
        return w
    if w.ndim != compiled.ndim + 1 or w.shape[1:] != compiled.shape or len(w) == 0:
        raise ValueError(f"option {name}: shape {w.shape} of the training simulator, {compiled.shape} of the MjModel")
    if not (w == w[:1]).all():
        raise ValueError(f"option {name} differs between the worlds of the training simulator; one MjModel cannot hold it")
    return w[0]


def option_comparison(model, warp_values: dict) -> dict:
    """The options of ``model`` beside those of the training simulator's model ``warp_values`` ({name: value}, from
    warp_option_values): {name: {"mujoco_c", "training_simulator", "equal"}} for each option both hold, a float
    compared at the float32 rounding of the MjModel's value; impratio against MuJoCo Warp's impratio_invsqrt
    (1 / sqrt(max(impratio, mjMINVAL))). Raises for a name that is not an option of the MjModel."""
    out = {}
    for name, value in sorted(warp_values.items()):
        if name == "impratio_invsqrt":
            c = np.asarray(1.0 / np.sqrt(max(model.opt.impratio, mujoco.mjMINVAL)))
        elif hasattr(model.opt, name):
            c = _option(model, name)
        else:
            raise ValueError(f"the training simulator's option {name} is not an option of the MjModel")
        w = _one_world(name, value, c)
        equal = bool((w == c.astype(w.dtype)).all())
        out[name] = {"mujoco_c": c.tolist(), "training_simulator": w.tolist(), "equal": equal}
    return out


def trainer_options(model, warp_values: dict) -> tuple[dict, dict]:
    """Write into ``model`` (the copy of trainer_model) the options of WRITTEN_OPTIONS whose value on the training
    simulator's model (``warp_values``, as option_comparison takes them) differs from the float32 rounding of the
    compiled value, as that value widened to float64. Returns the written options {name: {"compiled", "written",
    "reason"}} and the other options that differ {name: the row of option_comparison}, which are not written; the
    rollouts refuse a model with any (check_options)."""
    written, other = {}, {}
    for name, row in option_comparison(model, warp_values).items():
        if row["equal"]:
            continue
        if name not in WRITTEN_OPTIONS:
            other[name] = row
            continue
        setattr(model.opt, name, float(row["training_simulator"]))
        written[name] = {"compiled": row["mujoco_c"], "written": float(getattr(model.opt, name)),
                         "reason": WRITTEN_OPTIONS[name]}  # fmt: skip
    return written, other


def check_options(other: dict):
    """Raise where trainer_options left options that differ between the two engines' models."""
    if other:
        rows = {k: (v["training_simulator"], v["mujoco_c"]) for k, v in other.items()}
        raise ValueError(f"options (training simulator, MjModel) that differ and that this program does not reproduce: "
                         f"{rows}")  # fmt: skip


def joint_maps(model, joint_names, prefix: str = PREFIX) -> dict:
    """Addresses of the robot in ``model`` by name: the free joint's qpos (7) and dofs (6), each joint's qpos and dof
    address, and the actuator of each joint, in the order of ``joint_names``. Raises unless the model has one free
    joint, every joint is a one-dof joint named ``prefix`` + name, the free joint and the joints fill qpos and qvel,
    and each joint has exactly one actuator, a position actuator (fixed gain kp, affine bias -kp q - kd qdot, no
    dynamics, ctrl not limited), the actuator model whose ctrl mjlab sets to the position target."""
    J = mujoco.mjtJoint
    free = [j for j in range(model.njnt) if model.jnt_type[j] == J.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError(f"the model has {len(free)} free joints, not 1")
    ids = []
    for name in joint_names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        if j < 0:
            raise ValueError(f"the model has no joint {prefix + name}")
        if model.jnt_type[j] not in (J.mjJNT_HINGE, J.mjJNT_SLIDE):
            raise ValueError(f"{prefix + name} is not a one-dof joint")
        ids.append(j)
    fq, fv = int(model.jnt_qposadr[free[0]]), int(model.jnt_dofadr[free[0]])
    out = {"free_q": np.arange(fq, fq + 7), "free_v": np.arange(fv, fv + 6),
           "joint_q": model.jnt_qposadr[ids].astype(np.int64), "joint_v": model.jnt_dofadr[ids].astype(np.int64)}  # fmt: skip
    if sorted([*out["free_q"], *out["joint_q"]]) != list(range(model.nq)):
        raise ValueError("the free joint and the joints do not fill qpos")
    if sorted([*out["free_v"], *out["joint_v"]]) != list(range(model.nv)):
        raise ValueError("the free joint and the joints do not fill qvel")
    if model.nu != len(joint_names):
        raise ValueError(f"the model has {model.nu} actuators for {len(joint_names)} joints")
    actuator = [-1] * len(joint_names)
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        kp, bias = model.actuator_gainprm[a, 0], model.actuator_biasprm[a, :3]
        position = (model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE
                    and model.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
                    and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
                    and not model.actuator_ctrllimited[a] and kp > 0 and bias[0] == 0 and bias[1] == -kp and bias[2] <= 0)  # fmt: skip
        if not position:
            raise ValueError(f"actuator {name} is not an unlimited position actuator on a joint")
        joint = int(model.actuator_trnid[a, 0])
        if joint not in ids:
            raise ValueError(f"actuator {name} drives {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)}, "
                             "which is not one of the robot's joints")  # fmt: skip
        if actuator[ids.index(joint)] >= 0:
            raise ValueError(f"joint {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)} has two actuators")
        actuator[ids.index(joint)] = a
    out["actuator"] = np.array(actuator, dtype=np.int64)
    return out


class MujocoC:
    """The simulator object of rollouts.roll_out with the physics stepped by MuJoCo C: one MjData per trial, kept for
    the whole rollout, stepped DECIMATION times per control in a pool of ``threads`` threads over contiguous chunks of
    trials. The placement, the state tuple, the observation, the push and the target sent to the actuators are the
    float32 torch expressions of TrackingSim (reset, state, observe, step_core) and rollouts.Simulator on ``device``;
    float32 values enter the float64 MjData only through the placement, the push and ctrl, and the MjData are never
    written from the float32 state. The object owns the previous action (zero at a placement, the action after a
    step) and the context ``xi`` (TrackingSim's in evaluation: friction 1, the rest 0), which the observation and
    the target read. ``origins`` are the worlds' origins (Scene.env_origins). The report of a step holds MuJoCo C's
    ncon and nefc after the last substep, and whether its contact or constraint arena ran full during the control
    (the CONTACTFULL and CNSTRFULL warnings); ``substep_peaks`` holds, for the checks, the largest ncon, nefc and nl
    (joint-limit rows) of each trial over the substeps of the last control. A control in which MuJoCo C finds a bad position, velocity,
    acceleration or ctrl (and resets its data) gives a non-finite state until the trial is placed again."""

    def __init__(self, model, trials: int, joint_names, motion, mdp, dr, default_q, alpha, xi, origins, threads: int,
                 device: str):  # fmt: skip
        if threads < 1:
            raise ValueError(f"threads = {threads}, not at least 1")
        J = len(joint_names)
        shapes = {"default_q": (default_q, (J,)), "alpha": (alpha, (J,)), "xi": (xi, (trials, mdp.xi_dim)),
                  "origins": (origins, (trials, 3))}  # fmt: skip
        for key, (x, shape) in shapes.items():
            if tuple(x.shape) != shape:
                raise ValueError(f"{key} has shape {tuple(x.shape)}, not {shape}")
        self.model, self.T, self.J, self.motion, self.mdp, self.dr, self.device = model, trials, J, motion, mdp, dr, device
        self.maps = joint_maps(model, joint_names)
        self.index = {k: torch.as_tensor(v, device=device) for k, v in self.maps.items()}
        self.default_q, self.alpha = default_q.to(device).clone(), alpha.to(device).clone()
        self.xi, self.origins = xi.to(device).clone(), origins.to(device).clone()
        self.a_prev = torch.zeros(trials, J, device=device)
        self.data = [mujoco.MjData(model) for _ in range(trials)]
        self.bad = np.zeros(trials, dtype=bool)
        self.substep_peaks = np.zeros((trials, 3), dtype=np.int64)  # of the last control: largest ncon, nefc, nl
        self.chunks = [c for c in np.array_split(np.arange(trials), min(threads, trials)) if len(c)]
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.chunks))

    @classmethod
    def from_training(cls, env: TrackingSim, model, threads: int):
        """The object for the evaluation-mode TrackingSim ``env`` (its joint names, motion, MDP, reset half-widths,
        default pose, action scale, context and world origins) on ``model`` (trainer_model of ``env``). Raises where
        ``env`` is not the evaluation set-up this object reproduces, or where the addresses of joint_maps are not
        mjlab's."""
        from mjlab.actuator import BuiltinPositionActuator
        from mjlab.actuator.actuator import TransmissionType

        robot = env.spec
        refused = {"eval_mode is off": not env.eval_mode, "randomization is on": env.randomize,
                   "reset noise is on": env.reset_noise, "the recovery options are set": env.recovery is not None,
                   "the sensing is stateful": env.pipe.stateful, "an actuator delays its command": robot.has_action_delay,
                   "an actuator is not mjlab's built-in joint position actuator": not all(
                       isinstance(a, BuiltinPositionActuator) and a.transmission_type == TransmissionType.JOINT
                       for a in env.robot.actuators)}  # fmt: skip
        if any(refused.values()):
            raise ValueError(f"not the evaluation set-up of tools/rollouts.py: {[k for k, v in refused.items() if v]}")
        c = cls(model, env.N, robot.joint_names, env.motion, env.mdp, robot.dr, env.default_q, env.alpha, env.xi,
                env.scene.env_origins, threads, env.device)  # fmt: skip
        idx = env.robot.indexing
        mjlab = {"free_q": idx.free_joint_q_adr, "free_v": idx.free_joint_v_adr, "joint_q": idx.joint_q_adr,
                 "joint_v": idx.joint_v_adr}  # fmt: skip
        # mjlab's path from the position target to ctrl (BuiltinActuatorGroup.apply_controls): for a built-in position
        # actuator, ctrl[global_ctrl_ids] = joint_pos_target[:, target_ids], unclamped
        target_ctrl = [-1] * c.J
        for act in env.robot.actuators:
            for t, ctrl in zip(act.target_ids.tolist(), act.global_ctrl_ids.tolist()):
                target_ctrl[t] = ctrl
        mjlab["actuator"] = target_ctrl
        for key, value in mjlab.items():
            if [int(x) for x in torch.as_tensor(value).tolist()] != c.maps[key].tolist():
                raise ValueError(f"joint order: the {key} addresses of mjlab are not those found by name")
        return c

    # ---------------------------------------------------------------- the interface of rollouts.roll_out
    def place(self, ids, command_row, body_row, uniforms):
        """TrackingSim.reset in evaluation mode: the body row's reference state plus the uniforms' offsets, the joints
        clamped to their ranges, written after mj_resetData (which clears the warm start and ctrl, as mjlab's reset
        does), with no mj_forward; the previous action is zero."""
        ref = tuple(x.clone() for x in self.motion.ref_state(body_row))
        p, R, v, w, q, qd = placed_state(ref, self.dr, False, uniforms)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        pose = torch.cat((p + self.origins[ids], mat_to_quat(R)), -1)
        root_v = torch.cat((v, quat_apply_inverse(pose[:, 3:7], w)), -1)  # the free joint's angular velocity: body frame
        pose, q, root_v, qd = (x.double().cpu().numpy() for x in (pose, q, root_v, qd))
        m = self.maps
        for k, i in enumerate(ids.tolist()):
            d = self.data[i]
            mujoco.mj_resetData(self.model, d)
            d.qpos[m["free_q"]], d.qpos[m["joint_q"]] = pose[k], q[k]
            d.qvel[m["free_v"]], d.qvel[m["joint_v"]] = root_v[k], qd[k]
            self.bad[i] = False
        self.a_prev[ids] = 0.0

    def state(self):
        """TrackingSim.state of the trials' MjData, rounded to float32; NaN for a trial whose data MuJoCo C reset."""
        qpos = np.stack([d.qpos for d in self.data])
        qvel = np.stack([d.qvel for d in self.data])
        qpos[self.bad], qvel[self.bad] = np.nan, np.nan
        qpos = torch.from_numpy(qpos).to(self.device, torch.float32)
        qvel = torch.from_numpy(qvel).to(self.device, torch.float32)
        root_q, root_v = qpos[:, self.index["free_q"]], qvel[:, self.index["free_v"]]
        p = root_q[:, :3] - self.origins
        R = quat_to_mat(root_q[:, 3:7])
        w = (R @ root_v[:, 3:6].unsqueeze(-1)).squeeze(-1)
        return (p, R, root_v[:, :3].clone(), w, qpos[:, self.index["joint_q"]], qvel[:, self.index["joint_v"]])

    def observe(self, command_row):
        """TrackingSim.observe in evaluation mode (no noise, no delay) at the command row."""
        return self.mdp.obs(self.state(), self.a_prev, self.xi, command_row)

    def push(self, ids, velocity_xy):
        """Add the change to the roots' planar world velocity, the free joints' qvel[0:2]; nothing else changes."""
        dv = velocity_xy.double().cpu().numpy()
        adr = self.maps["free_v"][:2]
        for k, i in enumerate(ids.tolist()):
            self.data[i].qvel[adr] += dv[k]

    def step(self, action, command_row):
        """One control: ctrl of each joint's actuator = TrackingSim.step_core's target, the default pose plus the
        joint offsets of xi plus the action scale times the action; then DECIMATION mj_step calls per trial. The
        command row is not read: the record's command quantities come from roll_out."""
        target = self.default_q + self.xi[:, 1 : 1 + self.J] + self.alpha * action
        ctrl = target.double().cpu().numpy()
        parts = list(self.pool.map(lambda chunk: self._advance(chunk, ctrl), self.chunks))
        rep = np.concatenate(parts)  # [T, 4]: ncon, nefc, contact arena full, constraint arena full
        self.a_prev = action.clone()
        report = {"contacts": torch.as_tensor(rep[:, 0], device=self.device),
                  "constraint_rows": torch.as_tensor(rep[:, 1], device=self.device),
                  "contacts_beyond_capacity": torch.as_tensor(rep[:, 2] != 0, device=self.device),
                  "rows_beyond_capacity": torch.as_tensor(rep[:, 3] != 0, device=self.device)}  # fmt: skip
        return self.state(), report

    def _advance(self, chunk, ctrl):
        m, out = self.model, np.zeros((len(chunk), 4), dtype=np.int64)
        for k, i in enumerate(chunk):
            d = self.data[i]
            d.ctrl[self.maps["actuator"]] = ctrl[i]
            before = [d.warning[w].number for w in BAD_STATE + FULL]
            t, peak = d.time, [0, 0, 0]
            for _ in range(DECIMATION):
                mujoco.mj_step(m, d)  # releases the interpreter lock
                t += m.opt.timestep
                peak = [max(peak[0], d.ncon), max(peak[1], d.nefc), max(peak[2], d.nl)]
            after = [d.warning[w].number for w in BAD_STATE + FULL]
            # a reset inside a step also sets the time back, which catches a warning count that a reset cleared
            if before[: len(BAD_STATE)] != after[: len(BAD_STATE)] or d.time != t:
                self.bad[i] = True
            out[k] = (d.ncon, d.nefc, after[-2] != before[-2], after[-1] != before[-1])
            self.substep_peaks[i] = peak
        return out

    # ---------------------------------------------------------------- for the checks
    def raw(self):
        """qpos [T, nq] and qvel [T, nv] (float64)."""
        return np.stack([d.qpos for d in self.data]), np.stack([d.qvel for d in self.data])

    def ctrl(self):
        return np.stack([d.ctrl for d in self.data])

    def lift(self, dz: float):
        """Add ``dz`` to every root height, in float32 as on MuJoCo Warp's float32 qpos (the placed heights are float32
        values)."""
        z = self.maps["free_q"][2]
        for d in self.data:
            d.qpos[z] = float(np.float32(d.qpos[z]) + np.float32(dz))

    def close(self):
        self.pool.shutdown()


def warp_fields(env: TrackingSim) -> dict:
    """The fields of MuJoCo Warp's model that mjlab holds per world (expanded for randomization), [worlds, ...]."""
    import warp as wp

    return {name: wp.to_torch(getattr(env.sim.wp_model, name)) for name in sorted(env.sim.expanded_fields)}


def warp_model_arrays(env: TrackingSim) -> dict:
    """Every array field of MuJoCo Warp's model, as numpy."""
    import warp as wp

    m = env.sim.wp_model
    fields = [f.name for f in dataclasses.fields(m)]
    return {k: getattr(m, k).numpy() for k in fields if isinstance(getattr(m, k), wp.array)}


class TrainingSimulator:
    """rollouts.Simulator (the evaluation-mode TrackingSim) with the raw state, ctrl and lift of the checks."""

    def __init__(self, env: TrackingSim):
        self.env, self.sim = env, rollouts.Simulator(env)
        self.place, self.observe, self.push, self.step = self.sim.place, self.sim.observe, self.sim.push, self.sim.step

    def state(self):
        self._synchronize()
        return self.sim.state()

    def _synchronize(self):
        import warp as wp

        wp.synchronize_device(self.env.sim.wp_device)
        torch.cuda.synchronize()

    def raw(self):
        self._synchronize()
        d = self.env.sim.data
        return d.qpos[:].double().cpu().numpy(), d.qvel[:].double().cpu().numpy()

    def ctrl(self):
        self._synchronize()
        return self.env.sim.data.ctrl[:].double().cpu().numpy()

    def lift(self, dz: float):
        z, qpos = self.env.robot.indexing.free_joint_q_adr[2], self.env.sim.data.qpos
        qpos[:, z] = qpos[:, z] + dz


# -------------------------------------------------------------------- reports
def _flags(value: int, enum, prefix: str) -> list:
    return [n for n in dir(enum) if n.startswith(prefix) and value & int(getattr(enum, n))]


def options(model) -> dict:
    """The model's simulation options."""
    o = model.opt
    out = {k: float(getattr(o, k)) for k in ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance",
                                             "ccd_tolerance", "density", "viscosity", "o_margin")}  # fmt: skip
    out |= {k: int(getattr(o, k)) for k in ("iterations", "ls_iterations", "noslip_iterations", "ccd_iterations",
                                            "sdf_iterations", "sdf_initpoints")}  # fmt: skip
    out |= {k: [float(x) for x in getattr(o, k)] for k in ("gravity", "wind", "magnetic", "o_solref", "o_solimp", "o_friction")}
    out |= {"integrator": mujoco.mjtIntegrator(o.integrator).name, "cone": mujoco.mjtCone(o.cone).name,
            "jacobian": mujoco.mjtJacobian(o.jacobian).name, "solver": mujoco.mjtSolver(o.solver).name,
            "disableflags": _flags(o.disableflags, mujoco.mjtDisableBit, "mjDSBL_"),
            "enableflags": _flags(o.enableflags, mujoco.mjtEnableBit, "mjENBL_")}  # fmt: skip
    return out


def describe_model(model) -> dict:
    """Options, per-geom contact settings, explicit and excluded pairs, actuators and joint dofs of ``model``."""
    name = lambda obj, i: mujoco.mj_id2name(model, obj, int(i))  # noqa: E731
    O = mujoco.mjtObj
    geoms = [{"geom": name(O.mjOBJ_GEOM, g), "body": name(O.mjOBJ_BODY, model.geom_bodyid[g]),
              "type": mujoco.mjtGeom(model.geom_type[g]).name, "contype": int(model.geom_contype[g]),
              "conaffinity": int(model.geom_conaffinity[g]), "condim": int(model.geom_condim[g]),
              "priority": int(model.geom_priority[g]), "friction": model.geom_friction[g].tolist(),
              "solmix": float(model.geom_solmix[g]), "solref": model.geom_solref[g].tolist(),
              "solimp": model.geom_solimp[g].tolist(), "margin": float(model.geom_margin[g]), "gap": float(model.geom_gap[g])}
             for g in range(model.ngeom)]  # fmt: skip
    pairs = [{"geom1": name(O.mjOBJ_GEOM, model.pair_geom1[i]), "geom2": name(O.mjOBJ_GEOM, model.pair_geom2[i]),
              "condim": int(model.pair_dim[i]), "friction": model.pair_friction[i].tolist()} for i in range(model.npair)]  # fmt: skip
    excluded = [[name(O.mjOBJ_BODY, int(s) >> 16), name(O.mjOBJ_BODY, int(s) & 0xFFFF)] for s in model.exclude_signature]
    actuators = [{"actuator": name(O.mjOBJ_ACTUATOR, a), "joint": name(O.mjOBJ_JOINT, model.actuator_trnid[a, 0]),
                  "kp": float(model.actuator_gainprm[a, 0]), "biasprm": model.actuator_biasprm[a, :3].tolist(),
                  "forcelimited": bool(model.actuator_forcelimited[a]), "forcerange": model.actuator_forcerange[a].tolist(),
                  "ctrllimited": bool(model.actuator_ctrllimited[a]), "gear": float(model.actuator_gear[a, 0])}
                 for a in range(model.nu)]  # fmt: skip
    dofs = [{"joint": name(O.mjOBJ_JOINT, model.dof_jntid[v]), "armature": float(model.dof_armature[v]),
             "damping": float(model.dof_damping[v]), "frictionloss": float(model.dof_frictionloss[v])}
            for v in range(model.nv)]  # fmt: skip
    return {"options": options(model), "nq": model.nq, "nv": model.nv, "nu": model.nu, "nbody": model.nbody,
            "ngeom": model.ngeom, "narena": model.narena, "geoms": geoms, "pairs": pairs, "excluded_body_pairs": excluded,
            "actuators": actuators, "dofs": dofs}  # fmt: skip


def _unequal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Entries of two arrays of one shape that differ, NaN equal to NaN."""
    out = x != y
    if x.dtype.kind == "f" and y.dtype.kind == "f":
        out &= ~(np.isnan(x) & np.isnan(y))
    return out


def model_differences(a, b) -> dict:
    """{field: entries that differ} over the array fields and options of two MjModels of one structure."""
    out = {}
    for k in dir(a):
        x = getattr(a, k, None)
        if k.startswith("_") or not isinstance(x, np.ndarray):
            continue
        y = getattr(b, k)
        if x.shape != y.shape:
            out[k] = f"shape {x.shape} against {y.shape}"
        elif _unequal(x, y).any():
            out[k] = int(_unequal(x, y).sum())
    oa, ob = options(a), options(b)
    out |= {f"opt.{k}": [oa[k], ob[k]] for k in oa if oa[k] != ob[k]}
    return out


def warp_option_values(wp_opt, names) -> dict:
    """MuJoCo Warp's Option ``wp_opt`` under those of the option names ``names`` (of ``options``) it holds, and its
    impratio_invsqrt: {name: numpy value}, a per-world option (a warp array) as [worlds, ...], an integer, enumeration or
    flags option as an int. Another type raises."""
    import warp as wp

    out = {}
    for k in [*names, "impratio_invsqrt"]:
        if not hasattr(wp_opt, k):
            continue
        x = getattr(wp_opt, k)
        if isinstance(x, wp.array):
            out[k] = x.numpy()
        elif isinstance(x, int):  # enumerations and flags among them
            out[k] = np.asarray(int(x))
        else:
            raise ValueError(f"MuJoCo Warp's option {k} is a {type(x).__name__}, neither a warp array nor an integer")
    return out


def model_field_comparison(model, arrays: dict) -> dict:
    """The fields of the training simulator's model ``arrays`` ({name: [worlds, ...] array, or one of the MjModel's
    shape}) against the field of that name of ``model``: the number of fields compared, and for each field where a
    world differs from the MjModel's value (a float at its float32 rounding) the entries that differ; fields of another
    shape or type, and names the MjModel does not hold, are listed."""
    differ, layout, absent, compared = {}, [], [], 0
    for name, a in sorted(arrays.items()):
        b = getattr(model, name, None)
        if not isinstance(b, np.ndarray):
            absent.append(name)
            continue
        a = np.asarray(a)
        if a.dtype.fields is not None or a.dtype.kind not in "biuf" or b.dtype.kind not in "biuf":
            layout.append(f"{name}: type {a.dtype} against {b.dtype}")
            continue
        if a.shape == b.shape:
            a = a[None]
        if a.shape[1:] != b.shape:
            layout.append(f"{name}: shape {a.shape} against {b.shape}")
            continue
        compared += 1
        ref = np.broadcast_to(b.astype(a.dtype) if a.dtype.kind == "f" else b, a.shape)
        unequal = _unequal(a, ref).any(0)
        if unequal.any():
            differ[name] = int(unequal.sum())
    return {"fields_compared": compared, "differing_entries": differ, "other_layout": layout,
            "not_in_the_mjmodel": absent}  # fmt: skip


def largest(a, b, quaternion: bool = False) -> dict:
    """Largest absolute difference of ``a`` and ``b`` [T, ...] over the trials whose values are finite in both, the
    trial where it is, the median over those trials of each trial's largest difference, and the trials left out.
    ``quaternion``: the last axis holds quaternions, and a quaternion's difference is the smaller of its largest
    component difference to the other and to the other's negative."""
    a, b = np.asarray(a, np.float64).reshape(len(a), -1), np.asarray(b, np.float64).reshape(len(b), -1)
    d = np.abs(a - b)
    if quaternion:
        qa, qb = a.reshape(len(a), -1, 4), b.reshape(len(b), -1, 4)
        d = np.minimum(np.abs(qa - qb).max(-1), np.abs(qa + qb).max(-1))
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    per = d[ok].max(1) if d.shape[1] else np.zeros(int(ok.sum()))
    if not ok.any():
        return {"largest": None, "trial": None, "median_of_trials": None, "nonfinite_trials": int((~ok).sum())}
    trials = np.flatnonzero(ok)
    return {"largest": float(per.max()), "trial": int(trials[per.argmax()]), "median_of_trials": float(np.median(per)),
            "nonfinite_trials": int((~ok).sum())}  # fmt: skip


def state_differences(x, y, J: int, mask=None) -> dict:
    """largest() per group of the release's flat state (mdp.flatten_state) [T, 13 + 2J]."""
    x, y = np.asarray(x), np.asarray(y)
    if mask is not None:
        x, y = x[mask], y[mask]
    if len(x) == 0:
        return {"trials": 0}
    cut = np.cumsum([0, 3, 4, 3, 3, J, J])
    out = {g: largest(x[:, cut[i] : cut[i + 1]], y[:, cut[i] : cut[i + 1]], g == "root_quaternion")
           for i, g in enumerate(STATE_GROUPS)}  # fmt: skip
    return {"trials": len(x)} | out


def raw_differences(c_raw, t_raw, maps: dict) -> dict:
    """largest() of qpos and qvel by part (free joint position and quaternion, joints; linear, body-frame angular and
    joint velocities)."""
    (cq, cv), (tq, tv) = c_raw, t_raw
    fq, fv = maps["free_q"], maps["free_v"]
    return {"qpos_root_position": largest(cq[:, fq[:3]], tq[:, fq[:3]]),
            "qpos_root_quaternion": largest(cq[:, fq[3:]], tq[:, fq[3:]], True),
            "qpos_joints": largest(cq[:, maps["joint_q"]], tq[:, maps["joint_q"]]),
            "qvel_root_linear": largest(cv[:, fv[:3]], tv[:, fv[:3]]),
            "qvel_root_angular_body": largest(cv[:, fv[3:]], tv[:, fv[3:]]),
            "qvel_joints": largest(cv[:, maps["joint_v"]], tv[:, maps["joint_v"]])}  # fmt: skip


def same_records(r1: dict, r2: dict) -> dict:
    """Whether two records are equal field by field (NaN equal to NaN); for the fields that differ, the number of
    entries and the largest absolute difference."""
    if set(r1) != set(r2):
        raise ValueError(f"the records hold different fields: {sorted(set(r1) ^ set(r2))}")
    differ = {}
    for k in sorted(r1):
        a, b = r1[k], r2[k]
        if a.shape != b.shape:
            differ[k] = {"entries": f"shape {a.shape} against {b.shape}", "largest": None}
        elif _unequal(a, b).any():
            both = np.isfinite(a) & np.isfinite(b) if a.dtype.kind == "f" else np.ones(a.shape, bool)
            gap = np.abs(a.astype(np.float64) - b.astype(np.float64))[both]
            differ[k] = {"entries": int(_unequal(a, b).sum()), "largest": float(gap.max()) if gap.size else None}
    return {"identical": not differ, "differing_fields": differ}


def first_action(env, policy, goal_frames: int, motion, row):
    """The action of roll_out's first control at ``row``: the history filled with the first observation."""
    o = env.observe(row)
    return policy(o.unsqueeze(1).repeat(1, policy.hist, 1), motion.goal(row, goal_frames))


@torch.no_grad()
def checks(make_c, training, policy, goal_frames: int, mdp: MDP, trials: dict, record: dict, record_bodies: list,
           bodies: list, controls: int, recipe: bool) -> dict:  # fmt: skip
    """Checks (a), (b) and (d) of ROLLOUTS.md ("Rollouts on MuJoCo C") between MuJoCo C objects ``make_c()`` and the
    training simulator ``training`` (TrainingSimulator, or any object with its methods) on the scene's ``trials``,
    against ``record`` (the arrays of a tools/rollouts.py record of the same scene; ``record_bodies`` its tracked body
    names). Raises where the record's rows or bodies are not those of the scene and the robot, or where the two
    engines' ctrl differ after the same action (a joint-to-actuator mismatch). Reports differences; decides nothing."""
    kin, motion = mdp.kin, mdp.motion
    T, J = len(trials["command_row"]), mdp.num_j
    if record_bodies != bodies:
        raise ValueError(f"body order: the record's tracked bodies {record_bodies} are not the robot's {bodies}")
    for k in ("command_row", "body_row"):
        if record[k].tolist() != trials[k].tolist():
            raise ValueError(f"the record's {k} are not the scene's")
    ids, row = torch.arange(T, device=trials["command_row"].device), trials["command_row"]
    uniforms = {k: trials[k] for k in rollouts.UNIFORMS}
    place = lambda env: env.place(ids, row, trials["body_row"], uniforms)  # noqa: E731
    out = {}

    # (a) placement against the record
    c = make_c()
    place(c), place(training)
    S = c.state()
    bp, bR = kin.forward(*S)[:2]
    bq = mat_to_quat(bR)
    x = flatten_state(S).cpu().numpy()
    tx = flatten_state(training.state()).cpu().numpy()
    body_ids = []
    for b in bodies:
        i = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + b)
        if i < 0:
            raise ValueError(f"body order: the model has no body {PREFIX + b}")
        body_ids.append(i)
    xpos, xquat = [], []
    for d in c.data:  # MuJoCo C's own kinematics of the placed state, on a copy of each trial's data
        k = copy.copy(d)
        mujoco.mj_kinematics(c.model, k)
        xpos.append(k.xpos[body_ids]), xquat.append(k.xquat[body_ids])
    origins = c.origins.cpu().numpy()[:, None]
    out["a_placement"] = {
        "trials": T,
        "start_state_against_record": state_differences(x, record["start_state"], J),
        "start_body_pos_w_against_record": largest(bp.cpu().numpy(), record["start_body_pos_w"]),
        "start_body_quat_w_against_record": largest(bq.cpu().numpy(), record["start_body_quat_w"], True),
        "training_simulator_start_state_against_record": state_differences(tx, record["start_state"], J),
        "qpos_qvel_against_training_simulator": raw_differences(c.raw(), training.raw(), c.maps),
        "mujoco_kinematics_body_pos_against_record": largest(np.stack(xpos) - origins, record["start_body_pos_w"]),
        "mujoco_kinematics_body_quat_against_record": largest(np.stack(xquat), record["start_body_quat_w"], True),
    }
    c.close()

    # (b) one control from the same placed state and the same action
    def control(env, a, lift):
        place(env)
        if lift:
            env.lift(LIFT_M)
        before = env.raw()
        x0 = flatten_state(env.state()).cpu().numpy()
        S1, rep = env.step(a, row)
        x1 = flatten_state(S1).cpu().numpy()
        return before, x1, {k: v.cpu().numpy() for k, v in rep.items()}, state_differences(x1, x0, J)

    c = make_c()
    place(c)
    a = first_action(c, policy, goal_frames, motion, row)
    out["b_one_control"] = {"action": "the policy's first action at the C engine's placed state, one tensor for both"}
    for name, lift in (("lifted", True), ("placed", False)):
        cb, cx, crep, _ = control(c, a, lift)
        tb, tx, trep, change = control(training, a, lift)
        cc, tc = c.ctrl(), training.ctrl()
        if not np.array_equal(cc, tc):
            raise ValueError(f"joint order: ctrl differs after the same action ({int((cc != tc).sum())} entries)")
        # MuJoCo C over every substep of the control; the training simulator after its last substep only
        peaks = c.substep_peaks.copy()
        free = (peaks[:, 0] == 0) & (trep["contacts"] == 0)
        no_rows = (peaks[:, 1] == 0) & (trep["constraint_rows"] == 0)
        part = {"lift_m": LIFT_M if lift else 0.0, "ctrl_equal": True,
                "state_before_against_training_simulator": raw_differences(cb, tb, c.maps),
                "contacts": {"mujoco_c": crep["contacts"].tolist(), "training_simulator": trep["contacts"].tolist(),
                             "mujoco_c_largest_over_substeps": peaks[:, 0].tolist()},
                "constraint_rows": {"mujoco_c": crep["constraint_rows"].tolist(),
                                    "training_simulator": trep["constraint_rows"].tolist(),
                                    "mujoco_c_largest_over_substeps": peaks[:, 1].tolist(),
                                    "mujoco_c_joint_limit_rows_largest_over_substeps": peaks[:, 2].tolist()},
                "state_change_over_the_control_on_the_training_simulator": change,
                "state_after_all_trials": state_differences(cx, tx, J),
                "state_after_without_contact": state_differences(cx, tx, J, free),
                "state_after_with_contact": state_differences(cx, tx, J, ~free),
                "state_after_without_constraint_rows": state_differences(cx, tx, J, no_rows)}  # fmt: skip
        _, tx2, _, _ = control(training, a, lift)
        _, cx2, _, _ = control(c, a, lift)
        part["training_simulator_twice"] = state_differences(tx, tx2, J)
        part["mujoco_c_twice"] = state_differences(cx, cx2, J)
        out["b_one_control"][name] = part
    c.close()

    # (d) two rollouts of the first controls on MuJoCo C
    runs, seconds = [], []
    for _ in range(2):
        c = make_c()
        t0 = time.perf_counter()
        runs.append(rollouts.roll_out(c, policy, goal_frames, mdp, trials, controls, recipe))
        seconds.append(time.perf_counter() - t0)
        c.close()
    out["d_repeat"] = {"controls": controls, "seconds": seconds, **same_records(*runs)}
    return out


# -------------------------------------------------------------------- the program
def describe(args, run: dict, ckpt: str, ck: dict, library: dict, scene: dict, motion, robot, mdp, env, simulator: dict,
             card: str) -> dict:  # fmt: skip
    """description.json of tools/rollouts.py with the simulator section of MuJoCo C and the mujoco version."""
    a_, bodies = ck["args"], list(robot.tracked_bodies)
    versions = {p: importlib.metadata.version(p) for p in ("torch", "warp-lang", "mujoco-warp", "mjlab", "mujoco")}
    return {
        "run_dir": os.path.abspath(args.run_dir), "checkpoint": os.path.abspath(ckpt), "iteration": ck["iteration"],
        "run_args": run,
        "source_revision": {"run": os.environ.get("RUN_SOURCE_REVISION"), "rollouts": os.environ.get("SOURCE_REVISION")},
        "scene_path": os.path.abspath(args.scene), "scene": scene, "termination": args.termination, "controls": args.controls,
        "library": library | {"clips": len(motion.names), "rows": motion.num_frames},
        "card": card,
        "simulator": simulator,
        "bodies": {"names": bodies, "pelvis": bodies.index("pelvis"), "torso_link": bodies.index("torso_link")},
        "settings": {"eval_mode": env.eval_mode, "randomize": env.randomize, "reset_noise": env.reset_noise,
                     "action_noise": 0.0, "goal_frames": a_["goal_frames"], "hist": a_["hist"], "amp": a_["amp"],
                     "allow_tf32": True, "elementwise_rotations": True, "control_dt": CONTROL_DT,
                     "term_height": mdp.cfg.term_height, "term_tilt": mdp.cfg.term_tilt,
                     "reset_half_widths": {"position": robot.dr.reset_pos, "rotation": robot.dr.reset_rot,
                                           "velocity": robot.dr.reset_lin_vel + robot.dr.reset_ang_vel,
                                           "joints": robot.dr.reset_joint_pos},
                     "versions": versions},
    }  # fmt: skip


DIFFERENCES = (
    "the model and the state are float64 (MuJoCo Warp: float32); the policy reads the state rounded to float32",
    "a push's velocity change is added in float64",
    "no fixed contact or constraint-row capacity: contacts and rows come from the MjData arena (narena bytes); "
    "contacts_beyond_capacity and rows_beyond_capacity are set where MuJoCo C's CONTACTFULL or CNSTRFULL warning "
    "count rose during the control",
    "a control in which MuJoCo C warns of a bad position, velocity, acceleration or ctrl, or in which the data's time "
    "went back (MuJoCo C resets the data of a bad state), is recorded as a non-finite state",
    "where MuJoCo Warp's model holds another solver tolerance than the compiled MjModel (put_model raises it to at "
    "least 1e-6), MuJoCo C steps with MuJoCo Warp's value (options_written)",
    "the engine itself: collision routines, constraint solver implementation (MuJoCo Warp, for one, puts a floor of "
    "1e-6 under a line search's gradient tolerance), Jacobian representation, arithmetic",
)


def simulator_section(model, written: dict, options_written: dict, robot, threads: int) -> dict:
    """The description's simulator section. ``solver`` is the robot's solver setting, as tools/rollouts.py records it;
    ``options`` are those MuJoCo C steps with, ``options_written`` those of them taken from MuJoCo Warp's model."""
    return {"engine": "MuJoCo C", "mujoco": mujoco.__version__, "threads": threads, "differences": list(DIFFERENCES),
            "model": "the evaluation-mode TrackingSim's MjModel (sim.mj_model) with the entries and options of MuJoCo "
                     "Warp's model that differ from it",
            "written_after_compilation": written, "options_written": options_written,
            "state": "float64, one MjData per trial for the whole rollout; float32 at the policy's interface",
            "nconmax": None, "njmax": None, "naconmax_pooled": None, "narena": int(model.narena),
            "solver": list(robot.solver), "physics_dt": float(model.opt.timestep), "decimation": DECIMATION,
            "options": options(model)}  # fmt: skip


def parse(argv=None):
    parser = argparse.ArgumentParser(description="Roll a checkpoint's policy out from the placed starts of a scene, "
                                                 "with the physics stepped by MuJoCo C.")  # fmt: skip
    parser.add_argument("--run_dir", required=True, help="run directory of blinkmimic2.train (args.json, ckpt_NNNNNN.pt)")
    parser.add_argument("--ckpt", type=int, required=True, help="iteration N of the checkpoint ckpt_NNNNNN.pt")
    parser.add_argument("--scene", required=True, help="scene file (JSON, ROLLOUTS.md)")
    parser.add_argument("--termination", required=True, choices=["recipe", "nonfinite"],
                        help="recipe: the training recipe's termination (MDP.termination) or a non-finite state ends a trial; "
                             "nonfinite: only a non-finite state")
    parser.add_argument("--controls", type=int, required=True, help="controls per trial (50 per second)")
    parser.add_argument("--threads", type=int, required=True, help="threads that step the trials' MjData")
    parser.add_argument("--checks", metavar="RECORD",
                        help="write checks.json instead of rollouts: RECORD is a tools/rollouts.py output on the same scene")
    parser.add_argument("--out", required=True, help="output directory; it must not exist")
    args = parser.parse_args(argv)
    if args.controls < 1:
        parser.error("--controls must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if os.path.exists(args.out):
        parser.error(f"the output directory {args.out} exists")
    if args.checks is not None:
        for f in ("description.json", "rollouts.npz"):
            if not os.path.isfile(os.path.join(args.checks, f)):
                parser.error(f"--checks {args.checks} holds no {f}")
    return args


def main(argv=None):
    args = parse(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("the policy runs on cuda:0, as in tools/rollouts.py, and no CUDA device is available")
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True  # as tools/rollouts.py
    rotations.ELEMENTWISE = True
    run = json.load(open(os.path.join(args.run_dir, "args.json")))
    ckpt = os.path.join(args.run_dir, f"ckpt_{args.ckpt:06d}.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    robot = get_robot(run["robot"])
    kin = Kinematics(robot, dev)
    library = {k: run[k] for k in ("motion_dir", "bones_list", "bones_lib", "motion_store")}
    motion = MotionLib.load(robot, kin, dev, *library.values())
    scene = json.load(open(args.scene))
    trials = {k: v.to(dev) for k, v in rollouts.read_scene(scene, motion.num_frames, robot.num_joints, args.controls).items()}
    record = recorded = None
    if args.checks is not None:
        recorded = json.load(open(os.path.join(args.checks, "description.json")))
        expected = {"scene": scene, "library": library | {"clips": len(motion.names), "rows": motion.num_frames}}
        differing = [k for k, v in expected.items() if recorded[k] != v]
        if differing:
            raise ValueError(f"--checks {args.checks}: its {differing} are not those of this scene and run")
        with np.load(os.path.join(args.checks, "rollouts.npz")) as z:
            record = {k: z[k] for k in ("command_row", "body_row", "start_state", "start_body_pos_w", "start_body_quat_w")}
    os.makedirs(args.out)

    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
    a_ = ck["args"]
    widths = lambda key: tuple(int(x) for x in a_[key].split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, hist=a_["hist"],
                enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
    pi.load_state_dict(ck["pi"])
    pi.eval()
    networks.AMP_DTYPE = torch.bfloat16 if a_["amp"] == "bf16" else None
    env = TrackingSim(make_sim(robot), len(trials["command_row"]), robot, motion, mdp, dev, eval_mode=True)
    model, written = trainer_model(env.sim.mj_model, warp_fields(env))
    warp_opt = warp_option_values(env.sim.wp_model.opt, list(options(model)))
    options_written, options_other = trainer_options(model, warp_opt)
    if record is None:
        check_options(options_other)  # the check mode reports them instead
    simulator = simulator_section(model, written, options_written, robot, args.threads)
    description = describe(args, run, ckpt, ck, library, scene, motion, robot, mdp, env, simulator,
                           torch.cuda.get_device_name(0))  # fmt: skip
    make_c = lambda: MujocoC.from_training(env, model, args.threads)  # noqa: E731
    if record is not None:
        bodies, recorded_bodies = list(robot.tracked_bodies), recorded["bodies"]["names"]
        out = {"description": description, "record": os.path.abspath(args.checks),
               "default_pose": {"mjlab_init_state_minus_robot_default": float(
                   (env.default_q - mdp.default_q).abs().max())},
               "c_model": describe_model(model),
               "c_model_against_training_mj_model": model_differences(model, env.sim.mj_model),
               "c_options_against_training_simulator": option_comparison(model, warp_opt),
               "options_that_differ_and_are_not_written": options_other,
               "training_simulator_only_options": sorted({f.name for f in dataclasses.fields(env.sim.wp_model.opt)}
                                                         - set(warp_opt)),
               "c_only_options": sorted(set(options(model)) - set(warp_opt) - {"impratio"}),
               "training_simulator_model_against_c_model": model_field_comparison(model, warp_model_arrays(env))}  # fmt: skip
        out |= checks(make_c, TrainingSimulator(env), pi, a_["goal_frames"], mdp, trials, record, recorded_bodies, bodies,
                      min(50, args.controls), args.termination == "recipe")  # fmt: skip
        with open(os.path.join(args.out, "checks.json"), "x") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {os.path.join(args.out, 'checks.json')}")
        return
    c = make_c()  # its refusals come before any file is written
    with open(os.path.join(args.out, "description.json"), "x") as f:
        json.dump(description, f, indent=1)
    rec = rollouts.roll_out(c, pi, a_["goal_frames"], mdp, trials, args.controls, args.termination == "recipe")
    c.close()
    np.savez_compressed(os.path.join(args.out, "rollouts.npz"), **rec)
    print(f"{len(rec['command_row'])} trials x {args.controls} controls on MuJoCo C: {int(rec['done'].sum())} ended; "
          f"wrote {args.out}")  # fmt: skip


if __name__ == "__main__":
    main()
