"""Roll the deployed policy of one checkpoint out from placed starts and record every control (ROLLOUTS.md).

    python tools/rollouts.py --run_dir logs/run_s1 --ckpt 6500 --scene scene.json --termination nonfinite \\
        --controls 1000 --out rollouts/run_s1_6500

The policy is loaded as tools/evaluate.py loads it (the EMA weights "pi" of ckpt_NNNNNN.pt) and acts closed-loop and
deterministically in the evaluation-mode simulator (no randomization, no noise), one world per trial of the scene. The
command (the reference row the policy tracks) advances one row per control and holds at its clip's last row. The control
period (--control_period) must be the one the run trained at (motion.run_control_dt). "The release" in this file is
BlinkMimic 2 v1.0.0, the code this tool reads.
"""

import argparse
import importlib.metadata
import json
import math
import os

import numpy as np
import torch
import warp as wp
from mujoco_warp._src.types import OverflowType  # not exported by the mujoco_warp package

import blinkmimic2.networks as networks
import blinkmimic2.rotations as rotations
from blinkmimic2.dr import XiLayout
from blinkmimic2.env import PHYSICS_DT, TrackingSim, make_sim
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import MDP, flatten_state
from blinkmimic2.motion import MotionLib, run_control_dt
from blinkmimic2.networks import Policy, policy_kwargs
from blinkmimic2.robot import get_robot
from blinkmimic2.rotations import mat_to_quat

UNIFORMS = ("position", "rotation", "velocity", "joints")


def _keys(x, required: set, optional: set, where: str):
    if not isinstance(x, dict) or not required <= set(x) <= required | optional:
        found = sorted(x) if isinstance(x, dict) else type(x).__name__
        raise ValueError(f"{where} holds {found}; required keys {sorted(required)}, optional {sorted(optional)}")


def _index(x, end: int, where: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < end:
        raise ValueError(f"{where} = {x!r} is not an integer in [0, {end})")
    return x


def _numbers(x, n: int, bound: float, where: str) -> list:
    ok = isinstance(x, list) and len(x) == n and all(
        isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e) and abs(e) <= bound for e in x)
    if not ok:
        raise ValueError(f"{where} is not a list of {n} finite numbers in [-{bound}, {bound}]")
    return x


def read_scene(scene, num_rows: int, num_joints: int, controls: int) -> dict:
    """The trials of a scene as CPU tensors: command_row, body_row [T] (long); position [T, 3], rotation [T, 3], velocity
    [T, 6], joints [T, J] (the unit reset uniforms); push_control [T] (long, -1 without a push) and push_velocity_xy
    [T, 2]. Unknown keys, rows outside the library of ``num_rows`` rows, uniforms outside [-1, 1] and pushes outside the
    ``controls`` controls raise ValueError."""
    _keys(scene, {"trials"}, set(), "the scene")
    if not isinstance(scene["trials"], list) or not scene["trials"]:
        raise ValueError("the scene's trials are not a non-empty list")
    sizes = {"position": 3, "rotation": 3, "velocity": 6, "joints": num_joints}
    out = {k: [] for k in ("command_row", "body_row", *UNIFORMS, "push_control", "push_velocity_xy")}
    for i, t in enumerate(scene["trials"]):
        _keys(t, {"command_row"}, {"body_row", "reset_uniforms", "push"}, f"trial {i}")
        out["command_row"].append(_index(t["command_row"], num_rows, f"trial {i} command_row"))
        out["body_row"].append(_index(t.get("body_row", t["command_row"]), num_rows, f"trial {i} body_row"))
        u = t.get("reset_uniforms", {k: [0.0] * n for k, n in sizes.items()})
        _keys(u, set(UNIFORMS), set(), f"trial {i} reset_uniforms")
        for k, n in sizes.items():
            out[k].append(_numbers(u[k], n, 1.0, f"trial {i} reset_uniforms {k}"))
        push = t.get("push", {"control": -1, "velocity_xy": [0.0, 0.0]})
        if "push" in t:
            _keys(push, {"control", "velocity_xy"}, set(), f"trial {i} push")
            _index(push["control"], controls, f"trial {i} push control")
        out["push_control"].append(push["control"])
        out["push_velocity_xy"].append(_numbers(push["velocity_xy"], 2, math.inf, f"trial {i} push velocity_xy"))
    return {k: torch.tensor(v, dtype=torch.long if k in ("command_row", "body_row", "push_control") else torch.float32)
            for k, v in out.items()}  # fmt: skip


@torch.no_grad()
def roll_out(env, policy, goal_frames: int, mdp: MDP, trials: dict, controls: int, recipe: bool) -> dict:
    """Run the trials (read_scene, on the device of ``env``) closed-loop for ``controls`` controls and return the record
    of ROLLOUTS.md as numpy arrays.

    ``env`` holds one world per trial: place(ids, command_row, body_row, uniforms) places the trials ``ids``, state() is
    the state tuple (MDP) of every world, observe(command_row) the observation, push(ids, velocity_xy) adds planar world
    velocities to roots, and step(action, command_row) runs one control and returns the state after it and a report of
    [T] tensors: contacts, constraint_rows, contacts_beyond_capacity, rows_beyond_capacity. ``policy(history, goal)``
    gives the actions, with ``policy.hist`` observations in the history. ``recipe``: the training recipe's termination
    ends a trial; otherwise only a non-finite state does. A trial that has ended is placed again at its start before every
    later control, with zero actions, so that it stays finite; its records are NaN, -1 or false."""
    m, kin, cfg = mdp.motion, mdp.kin, mdp.cfg
    row = trials["command_row"].clone()
    T = len(row)

    def place(ids):
        env.place(ids, trials["command_row"][ids], trials["body_row"][ids], {k: trials[k][ids] for k in UNIFORMS})

    place(torch.arange(T, device=row.device))
    S = env.state()
    bp, bR = kin.forward(*S)[:2]
    out = {"start_state": flatten_state(S), "start_body_pos_w": bp, "start_body_quat_w": mat_to_quat(bR)}
    ended = torch.zeros(T, dtype=torch.bool, device=row.device)
    hist, rec = None, []
    for c in range(controls):
        o = env.observe(row)
        hist = o.unsqueeze(1).repeat(1, policy.hist, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = torch.where(ended.unsqueeze(-1), 0.0, policy(hist, m.goal(row, goal_frames)))
        push = (trials["push_control"] == c) & ~ended
        if bool(push.any()):
            ids = push.nonzero().squeeze(-1)
            env.push(ids, trials["push_velocity_xy"][ids])
        S1, report = env.step(a, row)
        running = ~ended
        last = m.clip_end[m.clip_id[row]]
        after = torch.minimum(row + 1, last)
        x = flatten_state(S1)
        nonfinite = ~torch.isfinite(x).all(-1)
        # the release's termination (MDP.termination) against the command row after the control, term by term
        height = (m.root_p[after][:, 2] - S1[0][:, 2]).abs() > cfg.term_height
        tilt = (m.root_R[after][:, 2, 2] - S1[1][:, 2, 2]).abs() > cfg.term_tilt
        done = running & ((nonfinite | height | tilt) if recipe else nonfinite)
        ok = running & ~nonfinite
        bp, bR = kin.forward(*S1)[:2]
        cbp, cbR = m.ref_bodies(after)[:2]
        nan = lambda v, keep: torch.where(keep.view(-1, *([1] * (v.dim() - 1))), v, float("nan"))  # noqa: E731
        neg = lambda v: torch.where(running, v, -1)  # noqa: E731
        rec.append({
            "body_pos_w": nan(bp, ok), "body_quat_w": nan(mat_to_quat(bR), ok),
            "command_body_pos_w": nan(cbp, ok), "command_body_quat_w": nan(mat_to_quat(cbR), ok),
            "command_row_before": neg(row), "command_row_after": neg(after), "clip_last_row": neg(last),
            "available": ok, "done": done, "nonfinite": running & nonfinite,
            "termination_height": running & height, "termination_tilt": running & tilt,
            "contacts": neg(report["contacts"]), "constraint_rows": neg(report["constraint_rows"]),
            "contacts_beyond_capacity": running & report["contacts_beyond_capacity"],
            "rows_beyond_capacity": running & report["rows_beyond_capacity"],
            "action": nan(a, running), "state": nan(x, ok), "push_applied": push,
        })  # fmt: skip
        ended = ended | done
        row = torch.where(ended, row, after)
        if bool(ended.any()):
            place(ended.nonzero().squeeze(-1))
    out |= {k: torch.stack([r[k] for r in rec]) for k in rec[0]}
    out |= {k: trials[k] for k in ("command_row", "body_row", "push_control", "push_velocity_xy")}
    cast = lambda v: v if v.dtype == torch.bool else v.to(torch.float32 if v.is_floating_point() else torch.int32)  # noqa: E731
    return {k: cast(v).cpu().numpy() for k, v in out.items()}


class Simulator:
    """The evaluation-mode TrackingSim of roll_out, with MuJoCo Warp's report of each control: the contacts (from
    contact.worldid over the stored contacts) and constraint rows (nefc, at most njmax) each world used in the
    control's last physics substep, and the overflow bits of the control's substeps (NEFC per world; BROADPHASE and
    NARROWPHASE, set on every world when the contact array pooled over the worlds overflows)."""

    def __init__(self, env: TrackingSim):
        self.env, self.data = env, env.sim.wp_data
        self.nacon, self.nefc = wp.to_torch(self.data.nacon), wp.to_torch(self.data.nefc)
        self.overflow, self.contact_world = wp.to_torch(self.data.overflow), wp.to_torch(self.data.contact.worldid)
        self.slot, self.njmax = torch.arange(int(self.data.naconmax), device=env.device), int(self.data.njmax)

    def place(self, ids, command_row, body_row, uniforms):
        self.env.reset(ids, command_row, uniforms, body_row)

    def state(self):
        return self.env.state()

    def observe(self, command_row):
        self.env.f = command_row.clone()
        return self.env.observe()

    def push(self, ids, velocity_xy):
        """Add the change to the roots' planar world velocity, the free joints' qvel[0:2]; nothing else changes."""
        adr, qvel = self.env.robot.indexing.free_joint_v_adr[:2], self.env.sim.data.qvel
        qvel[ids[:, None], adr] = qvel[ids[:, None], adr] + velocity_xy

    def step(self, action, command_row):
        env, m = self.env, self.env.motion
        # TrackingSim.step reads the rows f and f + 1; f + 1 is the command row after the control, held at the clip end
        env.f = torch.minimum(command_row + 1, m.clip_end[m.clip_id[command_row]]) - 1
        self.data.overflow.zero_()  # the steps OR their bits into it; only a reset clears it
        env.step(action)
        wp.synchronize_device(env.sim.wp_device)
        world = torch.where(self.slot < self.nacon.clamp(max=len(self.slot)), self.contact_world.long(), env.N)
        contacts = torch.zeros(env.N + 1, dtype=torch.long, device=env.device).index_add_(0, world, torch.ones_like(world))
        bits = self.overflow.long()
        report = {"contacts": contacts[: env.N], "constraint_rows": self.nefc.long().clamp(max=self.njmax),
                  "contacts_beyond_capacity": (bits & int(OverflowType.BROADPHASE | OverflowType.NARROWPHASE)) != 0,
                  "rows_beyond_capacity": (bits & int(OverflowType.NEFC)) != 0}  # fmt: skip
        torch.cuda.synchronize()  # the copies are made before the next step writes the counters
        return env.state(), report


def main():
    parser = argparse.ArgumentParser(description="Roll a checkpoint's policy out from the placed starts of a scene.")
    parser.add_argument("--run_dir", required=True, help="run directory of blinkmimic2.train (args.json, ckpt_NNNNNN.pt)")
    parser.add_argument("--ckpt", type=int, required=True, help="iteration N of the checkpoint ckpt_NNNNNN.pt")
    parser.add_argument("--scene", required=True, help="scene file (JSON, ROLLOUTS.md)")
    parser.add_argument("--termination", required=True, choices=["recipe", "nonfinite"],
                        help="recipe: the training recipe's termination (MDP.termination) or a non-finite state ends a trial; "
                             "nonfinite: only a non-finite state")
    parser.add_argument("--controls", type=int, required=True, help="controls per trial (50 per second at 0.02 s)")
    parser.add_argument("--control_period", type=float, default=0.02,
                        help="seconds per control: the period the run trained at (args.json control_period; 0.02 for runs "
                             "without it), else refused")
    parser.add_argument("--out", required=True, help="output directory; it must not exist")
    args = parser.parse_args()
    if args.controls < 1:
        parser.error("--controls must be at least 1")
    if os.path.exists(args.out):
        parser.error(f"the output directory {args.out} exists")
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True  # as tools/evaluate.py
    rotations.ELEMENTWISE = True  # as train.py: the 3x3 products of the kinematics, observation and goal without TF32
    run = json.load(open(os.path.join(args.run_dir, "args.json")))
    if args.control_period != run_control_dt(run):
        parser.error(f"--control_period {args.control_period} s: the run trained at {run_control_dt(run)} s")
    ckpt = os.path.join(args.run_dir, f"ckpt_{args.ckpt:06d}.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    robot = get_robot(run["robot"])
    kin = Kinematics(robot, dev)
    library = {k: run[k] for k in ("motion_dir", "bones_list", "bones_lib", "motion_store")}
    motion = MotionLib.load(robot, kin, dev, *library.values(), args.control_period)
    scene = json.load(open(args.scene))
    trials = {k: v.to(dev) for k, v in read_scene(scene, motion.num_frames, robot.num_joints, args.controls).items()}
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
    bodies = list(robot.tracked_bodies)
    description = {
        "run_dir": os.path.abspath(args.run_dir), "checkpoint": os.path.abspath(ckpt), "iteration": ck["iteration"],
        "run_args": run,
        "source_revision": {"run": os.environ.get("RUN_SOURCE_REVISION"), "rollouts": os.environ.get("SOURCE_REVISION")},
        "scene_path": os.path.abspath(args.scene), "scene": scene, "termination": args.termination, "controls": args.controls,
        "library": library | {"clips": len(motion.names), "rows": motion.num_frames},
        "card": torch.cuda.get_device_name(0),
        "simulator": {"nconmax": robot.nconmax, "njmax": robot.njmax, "naconmax_pooled": int(env.sim.wp_data.naconmax),
                      "solver": list(robot.solver), "physics_dt": PHYSICS_DT, "decimation": env.decimation},
        "bodies": {"names": bodies, "pelvis": bodies.index("pelvis"), "torso_link": bodies.index("torso_link")},
        "settings": {"eval_mode": env.eval_mode, "randomize": env.randomize, "reset_noise": env.reset_noise,
                     "action_noise": 0.0, "goal_frames": a_["goal_frames"], "hist": a_["hist"], "amp": a_["amp"],
                     "allow_tf32": True, "elementwise_rotations": True, "control_dt": motion.dt,
                     "term_height": mdp.cfg.term_height, "term_tilt": mdp.cfg.term_tilt,
                     "reset_half_widths": {"position": robot.dr.reset_pos, "rotation": robot.dr.reset_rot,
                                           "velocity": robot.dr.reset_lin_vel + robot.dr.reset_ang_vel,
                                           "joints": robot.dr.reset_joint_pos},
                     "versions": {p: importlib.metadata.version(p) for p in ("torch", "warp-lang", "mujoco-warp", "mjlab")}},
    }  # fmt: skip
    json.dump(description, open(os.path.join(args.out, "description.json"), "w"), indent=1)
    rec = roll_out(Simulator(env), pi, a_["goal_frames"], mdp, trials, args.controls, args.termination == "recipe")
    np.savez_compressed(os.path.join(args.out, "rollouts.npz"), **rec)
    print(f"{len(rec['command_row'])} trials x {args.controls} controls: {int(rec['done'].sum())} ended; wrote {args.out}")


if __name__ == "__main__":
    main()
