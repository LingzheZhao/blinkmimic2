"""Open-loop error of a training's world model on simulator segments recorded by tools/rollouts.py, split by
the robot's posture and by the reference's get-ups.

    python tools/world_model_error.py --run_dir RUN [--ckpt last] --scene SCENE --motion_dir DIR \\
        --first_rows ROLLOUTS [...] --placed ROLLOUTS [...] --out OUT.json [--controls_ahead 8] [--contacts]

The world model (the class ``WorldModel`` of networks.py, trained by train.py) is the one of RUN/last.pt
(or of ckpt_NNNNNN.pt with --ckpt N, which holds it at the last iteration only), built as train.py builds it
(networks.build_world_model) with its normalizers from the checkpoint. A rollout record holds, per trial,
the placed state before the first control (start_state), the state after each control (state) and each control's
action (action, the policy's output). The **segment** starting at control t takes the state before it, the actions of
controls t .. t+H-1 (H = --controls_ahead) and, as the previous action of control t, the action of control t-1 (0 at
the first control, as a reset sets it). The simulator context xi is the evaluation simulator's (``evaluation_xi``):
randomization is off in the rollouts. The model is stepped open loop over the H controls and compared with the
recorded states.

Error: train.py's error of the world model (networks.world_model_normalized_error), the frame-local
state error over max(inc_norm.std(), lower bound), squared and averaged over the 6 + 6 + 2J components, after the
first control of a segment and after its last (train.py logs the latter as wm_err8 on its own training segments of 8
controls); and the same average within each state group (root position, root rotation, root linear velocity, root
angular velocity, joint positions, joint velocities), in the same normalized units.

Segments: every control t at which the H states after t are recorded and finite and no control before the segment's
last ends the trial; for a first-row trial also t + H - 1 <= the clip's last control (control k targets the clip's
first row + k + 1), so the hold at the clip's last row is left out. Splits, by the recorded robot state before the
segment: a **low-posture state** when the pelvis is below 0.50 m or the z component of the pelvis up axis in the world
frame is below 0.5, an **upright-posture state** otherwise ("low row" and "upright row" are kept for rows of the motion
library); and for first-row trials, **reference get-up windows**: t inside a get-up
window of the reference (studies/recovery_pair/groups.py: rows [e - 50, e + 50] around the last row e of a low span of
at least 25 rows). With --contacts, a further split by **non-foot contact**: MuJoCo (C, on the CPU) collision of the
robot's model as mjlab compiles it, at the recorded state before the segment; a segment starts in non-foot contact
when a contact with penetration (dist < 0) joins a floor geom and a robot geom outside the two ankle-roll links.

Runs on the CPU in float32 (the trainer runs the model's matrix products in bf16 autocast).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import blinkmimic2.networks as networks
import blinkmimic2.rotations as rotations
from blinkmimic2.dr import XiLayout
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import unflatten_state
from blinkmimic2.networks import build_world_model, world_model_error_components, world_model_error_scale
from blinkmimic2.robot import get_robot

LOW_HEIGHT, LOW_UP_Z = 0.50, 0.5
# the state groups of the error's components, in order: (name, width as a function of the joint count J)
STATE_GROUPS = (("root_position", lambda J: 3), ("root_rotation", lambda J: 3), ("root_linear_velocity", lambda J: 3),
                ("root_angular_velocity", lambda J: 3), ("joint_positions", lambda J: J), ("joint_velocities", lambda J: J))
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def evaluation_xi(layout: XiLayout) -> torch.Tensor:
    """xi of the evaluation-mode simulator (DomainRandomizer.startup without randomization, no per-episode draws):
    friction 1, joint offsets and centre-of-mass offset 0, scale terms 1, sensing terms 0. A layout with a contact
    solref term (read from the compiled model) is refused."""
    xi = torch.zeros(layout.dim)
    for name, sl in layout.slices.items():
        if name == "solref":
            raise ValueError("a solref term needs the compiled model's default")
        if name in ("friction", "com_mass", "link_mass", "kp", "kd", "armature", "damping", "qd_tau_scale"):
            xi[sl] = 1.0
    return xi


def segments(start_state, state, action, available, done, controls_ahead, last_control=None, stride=1):
    """The segments of one record: trial and first-control indices [W] of every segment, with ``seq`` [C + 1, T, D]
    (seq[k] the state before control k) and ``prev`` [C, T, J] (the previous action of each control, 0 at the first).
    ``last_control`` [T] (first-row trials): the last control of each trial's clip; a segment must not pass it."""
    C, T, D = state.shape
    H = controls_ahead
    seq = np.concatenate((start_state[None], state), 0)
    ok_state = np.concatenate((np.ones((1, T), bool), available & np.isfinite(state).all(-1)), 0)
    prev = np.concatenate((np.zeros((1, T, action.shape[-1]), action.dtype), action[:-1]), 0)
    trial, first = [], []
    for t in range(0, C - H + 1, stride):
        ok = ok_state[t:t + H + 1].all(0)
        if H > 1:
            ok &= ~done[t:t + H - 1].any(0)
        if last_control is not None:
            ok &= t + H - 1 <= last_control
        i = np.nonzero(ok)[0]
        trial.append(i)
        first.append(np.full(len(i), t))
    return np.concatenate(trial).astype(np.int64), np.concatenate(first).astype(np.int64), seq, prev


def in_get_up_windows(trial, first, per_trial):
    """[W] bool: the segment's first control lies in one of its trial's get-up windows (``per_trial``: per trial the
    window list of groups.get_up_windows)."""
    out = np.zeros(len(trial), bool)
    for i, ws in enumerate(per_trial):
        sel = trial == i
        for w in ws:
            c0, c1 = w["window_controls"]
            out[sel] |= (first[sel] >= c0) & (first[sel] <= c1)
    return out


def low_posture(s):
    """Low-posture states [..., D] (flattened state: root position 0:3, WXYZ quaternion 3:7): pelvis below 0.50 m or the z
    component of the pelvis up axis in the world frame, 1 - 2 (x^2 + y^2), below 0.5."""
    up_z = 1.0 - 2.0 * (s[..., 4] ** 2 + s[..., 5] ** 2)
    return (s[..., 2] < LOW_HEIGHT) | (up_z < LOW_UP_Z)


@torch.no_grad()
def open_loop(model, trial, first, seq, action, prev, xi_n, controls_ahead, batch=8192, scale=None):
    """Errors [W, H] of the world model stepped open loop over the segments (``segments``): from seq[t] with
    the actions and previous actions of controls t .. t+H-1, against seq[t+1 .. t+H]; and the error of each state
    group (STATE_GROUPS: its components' mean) after the first and after the last control, [W, 6] each."""
    out, first_groups, last_groups = [], [], []
    J = action.shape[-1]
    bounds = np.cumsum([0] + [width(J) for _, width in STATE_GROUPS])
    by_group = lambda c: torch.stack([c[:, a:b].mean(-1) for a, b in zip(bounds[:-1], bounds[1:])], -1)  # noqa: E731
    for b in range(0, len(trial), batch):
        tr, t0 = trial[b:b + batch], first[b:b + batch]
        S = unflatten_state(torch.as_tensor(seq[t0, tr]))
        x = xi_n.expand(len(tr), -1)
        errs = []
        for j in range(controls_ahead):
            S = model.step(S, torch.as_tensor(action[t0 + j, tr]), torch.as_tensor(prev[t0 + j, tr]), x)
            c = world_model_error_components(model, S, unflatten_state(torch.as_tensor(seq[t0 + j + 1, tr])), scale)
            errs.append(c.mean(-1))
            if j == 0:
                first_groups.append(by_group(c))
            if j == controls_ahead - 1:
                last_groups.append(by_group(c))
        out.append(torch.stack(errs, -1))
    if not out:
        z = np.zeros((0, len(STATE_GROUPS)), np.float32)
        return np.zeros((0, controls_ahead), np.float32), z, z
    return torch.cat(out).numpy(), torch.cat(first_groups).numpy(), torch.cat(last_groups).numpy()


def summary(e, mask, first_groups=None, last_groups=None):
    """Count, mean and median of the errors after the first and after the last control of the segments, the mean over
    all their controls, and with the state groups' errors their means after the first and after the last control."""
    sel = e[mask]
    H = e.shape[1]
    if not len(sel):
        return dict(segments=0)
    last = f"error_after_{H}_controls"
    out = {"segments": int(len(sel)), "error_after_1_control_mean": float(sel[:, 0].mean()),
           f"{last}_mean": float(sel[:, -1].mean()), "error_over_all_controls_mean": float(sel.mean()),
           "error_after_1_control_median": float(np.median(sel[:, 0])), f"{last}_median": float(np.median(sel[:, -1]))}
    if first_groups is not None:
        out["state_groups"] = {name: {"error_after_1_control_mean": float(first_groups[mask][:, i].mean()),
                                      f"{last}_mean": float(last_groups[mask][:, i].mean())}
                               for i, (name, _) in enumerate(STATE_GROUPS)}
    return out


def ratios(groups, controls_ahead):
    """Mean error of the segments from low-posture states over that of the segments from upright-posture states,
    after 1 and after H controls."""
    lo, up = groups.get("low_posture", {}), groups.get("upright_posture", {})
    if not lo.get("segments") or not up.get("segments"):
        return None
    keys = ("error_after_1_control_mean", f"error_after_{controls_ahead}_controls_mean")
    return {k.removesuffix("_mean"): lo[k] / up[k] for k in keys}


class Contacts:
    """Non-foot floor contact of recorded states by MuJoCo (C) collision on mjlab's compiled model of the robot."""

    def __init__(self, robot):
        import mujoco
        from mjlab.scene import Scene

        from blinkmimic2.env import make_scene_cfg

        self.mj = mujoco
        self.m = Scene(make_scene_cfg(robot, 1), device="cpu").compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        names = [m.body(b).name for b in range(m.nbody)]
        robot_bodies = {b for b in range(m.nbody) if names[b].startswith("robot/")}
        feet = {names.index("robot/" + f) for f in FEET}
        self.floor = np.array([m.geom_bodyid[g] not in robot_bodies for g in range(m.ngeom)])
        self.nonfoot = np.array([m.geom_bodyid[g] in robot_bodies and m.geom_bodyid[g] not in feet for g in range(m.ngeom)])
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free) == 1
        self.free_adr = int(m.jnt_qposadr[free[0]])
        self.q_adr = np.array([int(m.jnt_qposadr[m.joint("robot/" + n).id]) for n in robot.joint_names])

    def __call__(self, states):
        """[W] bool: non-foot floor contact of the flattened states [W, D]."""
        out = np.zeros(len(states), bool)
        J = len(self.q_adr)
        for i, s in enumerate(states):
            self.d.qpos[self.free_adr:self.free_adr + 7] = s[:7]
            self.d.qpos[self.q_adr] = s[13:13 + J]
            self.mj.mj_kinematics(self.m, self.d)
            self.mj.mj_collision(self.m, self.d)
            for c in self.d.contact[: self.d.ncon]:
                g1, g2 = c.geom1, c.geom2
                if c.dist < 0 and ((self.floor[g1] and self.nonfoot[g2]) or (self.floor[g2] and self.nonfoot[g1])):
                    out[i] = True
                    break
        return out


def load_world_model(path):
    """The world model of a checkpoint (``wm``), built as train.py builds it for the checkpoint's ``args`` (with the
    trainer's kinematics on the CPU for its link inputs, if any), its normalizers loaded: (model, checkpoint, robot,
    XiLayout)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "wm" not in ck:
        raise ValueError(f"{path} holds no world model (keys {sorted(ck)})")
    robot = get_robot(ck["args"]["robot"])
    layout = XiLayout(robot, "cpu")
    model = build_world_model(robot, layout.dim, ck["args"], Kinematics(robot, "cpu"))
    model.load_state_dict(ck["wm"])
    model.eval()
    return model, ck, robot, layout


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open-loop error of a training's world model on recorded segments.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt", default="last", help="'last' (last.pt) or the iteration of a ckpt_NNNNNN.pt that holds the world model")
    parser.add_argument("--scene", required=True, help="scene.py output (clips) for the first-row trials")
    parser.add_argument("--motion_dir", required=True, help="the 40 LAFAN1 G1 CSVs (reference rows of the get-ups)")
    parser.add_argument("--first_rows", nargs="*", default=[], help="tools/rollouts.py records of first-row trials")
    parser.add_argument("--placed", nargs="*", default=[], help="tools/rollouts.py records of placed starts")
    parser.add_argument("--controls_ahead", type=int, default=8, help="controls of a segment")
    parser.add_argument("--stride", type=int, default=1, help="segments start at every STRIDE-th control")
    parser.add_argument("--error_scale_checkpoint", default=None,
                        help="a checkpoint whose world model's error scale (max(inc_norm.std(), lower bound)) divides "
                             "the error in place of the read model's own, so that several models' errors on one "
                             "record are in one unit")
    parser.add_argument("--contacts", action="store_true", help="also split by non-foot floor contact (MuJoCo C)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error(f"{args.out} exists")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "recovery_pair"))
    import groups as G

    rotations.ELEMENTWISE = True  # as train.py
    networks.AMP_DTYPE = None
    path = os.path.join(args.run_dir, "last.pt" if args.ckpt == "last" else f"ckpt_{int(args.ckpt):06d}.pt")
    model, ck, robot, layout = load_world_model(path)
    a_ = ck["args"]
    scale = None
    if args.error_scale_checkpoint:
        scale = world_model_error_scale(load_world_model(args.error_scale_checkpoint)[0])
    norms = {n: dict(count=float(getattr(model, n).count), mean_abs_max=float(getattr(model, n).mean.abs().max()))
             for n in ("in_norm", "out_norm", "inc_norm")}
    xi_n = layout.normalize(evaluation_xi(layout)).unsqueeze(0)
    scene = json.load(open(os.path.join(args.scene, "scene.json")))
    clips = scene["clips"]
    reference, names, firsts, lasts = G.reference_rows(args.motion_dir, scene["robot"])
    if (names, firsts) != ([c["clip"] for c in clips], [c["first_row"] for c in clips]):
        raise ValueError("the library is not the scene's")
    low_ref = G.low_rows(*reference)
    gwin = {c["first_row"]: G.get_up_windows(low_ref[c["first_row"]:c["last_row"] + 1], c["first_row"], c["last_row"])
            for c in clips}
    contacts = Contacts(robot) if args.contacts else None
    H = args.controls_ahead
    keys = ("start_state", "state", "action", "available", "done", "command_row")
    result = dict(checkpoint=os.path.abspath(path), iteration=ck["iteration"], controls_ahead=H, xi=evaluation_xi(layout).tolist(),
                  xi_slices={k: [v.start, v.stop] for k, v in layout.slices.items()}, normalizers=norms,
                  hidden_widths=a_["wm_hidden"], stride=args.stride,
                  error_scale_checkpoint=os.path.abspath(args.error_scale_checkpoint) if args.error_scale_checkpoint else None, link_inputs=list(model.link_bodies), use_prev_action=model.use_prev_action, structured=model.structured,
                  contact_modes=model.contact_modes, precision="float32 on the CPU", records={})
    pooled = {}
    for kind, dirs in (("first_rows", args.first_rows), ("placed", args.placed)):
        for d in dirs:
            with np.load(os.path.join(d, "rollouts.npz")) as z:
                rec = {k: z[k] for k in keys}
            last_control = None
            if kind == "first_rows":
                by_first = {c["first_row"]: c for c in clips}
                last_control = np.array([by_first[int(r)]["last_row"] - int(r) - 1 for r in rec["command_row"]])
            trial, first, seq, prev = segments(rec["start_state"], rec["state"], rec["action"], rec["available"],
                                              rec["done"], H, last_control, args.stride)
            e, first_groups, last_groups = open_loop(model, trial, first, seq, rec["action"], prev, xi_n, H, scale=scale)
            s0 = seq[first, trial]
            low = low_posture(s0)
            masks = {"all": np.ones(len(e), bool), "upright_posture": ~low, "low_posture": low}
            if kind == "first_rows":
                ing = in_get_up_windows(trial, first, [gwin[int(r)] for r in rec["command_row"]])
                masks["reference_get_up_windows"] = ing
                masks["reference_get_up_windows_low_posture"] = ing & low
                masks["reference_get_up_windows_upright_posture"] = ing & ~low
                masks["outside_reference_get_up_windows"] = ~ing
            if contacts is not None:
                nf = contacts(s0)
                masks["non_foot_contact"] = nf
                masks["no_non_foot_contact"] = ~nf
                masks["upright_posture_without_non_foot_contact"] = ~low & ~nf
            groups = {k: summary(e, m, first_groups, last_groups) for k, m in masks.items()}
            result["records"][os.path.abspath(d)] = dict(kind=kind, trials=int(rec["state"].shape[1]), groups=groups,
                                                         low_posture_over_upright_posture=ratios(groups, H))
            for k, m in masks.items():
                for key in (f"{kind}:{k}", f"all:{k}"):
                    pooled.setdefault(key, []).append((e[m], first_groups[m], last_groups[m]))
            print(kind, d, json.dumps({k: list(v.values())[:3] for k, v in groups.items()}), flush=True)
    result["pooled"] = {}
    for k, parts in pooled.items():
        e, fg, lg = (np.concatenate([p[i] for p in parts]) for i in range(3))
        result["pooled"][k] = summary(e, np.ones(len(e), bool), fg, lg)
    for kind in ("first_rows", "placed", "all"):
        g = {k.split(":", 1)[1]: v for k, v in result["pooled"].items() if k.startswith(kind + ":")}
        result[f"low_posture_over_upright_posture_{kind}"] = ratios(g, H)
    with open(args.out, "x") as f:
        f.write(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in result if k.startswith("low_posture_over")}))


if __name__ == "__main__":
    main()
