"""Where the world model is inaccurate: the trainer's normalized one-control error of world models (class
``WorldModel`` in networks.py) and of the rigid-body integrator, split by what acts during the control (CPU).

Recorded controls: tools/contact_wrench_recording.py --store recordings. Each control is classed from its four physics
substeps (``contact_classes``): a force is "present" when its largest absolute entry over the control exceeds
``--threshold`` (N or N m).

  joint_limit          the joint-limit generalized force is present
  robot_robot          a robot-robot contact wrench is present on some body
  floor_non_foot       a floor contact wrench is present on a body other than the two ankle-roll links (the feet)
  feet_only_floor      a floor contact wrench is present on the feet and on no other body
  no_contact           no floor and no robot-robot contact wrench (a joint-limit force may be present)

A control can be in joint_limit, robot_robot and floor_non_foot at once; the combinations are reported as keys such
as ``joint_limit+robot_robot+floor_non_foot``, ``feet_only_floor`` (no joint limit, no robot-robot contact) or
``joint_limit+feet_only_floor``.

Posture class from the state at the start of the control: ``low`` if the pelvis is below 0.50 m or the root's up axis
has z below 0.5, else ``upright``. Per recording set, posture class, contact class and model: the share of controls,
the mean, median and 95th percentile of the per-control error and the share of the posture class's summed error that
the contact class carries.

Models: the world model of ``last.pt`` of each run given (built from the checkpoint's ``args.wm_hidden``), and two
rows of the rigid-body integrator (blinkmimic2.rigid_body; DynaMimic's physics-informed world model, arXiv 2611.32944,
with its contact wrenches taken from the simulator): the net contact wrenches of ``--links`` at two 10 ms
integration timesteps, without and with the recorded joint-limit generalized force (the rows
``net_wrenches_candidate_links_2x10ms`` and ``net_wrenches_candidate_links_and_joint_limits_2x10ms`` of
tools/rigid_body_step_comparison.py, computed the same way). Each model's error uses that run's normalizer; the
integrator rows are reported with each run's normalizer.

  python tools/world_model_error_by_contact_class.py --recordings SET_A:a1.npz,a2.npz SET_B:b1.npz \\
      --runs NAME1:RUN1 NAME2:RUN2 ... --links L1 ... --out classes.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blinkmimic2.dr import XiLayout  # noqa: E402
from blinkmimic2.networks import WorldModel  # noqa: E402
from blinkmimic2.rigid_body import RigidBody, from_mujoco, to_trainer, trainer_model  # noqa: E402
from blinkmimic2.robot import get_robot  # noqa: E402
from rigid_body_step_comparison import err_floor, load  # noqa: E402

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
FLAGS = ("joint_limit", "robot_robot", "floor_non_foot", "feet_only_floor", "no_contact")
INTEGRATOR_ROWS = ("net_wrenches_candidate_links_2x10ms", "net_wrenches_candidate_links_and_joint_limits_2x10ms")


def contact_classes(wrench_floor, wrench_self, tau_limit, feet, threshold=1e-6):
    """Boolean flags per control (N,) from per-physics-substep loads: wrench_floor and wrench_self (N, k, nb, 6),
    tau_limit (N, k, nv); ``feet`` the body indices of the feet. Returns {flag: (N,) bool} and the combination key
    of every control (N,) as strings."""
    wf = np.abs(np.asarray(wrench_floor)).max(axis=(1, 3))  # (N, nb)
    ws = np.abs(np.asarray(wrench_self)).max(axis=(1, 2, 3))
    tl = np.abs(np.asarray(tau_limit)).max(axis=(1, 2))
    on = wf > threshold
    foot = np.zeros(on.shape[1], dtype=bool)
    foot[list(feet)] = True
    f = {"joint_limit": tl > threshold, "robot_robot": ws > threshold, "floor_non_foot": on[:, ~foot].any(1)}
    f["feet_only_floor"] = on[:, foot].any(1) & ~f["floor_non_foot"]
    f["no_contact"] = ~on.any(1) & ~f["robot_robot"]
    keys = []
    for i in range(len(tl)):
        parts = [k for k in ("joint_limit", "robot_robot", "floor_non_foot", "feet_only_floor") if f[k][i]]
        if f["no_contact"][i]:
            parts.append("no_contact")
        keys.append("+".join(parts))
    return f, np.array(keys)


def integrator_rows(rb, d, sel, links, com, chunk=1024):
    """Next states (trainer tuple) of the two integrator rows for the selected recorded controls."""
    out = {k: [] for k in INTEGRATOR_ROWS}
    half = lambda x: 0.5 * (x[:, 0::2] + x[:, 1::2])  # noqa: E731
    for s in range(0, len(sel), chunk):
        e = slice(s, min(s + chunk, len(sel)))
        ix = sel[e]
        qpos, qvel = torch.tensor(d["qpos"][ix]), torch.tensor(d["qvel"][ix])
        target = torch.tensor(d["target"][ix], dtype=torch.float64)
        Wn = torch.tensor(d["wrench_floor"][ix], dtype=torch.float64) + torch.tensor(d["wrench_self"][ix], dtype=torch.float64)
        TL = torch.tensor(d["tau_limit"][ix], dtype=torch.float64)
        c = com[e]
        p0, R0, q0, nu0 = from_mujoco(qpos[:, 0], qvel[:, 0])
        W2, T2 = half(Wn[:, :, links]), half(TL)
        for name, with_limits in zip(INTEGRATOR_ROWS, (False, True)):
            p, R, q, nu = p0, R0, q0, nu0
            for k in range(2):
                p, R, q, nu = rb.substep(p, R, q, nu, target, 0.01, tau_ext=T2[:, k] if with_limits else None, links=links,
                                         wrench=W2[:, k], com_offset=c)  # fmt: skip
            out[name].append(to_trainer(p, R, q, nu))
        print(f"integrator {e.stop} / {len(sel)}", flush=True)
    return {k: tuple(torch.cat([c[i] for c in v]) for i in range(6)) for k, v in out.items()}


def stats(err, groups, total):
    """mean, median, p95 of err over each boolean mask in groups and the mask's share of ``total``."""
    out = {}
    for k, m in groups.items():
        e = err[m]
        if len(e) == 0:
            continue
        out[k] = {"controls": int(len(e)), "mean": float(e.mean()), "median": float(np.median(e)),
                  "p95": float(np.quantile(e, 0.95)), "share_of_summed_error": float(e.sum() / total)}  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", nargs="+", required=True, help="NAME:file1.npz,file2.npz per recording set")
    ap.add_argument("--runs", nargs="+", required=True, help="NAME:run_directory per world model (last.pt)")
    ap.add_argument("--links", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=1e-6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    robot = get_robot("mjlab_g1")
    J = robot.num_joints
    layout = XiLayout(robot, "cpu")
    rb = RigidBody(trainer_model(robot), robot.joint_names, dtype=torch.float64)
    links = rb.index(a.links)
    feet = rb.index(FEET)
    models = []
    for spec in a.runs:
        name, run = spec.split(":", 1)
        ck = torch.load(os.path.join(run, "last.pt"), map_location="cpu", weights_only=False)
        hidden = tuple(int(x) for x in ck["args"]["wm_hidden"].split(","))
        wm = WorldModel(J, layout.dim, use_prev_action=robot.has_action_delay, hidden=hidden, structured=True, dt=0.02)
        wm.load_state_dict(ck["wm"])
        wm.eval()
        models.append((name, run, hidden, wm))
    report = {"threshold": a.threshold, "links": a.links, "feet": list(FEET),
              "runs": {n: {"run": r, "hidden_widths": h} for n, r, h, _ in models}, "sets": {}}  # fmt: skip
    for spec in a.recordings:
        set_name, files = spec.split(":", 1)
        d = load(files.split(","))
        if d["body_names"] != rb.names:
            raise ValueError("body order of the recording differs from the model's")
        qpos0, qvel0 = torch.tensor(d["qpos"][:, 0]), torch.tensor(d["qvel"][:, 0])
        low = ((qpos0[:, 2] < 0.5) | (from_mujoco(qpos0, qvel0)[1][:, 2, 2] < 0.5)).numpy()
        flags, keys = contact_classes(d["wrench_floor"], d["wrench_self"], d["tau_limit"], feet, a.threshold)
        sel = np.arange(len(low))
        xi = torch.tensor(d["xi"], dtype=torch.float64)
        S0 = to_trainer(*from_mujoco(torch.tensor(d["qpos"][:, 0]), torch.tensor(d["qvel"][:, 0])))
        S1 = to_trainer(*from_mujoco(torch.tensor(d["qpos"][:, 4]), torch.tensor(d["qvel"][:, 4])))
        rows = integrator_rows(rb, d, sel, links, xi[:, layout.slices["com"]])
        out = {"files": files.split(","), "postures": {}}
        for posture, pm in (("upright", ~low), ("low", low)):
            groups = {k: v & pm for k, v in flags.items()}
            combos = {k: (keys == k) & pm for k in sorted(set(keys[pm]))}
            n = int(pm.sum())
            entry = {"controls": n, "class_share": {k: float(v.sum() / n) for k, v in groups.items()},
                     "combination_share": {k: float(v.sum() / n) for k, v in combos.items()}, "models": {}}  # fmt: skip
            for name, _, _, wm in models:
                std = torch.maximum(wm.inc_norm.std().double(), err_floor(J))
                with torch.no_grad():
                    S_hat = wm.step(tuple(x.float() for x in S0), torch.tensor(d["action"]).float(),
                                    torch.tensor(d["a_prev"]).float(), layout.normalize(xi.float()))  # fmt: skip
                preds = {"world_model": S_hat} | rows
                for row, Sp in preds.items():
                    e = ((WorldModel.state_error(tuple(x.double() for x in Sp), S1) / std) ** 2).mean(-1).numpy()
                    tot = e[pm].sum()
                    entry["models"].setdefault(name, {})[row] = {
                        "all": stats(e, {"all": pm}, tot)["all"], "classes": stats(e, groups, tot),
                        "combinations": stats(e, combos, tot)}
            out["postures"][posture] = entry
            print(set_name, posture, n, json.dumps(entry["class_share"]), flush=True)
        report["sets"][set_name] = out
    json.dump(report, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
