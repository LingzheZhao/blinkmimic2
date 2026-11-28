"""The rigid-body step comparison: the simulator's recorded link wrenches fed into one rigid-body step of DynaMimic's
physics-informed world model (arXiv 2611.32944; the rigid-body integrator of blinkmimic2.rigid_body), against the
trained world model of the trainer (class ``WorldModel`` in networks.py), one control, in the trainer's normalized state
error (CPU).

Recorded controls: tools/contact_wrench_recording.py --store recordings (MuJoCo C, the trainer's model). A recorded
control holds the state at the control boundary, the action, the joint targets, xi, per physics substep the floor and
robot-robot contact wrenches on every body and the constraint forces, and the state after the control's 4 physics
substeps (the target). Posture class from the state at the start: ``low`` if the pelvis is below 0.50 m or the root's
up axis has z below 0.5, else ``upright``.

Error: the per-control value of train.py's ``_wm_err``,
``mean_D ((WorldModel.state_error(S_hat, S1) / max(inc_norm.std, ERR_FLOOR))^2)``, with the normalizer ``inc_norm`` of
each checkpoint, averaged over the recorded controls of a posture class (the mean, as in the training loss). The
world model and its normalizers are read from the checkpoint keys ``wm`` (state dict) and ``args.wm_hidden``
(hidden widths), which train.py writes.

Rows. The rigid-body step runs in float64, with Jacobians, mass matrix and bias force at its own state, as a force
predictor's prediction would meet them. A link's "net" wrench is its floor plus robot-robot contact wrench; its "floor"
wrench is the floor part alone, the part DynaMimic's Eq. 4 decoder can represent (F_z >= 0 along the world vertical, a
friction cone about it). 4x5ms: four 5 ms integration timesteps, one per physics substep of the simulator, each with
that physics substep's wrenches; 2x10ms: two 10 ms integration timesteps, each fed the mean of its two 5 ms wrenches
(DynaMimic's form).

  world_model                                   the checkpoint's WorldModel.step (float32 on the CPU; the
                                                           training ran its MLP in bf16)
  all_constraint_forces_4x5ms                              MuJoCo's qfrc_constraint of each physics substep (equal to
                                                           the simulator up to rounding)
  net_wrenches_all_bodies_4x5ms                            every robot body's net wrench: everything but the
                                                           joint-limit forces
  net_wrenches_candidate_links_{4x5ms,2x10ms}              the candidate links' net wrenches, no residual force
                                                           (the pass-rule rows)
  net_wrenches_candidate_links_and_joint_limits_{..}       the same plus the recorded joint-limit generalized force
                                                           (what an exact joint-limit model would add; averaged per
                                                           10 ms for 2x10ms)
  floor_wrenches_candidate_links_{4x5ms,2x10ms}            the candidate links' floor wrenches only, no residual force
  net_wrenches_candidate_links_and_exact_leftover_{..}     the net-wrench rows plus tau_res = their exact leftover
  floor_wrenches_candidate_links_and_exact_leftover_2x10ms the floor-wrench row plus its exact leftover

The exact leftover is the mean over the control's four physics substeps of qfrc_constraint - J^T W at the simulator's
physics-substep states: one generalized force per control, what DynaMimic's residual generalized force would have to
supply. It is
computed from the target and is a diagnostic, not a pass-rule row.

Pass rule (the declared rule of studies/rigid_body_integrator/README.md on whether a rigid-body integrator, given the
simulator's recorded contact wrenches, reproduces the simulator's next state better than each trained world model):
``net_wrenches_candidate_links_2x10ms`` (DynaMimic's form) and ``net_wrenches_candidate_links_4x5ms`` (one integration
timestep per physics substep of the simulator) are each a pass-rule row. A row passes when
its mean error is below the world model's on the upright and on the low recorded controls, for each
checkpoint given (each with its own world model and normalizer). Reported per row.

  python tools/rigid_body_step_comparison.py --recordings a.npz b.npz --runs RUN1 RUN2 ... --links L1 L2 ... \
      --out comparison.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from blinkmimic2.dr import XiLayout
from blinkmimic2.networks import WorldModel
from blinkmimic2.rigid_body import RigidBody, from_mujoco, to_trainer, trainer_model
from blinkmimic2.robot import get_robot

GROUPS = ("pos", "rot", "lin_vel", "ang_vel", "q", "qd")
PASS_RULE_ROWS = ("net_wrenches_candidate_links_2x10ms", "net_wrenches_candidate_links_4x5ms")
KEYS = ("qpos", "qvel", "action", "a_prev", "target", "xi", "wrench_floor", "wrench_self", "qfrc_constraint", "tau_self",
        "tau_limit", "row", "trial", "control", "identity_max")  # fmt: skip


def err_floor(J):
    return torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                      torch.full((J,), 3e-3), torch.full((J,), 5e-2))).double()  # fmt: skip


def load(paths):
    parts = [np.load(p) for p in paths]
    out = {k: np.concatenate([z[k] for z in parts]) for k in KEYS}
    out["body_names"] = list(parts[0]["body_names"])
    out["source"] = np.concatenate([np.full(len(z["trial"]), i) for i, z in enumerate(parts)])
    return out


def leftover(rb, qpos, qvel, com, QC, links, W):
    """Mean over the physics substeps of qfrc_constraint - J^T W(links) at the simulator's physics-substep states,
    (N, nv)."""
    out = 0
    for k in range(4):
        kin = rb.kinematics(*from_mujoco(qpos[:, k], qvel[:, k])[:3], com)
        Jp, Jr = rb.jacobians(kin)
        out = out + (QC[:, k] - rb.wrench_force(Jp, Jr, links, W[:, k][:, links]))
    return out / 4


def predict(rb, ch, links):
    """Next states (trainer tuple) of every rigid-body row for one chunk of recorded controls."""
    qpos, qvel, target, com = ch["qpos"], ch["qvel"], ch["target"], ch["com"]
    Wn, Wf, QC, TL = ch["Wf"] + ch["Ws"], ch["Wf"], ch["QC"], ch["TL"]
    every = list(range(rb.nb))
    p0, R0, q0, nu0 = from_mujoco(qpos[:, 0], qvel[:, 0])
    half = lambda x: 0.5 * (x[:, 0::2] + x[:, 1::2])  # noqa: E731
    res_net = leftover(rb, qpos, qvel, com, QC, links, Wn)
    res_floor = leftover(rb, qpos, qvel, com, QC, links, Wf)

    def run(n, dt, L=None, W=None, tau=None, tau_k=None):
        p, R, q, nu = p0, R0, q0, nu0
        for k in range(n):
            t = tau if tau_k is None else (tau_k[:, k] if tau is None else tau + tau_k[:, k])
            p, R, q, nu = rb.substep(p, R, q, nu, target, dt, tau_ext=t, links=L, wrench=None if W is None else W[:, k],
                                     com_offset=com)  # fmt: skip
        return to_trainer(p, R, q, nu)

    d5, d10 = 0.005, 0.01
    return {
        "all_constraint_forces_4x5ms": run(4, d5, tau_k=QC),
        "net_wrenches_all_bodies_4x5ms": run(4, d5, every, Wn),
        "net_wrenches_candidate_links_4x5ms": run(4, d5, links, Wn[:, :, links]),
        "net_wrenches_candidate_links_2x10ms": run(2, d10, links, half(Wn[:, :, links])),
        "net_wrenches_candidate_links_and_joint_limits_4x5ms": run(4, d5, links, Wn[:, :, links], tau_k=TL),
        "net_wrenches_candidate_links_and_joint_limits_2x10ms": run(2, d10, links, half(Wn[:, :, links]), tau_k=half(TL)),
        "floor_wrenches_candidate_links_4x5ms": run(4, d5, links, Wf[:, :, links]),
        "floor_wrenches_candidate_links_2x10ms": run(2, d10, links, half(Wf[:, :, links])),
        "net_wrenches_candidate_links_and_exact_leftover_4x5ms": run(4, d5, links, Wn[:, :, links], tau=res_net),
        "net_wrenches_candidate_links_and_exact_leftover_2x10ms": run(2, d10, links, half(Wn[:, :, links]), tau=res_net),
        "floor_wrenches_candidate_links_and_exact_leftover_2x10ms": run(2, d10, links, half(Wf[:, :, links]), tau=res_floor),
    }, {"net": res_net, "floor": res_floor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories; the world model of last.pt")
    ap.add_argument("--links", nargs="+", required=True)
    ap.add_argument("--max_per_class", type=int, default=0, help="subsample each posture class to this many controls (0: all)")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    robot = get_robot("mjlab_g1")
    J = robot.num_joints
    layout = XiLayout(robot, "cpu")
    rb = RigidBody(trainer_model(robot), robot.joint_names, dtype=torch.float64)
    links = rb.index(a.links)
    d = load(a.recordings)
    if d["body_names"] != rb.names:
        raise ValueError("body order of the recording differs from the model's")
    qpos_all, qvel_all = torch.tensor(d["qpos"]), torch.tensor(d["qvel"])
    low = (qpos_all[:, 0, 2] < 0.5) | (from_mujoco(qpos_all[:, 0], qvel_all[:, 0])[1][:, 2, 2] < 0.5)
    idx = {"upright": torch.nonzero(~low).squeeze(-1), "low": torch.nonzero(low).squeeze(-1)}
    if a.max_per_class:
        g = torch.Generator().manual_seed(a.seed)
        idx = {k: v[torch.randperm(len(v), generator=g)[: a.max_per_class]] for k, v in idx.items()}
    sel = torch.cat((idx["upright"], idx["low"])).numpy()
    cls = np.array(["upright"] * len(idx["upright"]) + ["low"] * len(idx["low"]))
    masks = {k: torch.tensor(cls == k) for k in ("upright", "low") if (cls == k).any()}  # posture classes present
    T64 = lambda k: torch.tensor(d[k][sel], dtype=torch.float64)  # noqa: E731
    qpos, qvel, target, xi = T64("qpos"), T64("qvel"), T64("target"), T64("xi")
    com = xi[:, layout.slices["com"]]
    N = len(sel)
    preds, res = {}, {"net": [], "floor": []}
    for s in range(0, N, a.chunk):
        e = slice(s, min(s + a.chunk, N))
        ch = {"qpos": qpos[e], "qvel": qvel[e], "target": target[e], "com": com[e],
              "Wf": torch.tensor(d["wrench_floor"][sel[e]], dtype=torch.float64),
              "Ws": torch.tensor(d["wrench_self"][sel[e]], dtype=torch.float64),
              "QC": torch.tensor(d["qfrc_constraint"][sel[e]], dtype=torch.float64),
              "TL": torch.tensor(d["tau_limit"][sel[e]], dtype=torch.float64)}  # fmt: skip
        P, Rz = predict(rb, ch, links)
        for k, v in P.items():
            preds.setdefault(k, []).append(v)
        for k in res:
            res[k].append(Rz[k])
        print(f"physics {e.stop} / {N}", flush=True)
    preds = {k: tuple(torch.cat([c[i] for c in v]) for i in range(6)) for k, v in preds.items()}
    res = {k: torch.cat(v) for k, v in res.items()}
    S0 = to_trainer(*from_mujoco(qpos[:, 0], qvel[:, 0]))
    S1 = to_trainer(*from_mujoco(qpos[:, 4], qvel[:, 4]))

    # what DynaMimic's residual generalized force has to absorb, RMS over recorded controls (and physics substeps) of each generalized-force part
    TS, TL = T64("tau_self"), T64("tau_limit")

    def rms(x, m):
        x = x[m]
        return {"root_force": float(x[..., :3].pow(2).mean().sqrt()), "root_torque": float(x[..., 3:6].pow(2).mean().sqrt()),
                "joints": float(x[..., 6:].pow(2).mean().sqrt()), "joints_max": float(x[..., 6:].abs().max()),
                "share_of_controls_nonzero": float((x.flatten(1).abs().amax(-1) > 1e-9).double().mean())}  # fmt: skip

    report = {"recorded_controls": {k: int(m.sum()) for k, m in masks.items()}, "links": a.links, "recordings": a.recordings,
              "identity_max": float(np.max(d["identity_max"])),
              "low_rule": "pelvis z < 0.5 m or R[2, 2] < 0.5 at the start of the control",
              "generalized_force_rms": {k: {"robot_robot_contacts": rms(TS, mk), "joint_limits": rms(TL, mk),
                                            "exact_leftover_net": rms(res["net"], mk), "exact_leftover_floor": rms(res["floor"], mk)}
                                        for k, mk in masks.items()},
              "checkpoints": {}}  # fmt: skip
    sl = np.cumsum([0, 3, 3, 3, 3, J, J])
    for run in a.runs:
        ck = torch.load(os.path.join(run, "last.pt"), map_location="cpu", weights_only=False)
        ar = ck["args"]
        hidden = tuple(int(x) for x in ar["wm_hidden"].split(","))
        world_model = WorldModel(J, layout.dim, use_prev_action=robot.has_action_delay, hidden=hidden, structured=True, dt=0.02)
        world_model.load_state_dict(ck["wm"])
        world_model.eval()
        std = torch.maximum(world_model.inc_norm.std().double(), err_floor(J))
        with torch.no_grad():
            S_hat = world_model.step(tuple(x.float() for x in S0), T64("action").float(), T64("a_prev").float(), layout.normalize(xi.float()))
        rows = {"world_model": S_hat} | preds
        E = {n: (WorldModel.state_error(tuple(x.double() for x in v), S1) / std) ** 2 for n, v in rows.items()}
        tab = {}
        for k, mk in masks.items():
            tab[k] = {}
            base = E["world_model"][mk].mean(-1)
            for n, e in E.items():
                em = e[mk]
                per = em.mean(-1)
                tab[k][n] = {"mean": float(per.mean()), "median": float(per.median()), "p90": float(per.quantile(0.9)),
                             "groups": {g: float(em[:, sl[i]:sl[i + 1]].mean()) for i, g in enumerate(GROUPS)},
                             "qd_per_joint": em[:, sl[5]:sl[6]].mean(0).tolist(),
                             "share_of_controls_below_world_model": float((per < base).double().mean())}  # fmt: skip
        passes = {row: all(tab[k][row]["mean"] < tab[k]["world_model"]["mean"] for k in masks) for row in PASS_RULE_ROWS}
        report["checkpoints"][run] = {"iteration": ck.get("iteration"), "hidden_widths": hidden, "table": tab, "passes_rule": passes}
        print(run, json.dumps({k: {n: round(v["mean"], 5) for n, v in t.items()} for k, t in tab.items()}), passes, flush=True)
    report["passes_rule"] = {row: all(c["passes_rule"][row] for c in report["checkpoints"].values()) for row in PASS_RULE_ROWS}
    json.dump(report, open(a.out, "w"), indent=1)
    print("passes rule", report["passes_rule"])


if __name__ == "__main__":
    main()
