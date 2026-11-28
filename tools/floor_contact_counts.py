"""Which robot bodies touch the floor: reference playback of every library row under the trainer's collision model,
and the simulated floor forces of tools/contact_wrench_recording.py recordings (CPU).

Playback: each row's reference pose (root position, rotation, joints clamped to their ranges) is put into the trainer's
MjModel (rigid_body.trainer_model) and MuJoCo's collision runs (mj_kinematics, mj_collision); a body touches the floor
in a row when one of its geoms has a floor contact with distance below 0 (penetration) or below 0.02 m (near).

Simulation: the per-clip, per-body sums that tools/contact_wrench_recording.py writes (floor-contact physics
substeps, summed floor force norm).

  python tools/floor_contact_counts.py --motion_dir DIR --stats recording1.npz recording2.npz ... --out contacts.json
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np
import torch

from blinkmimic2.kinematics import Kinematics
from blinkmimic2.motion import MotionLib
from blinkmimic2.rigid_body import trainer_model
from blinkmimic2.robot import get_robot
from blinkmimic2.rotations import mat_to_quat

GETUP = "fallAndGetUp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", required=True)
    ap.add_argument("--stats", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    robot = get_robot("mjlab_g1")
    m = trainer_model(robot)
    d = mujoco.MjData(m)
    bodies = [b for b in range(m.nbody) if m.body_jntnum[b] > 0]
    local = {b: i for i, b in enumerate(bodies)}
    names = [m.body(b).name.removeprefix("robot/") for b in bodies]
    floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] not in local}
    kin = Kinematics(robot, "cpu")
    motion = MotionLib.load(robot, kin, "cpu", a.motion_dir, "", None, "fp32")
    clips = list(motion.names)
    pen = np.zeros((len(clips), len(bodies)))
    near = np.zeros((len(clips), len(bodies)))
    rows = np.zeros(len(clips))
    low_rows = np.zeros(len(clips))
    # mj_collision reports the pairs within the geoms' margins: a 0.02 m margin on the robot geoms (collision only, no
    # dynamics run) lets one pass count both penetration (dist < 0) and near contact (dist < 0.02)
    m.geom_margin[:] = np.where(np.isin(np.arange(m.ngeom), list(floor)), 0.0, 0.02)
    for c in range(len(clips)):
        f = torch.arange(int(motion.clip_start[c]), int(motion.clip_end[c]) + 1)
        p, R, _, _, q, _ = motion.ref_state(f)
        q = torch.clamp(q, kin.q_lo, kin.q_hi)
        qpos = torch.cat((p, mat_to_quat(R), q), -1).double().numpy()
        low = (p[:, 2] < 0.5) | (R[:, 2, 2] < 0.5)
        rows[c], low_rows[c] = len(f), float(low.sum())
        for i in range(len(f)):
            d.qpos[:] = qpos[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            tp, tn = np.zeros(len(bodies), bool), np.zeros(len(bodies), bool)
            for k in range(d.ncon):
                ct = d.contact[k]
                g1, g2 = ct.geom[0], ct.geom[1]
                if (g1 in floor) == (g2 in floor):
                    continue
                b = local[m.geom_bodyid[g2 if g1 in floor else g1]]
                tn[b] |= ct.dist < 0.02
                tp[b] |= ct.dist < 0
            pen[c] += tp
            near[c] += tn
        print(f"{clips[c]}: {int(rows[c])} rows, {int(low_rows[c])} low", flush=True)
    getup = np.array([n.startswith(GETUP) for n in clips])
    out = {"bodies": names, "clips": clips, "rows": rows.tolist(), "low_rows": low_rows.tolist(),
           "playback_penetrating_rows": pen.tolist(), "playback_near_rows": near.tolist()}
    summary = {}
    for label, sel in (("getup", getup), ("other", ~getup), ("all", np.ones_like(getup))):
        summary[label] = {"rows": float(rows[sel].sum()),
                          "penetrating_share": dict(zip(names, (pen[sel].sum(0) / rows[sel].sum()).round(5).tolist())),
                          "near_share": dict(zip(names, (near[sel].sum(0) / rows[sel].sum()).round(5).tolist()))}  # fmt: skip
    if a.stats:
        n = np.zeros((len(clips), len(bodies)))
        fn = np.zeros((len(clips), len(bodies)))
        sub = np.zeros(len(clips))
        for path in a.stats:
            z = np.load(path)
            idx = [clips.index(c) for c in z["clip_names"]]
            if list(z["body_names"]) != names:
                raise ValueError(f"{path}: body order")
            n[idx] += z["stat_contact_substeps"]
            fn[idx] += z["stat_force_norm"]
            sub[idx] += z["stat_substeps"]
        out |= {"sim_contact_substeps": n.tolist(), "sim_force_norm": fn.tolist(), "sim_substeps": sub.tolist()}
        for label, sel in (("getup", getup), ("other", ~getup), ("all", np.ones_like(getup))):
            tot = fn[sel].sum()
            summary[label] |= {"sim_substeps": float(sub[sel].sum()),
                               "sim_contact_share": dict(zip(names, (n[sel].sum(0) / max(sub[sel].sum(), 1)).round(5).tolist())),
                               "sim_force_share": dict(zip(names, (fn[sel].sum(0) / max(tot, 1e-12)).round(6).tolist()))}  # fmt: skip
    out["summary"] = summary
    json.dump(out, open(a.out, "w"), indent=1)
    for label, s in summary.items():
        print(label, json.dumps(s))


if __name__ == "__main__":
    main()
