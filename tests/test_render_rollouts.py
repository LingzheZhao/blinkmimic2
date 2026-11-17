"""tools/render_rollouts.py on the CPU, with records written by rollouts.roll_out from a stand-in for the simulator over a
library of generated clips: the mapping from a record to MuJoCo poses, where the command pose is drawn, the frames of a
trial and where its video stops, the choice of trials, the refusals, and a small render when MuJoCo has an OpenGL
backend (MUJOCO_GL=egl), else the tool's refusal."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
TOOL = os.path.join(TOOLS, "render_rollouts.py")
sys.path[:0] = [HERE, TOOLS]
import render_rollouts as rr  # noqa: E402
import rollouts  # noqa: E402

from blinkmimic2.dr import DomainRandCfg, XiLayout  # noqa: E402
from blinkmimic2.env import placed_state  # noqa: E402
from blinkmimic2.kinematics import Kinematics  # noqa: E402
from blinkmimic2.mdp import MDP  # noqa: E402
from blinkmimic2.motion import CONTROL_DT, MotionLib  # noqa: E402
from blinkmimic2.robot import get_robot  # noqa: E402
from blinkmimic2.rotations import mat_to_quat  # noqa: E402

ROBOT = get_robot("mjlab_g1")
J, CONTROLS = ROBOT.num_joints, 8
# clips a_walk (rows 0-25) and b_run (rows 26-76), both along x at 0.5 m/s, 0.76 m high, turning and tilted
CLIPS = {"a_walk.csv": 16, "b_run.csv": 31}
# trials: 0 runs, placed with position, rotation and joint offsets; 1 body placed in b_run while the command starts at
# b_run's first row; 2 non-finite at control 3; 3 non-finite at control 0; 4 sinks 0.5 m at control 4, which the
# recipe's height term ends with a finite state
OFFSETS = {"position": [1.0, -1.0, 0.0], "rotation": [0.5, -0.5, 1.0], "velocity": [0.0] * 6,
           "joints": np.linspace(-1, 1, J).tolist()}  # fmt: skip
SCENE = {"trials": [{"command_row": 0, "reset_uniforms": OFFSETS}, {"command_row": 26, "body_row": 60},
                    {"command_row": 10}, {"command_row": 20}, {"command_row": 30}]}  # fmt: skip


def write_clip(path: str, rows: int):
    """A LAFAN1 G1 retarget CSV (30 fps: root position, root quaternion xyzw, 29 joint angles) whose root turns about
    the vertical at 0.6 rad/s with a fixed 0.15 rad roll (R = Rz(yaw) Rx(roll), so R[2, 2] stays cos 0.15) and whose
    joints each move with their own phase."""
    t = np.arange(rows)[:, None] / 30.0
    yaw, roll = 0.4 + 0.6 * t, 0.15 + 0 * t
    cz, sz, cx, sx = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(roll / 2), np.sin(roll / 2)
    root = np.concatenate((0.5 * t, 0 * t, 0 * t + 0.76, cz * sx, sz * sx, sz * cx, cz * cx), 1)
    joints = np.array(ROBOT.default_joint_pos_list())[None] + 0.1 * np.sin(t + 0.25 * np.arange(J)[None])
    np.savetxt(path, np.concatenate((root, joints), 1), delimiter=",")
NONFINITE_AT, SINK_AT = {2: 3, 3: 0}, {4: 4}


class StandIn:
    """Stand-in for rollouts.Simulator: places the body at its row's reference state as TrackingSim.reset does in
    evaluation mode; a control moves the root by its velocity and the joints by 0.02 times the action; the worlds of
    NONFINITE_AT get a non-finite state, those of SINK_AT a root 0.5 m lower, at their control."""

    def __init__(self, T: int):
        self.T, self.S, self.control = T, None, 0

    def place(self, ids, command_row, body_row, uniforms):
        ref = tuple(x.clone() for x in MOTION.ref_state(body_row))
        p, R, v, w, q, qd = placed_state(ref, DomainRandCfg(), False, uniforms)
        placed = (p, R, v, w, torch.clamp(q, KIN.q_lo, KIN.q_hi), qd)
        if self.S is None:
            self.S = tuple(torch.zeros(self.T, *x.shape[1:]) for x in placed)
        for x, y in zip(self.S, placed):
            x[ids] = y

    def state(self):
        return tuple(x.clone() for x in self.S)

    def observe(self, command_row):
        return torch.zeros(self.T, 1)

    def push(self, ids, velocity_xy):
        self.S[2][ids, :2] += velocity_xy

    def step(self, action, command_row):
        p, R, v, w, q, qd = self.S
        self.S = (p + 0.02 * v, R, v, w, q + 0.02 * action, qd)
        for world, c in NONFINITE_AT.items():
            if c == self.control and world < self.T:
                self.S[0][world] = float("nan")
        for world, c in SINK_AT.items():
            if c == self.control and world < self.T:
                self.S[0][world, 2] -= 0.5
        self.control += 1
        zero, no = torch.zeros(self.T, dtype=torch.long), torch.zeros(self.T, dtype=torch.bool)
        return self.state(), {"contacts": zero, "constraint_rows": zero, "contacts_beyond_capacity": no,
                              "rows_beyond_capacity": no}  # fmt: skip


class ConstantPolicy:
    hist = 1

    def __init__(self, action: float):
        self.action = action

    def __call__(self, hist, goal):  # another action per joint
        return (self.action + 0.01 * torch.arange(J)).expand(hist.shape[0], J)


def write_record(folder: str, scene: dict, action: float, library: dict, motion) -> rr.Record:
    """A record folder as tools/rollouts.py writes it (the description's keys the replay reads, rollouts.npz in full)
    of the scene under the recipe's termination."""
    trials = rollouts.read_scene(scene, motion.num_frames, J, CONTROLS)
    mdp = MDP(KIN, motion, torch.tensor(ROBOT.default_joint_pos_list()), XiLayout(ROBOT, "cpu").dim, ROBOT.sensing)
    rec = rollouts.roll_out(StandIn(len(scene["trials"])), ConstantPolicy(action), 1, mdp, trials, CONTROLS, True)
    os.makedirs(folder)
    description = {"run_args": {"robot": ROBOT.name, **library}, "scene": scene, "termination": "recipe",
                   "controls": CONTROLS, "library": library | {"clips": len(motion.names), "rows": motion.num_frames},
                   "bodies": {"names": list(ROBOT.tracked_bodies)}, "settings": {"control_dt": CONTROL_DT}}  # fmt: skip
    with open(os.path.join(folder, "description.json"), "w") as fh:
        json.dump(description, fh)
    np.savez_compressed(os.path.join(folder, "rollouts.npz"), **rec)
    return rr.Record(folder)


def clip_end_scene(a_last: int, b_last: int) -> dict:
    """Trials whose commands reach their clip's last row (a_walk's ``a_last``, b_run's ``b_last``): 0 at control 2 and
    runs; 1 starts at the last row (control 0) and runs; 2 at control 1, then ends at control 3 (non-finite); 3 at
    control 0, the control that ends it (non-finite); 4 at control 3, then ends at control 4 (sinks)."""
    rows = (a_last - 3, a_last, a_last - 2, a_last - 1, b_last - 4)
    return {"trials": [{"command_row": r} for r in rows]}


def setUpModule():
    global ROOT, KIN, MOTION, LIBRARY, A, B, E, MODEL, POSER
    ROOT = tempfile.mkdtemp()
    clips = os.path.join(ROOT, "clips")
    os.makedirs(clips)
    for name, rows in CLIPS.items():
        write_clip(os.path.join(clips, name), rows)
    KIN = Kinematics(ROBOT, "cpu")
    LIBRARY = {"motion_dir": clips, "bones_list": "", "bones_lib": "lib_all.npz", "motion_store": "full"}
    MOTION = MotionLib.load(ROBOT, KIN, "cpu", *LIBRARY.values())
    A = write_record(os.path.join(ROOT, "run_a"), SCENE, 0.5, LIBRARY, MOTION)
    B = write_record(os.path.join(ROOT, "run_b"), SCENE, -0.5, LIBRARY, MOTION)
    E = write_record(os.path.join(ROOT, "clip_end"), clip_end_scene(*map(int, MOTION.clip_end)), 0.5, LIBRARY, MOTION)
    MODEL = rr.build_model(ROBOT)
    POSER = rr.Poser(MODEL, ROBOT.joint_names)


def tearDownModule():
    shutil.rmtree(ROOT)


def field(rec: rr.Record, key: str) -> np.ndarray:
    """A field of the record's rollouts.npz, read whole."""
    with np.load(os.path.join(rec.folder, "rollouts.npz")) as z:
        return z[key]


def copy_record(rec: rr.Record, folder: str, drop: str | None = None, keep: list | None = None, **replace) -> str:
    """A copy of a record folder (the same description.json) without the field ``drop``, with the trials ``keep`` only
    (all when None) and with the arrays of ``replace``."""
    os.makedirs(folder)
    shutil.copy(os.path.join(rec.folder, "description.json"), folder)
    T = rec.trials
    with np.load(os.path.join(rec.folder, "rollouts.npz")) as z:
        arrays = {k: z[k] for k in z.files if k != drop}
    if keep is not None:
        arrays = {k: x[:, keep] if x.ndim > 1 and x.shape[:2] == (CONTROLS, T) else x[keep] if x.shape[:1] == (T,) else x
                  for k, x in arrays.items()}  # fmt: skip
    np.savez_compressed(os.path.join(folder, "rollouts.npz"), **(arrays | replace))
    return folder


def ended_copy(rec: rr.Record, folder: str, trial: int, control: int) -> rr.Record:
    """A copy of a record in which ``trial`` ends at ``control``, written after it as the rollout tool writes an ended
    trial: not available, NaN states and command bodies, rows -1."""
    edits = {k: field(rec, k) for k in ("done", "available", "state", "command_body_pos_w", "command_row_after",
                                        "clip_last_row")}  # fmt: skip
    edits["done"][:, trial] = False
    edits["done"][control, trial] = True
    edits["available"][control + 1 :, trial] = False
    for k, v in (("state", np.nan), ("command_body_pos_w", np.nan), ("command_row_after", -1), ("clip_last_row", -1)):
        edits[k][control + 1 :, trial] = v
    return rr.Record(copy_record(rec, folder, **edits))


class TestPoses(unittest.TestCase):
    def body_positions(self, qpos, poser=None):
        import mujoco

        data = mujoco.MjData(MODEL)
        (poser or POSER).place(data, qpos)
        return np.stack([data.xpos[MODEL.body(f"robot/{b}").id] for b in ROBOT.tracked_bodies])

    def test_the_records_roots_turn_and_tilt(self):  # so that the quaternion's order and sign matter below
        for trial, control in ((0, -1), (0, 5), (1, 5)):
            x = A["start_state"][trial] if control < 0 else A.column("state", trial)[control]
            self.assertGreater(abs(x[4]) + abs(x[5]), 0.03)  # roll or pitch
            self.assertGreater(abs(x[6]), 0.15)  # yaw
            self.assertGreater(np.ptp(x[13 : 13 + J] - ROBOT.default_joint_pos_list()), 0.05)  # per joint offsets

    def test_a_recorded_state_places_the_recorded_bodies(self):
        start, after = field(A, "start_body_pos_w"), field(A, "body_pos_w")
        for trial, control in ((0, -1), (0, 6), (1, 5), (4, 4), (2, 2)):
            x = A["start_state"][trial] if control < 0 else A.column("state", trial)[control]
            q = rr.body_qpos(A, trial, control)
            np.testing.assert_array_equal(q, np.concatenate((x[:3], x[3:7], x[13 : 13 + J])))
            bodies = start[trial] if control < 0 else after[control, trial]
            np.testing.assert_allclose(self.body_positions(q), bodies, atol=1e-4)

    def test_the_command_row_places_the_command_bodies(self):
        for trial, control in ((0, 0), (1, 7), (4, 2)):
            row = int(A.column("command_row_after", trial)[control])
            q = rr.command_qpos(MOTION, [row])[0]
            r = torch.tensor([row])
            np.testing.assert_allclose(q, torch.cat((MOTION.root_p[r], mat_to_quat(MOTION.root_R[r]), MOTION.q[r]), -1)[0])
            np.testing.assert_allclose(self.body_positions(q), A.column("command_body_pos_w", trial)[control], atol=1e-4)

    def test_joints_are_placed_by_name(self):
        q = rr.body_qpos(A, 1, 5)
        order = np.random.default_rng(0).permutation(J)
        permuted = np.concatenate((q[:7], q[7:][order]))
        poser = rr.Poser(MODEL, [ROBOT.joint_names[i] for i in order])
        np.testing.assert_allclose(self.body_positions(permuted, poser), self.body_positions(q), atol=1e-9)
        # placed by position instead, the permuted joints would move the bodies
        self.assertGreater(np.abs(self.body_positions(permuted) - self.body_positions(q)).max(), 0.01)

    def test_where_the_command_pose_is_drawn(self):
        # trial 1 is placed in another clip's pose than its command's; 2, 3 and 4 hold their last state after their end
        for trial in range(A.trials):
            for stride in (1, 3):
                plan = rr.trial_frames(A, trial, stride)
                command = rr.command_qpos(MOTION, [f["row"] for f in plan])
                body, world = rr.frame_poses(A, trial, plan, MOTION, "world")
                np.testing.assert_array_equal(body, np.stack([rr.body_qpos(A, trial, f["source"]) for f in plan]))
                np.testing.assert_array_equal(world, command)
                body_r, robot = rr.frame_poses(A, trial, plan, MOTION, "robot")
                np.testing.assert_array_equal(body_r, body)
                np.testing.assert_array_equal(robot[:, :2], body[:, :2])  # at the robot's root xy at every frame
                np.testing.assert_array_equal(robot[:, 2:], command[:, 2:])  # height, orientation and joints
        plan = rr.trial_frames(A, 1, 1)
        body, world = rr.frame_poses(A, 1, plan, MOTION, "world")
        self.assertGreater(np.linalg.norm(world[:, :2] - body[:, :2], axis=1).min(), 0.3)  # apart under "world"
        with self.assertRaisesRegex(ValueError, "drawn at one of"):
            rr.frame_poses(A, 1, plan, MOTION, "body")
        lines = rr.panel_text("s1", 1, ("b", "c"), plan[0], CONTROL_DT, "robot")
        self.assertEqual(lines[1], "body b | command c, drawn at the robot's position")
        self.assertEqual(rr.panel_text("s1", 1, ("b", "c"), plan[0], CONTROL_DT, "world")[1], "body b | command c")

    def test_rows_outside_the_library_are_refused(self):
        with self.assertRaisesRegex(ValueError, "rows outside the library"):
            rr.command_qpos(MOTION, [-1])

    def test_a_state_that_is_not_finite_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no finite state"):
            rr.body_qpos(A, 2, 3)


class TestFrames(unittest.TestCase):
    def test_frame_count_for_a_stride(self):
        self.assertEqual(rr.frame_times(8, 1), list(range(9)))
        self.assertEqual(rr.frame_times(8, 3), [0, 3, 6])
        self.assertEqual(rr.frame_times(8, 8), [0, 8])
        self.assertEqual(rr.frame_times(8, 9), [0])
        for s in range(1, 10):
            self.assertEqual(len(rr.trial_frames(A, 0, s)), CONTROLS // s + 1)
        with self.assertRaises(ValueError):
            rr.frame_times(8, 0)

    def test_frames_of_a_running_trial(self):
        frames = rr.trial_frames(A, 1, 2)
        self.assertEqual([f["time"] for f in frames], [0, 2, 4, 6, 8])
        self.assertEqual([f["control"] for f in frames], [-1, 1, 3, 5, 7])
        self.assertEqual([f["source"] for f in frames], [-1, 1, 3, 5, 7])
        self.assertEqual([f["row"] for f in frames], [26] + [int(A.column("command_row_after", 1)[c]) for c in (1, 3, 5, 7)])
        self.assertEqual([f["ended_at"] for f in frames], [None] * 5)

    def test_a_trial_ended_by_a_state_that_is_not_finite_holds_the_state_before(self):
        self.assertTrue(A.column("done", 2)[3] and not A.column("available", 2)[3])
        self.assertEqual((rr.ended_at(A, 2), rr.ended_at(A, 0)), (3, None))
        frames = rr.trial_frames(A, 2, 1)
        self.assertEqual([f["source"] for f in frames], [-1, 0, 1, 2, 2, 2, 2, 2, 2])
        self.assertEqual([f["ended_at"] for f in frames], [None] * 4 + [3] * 5)
        self.assertEqual({f["row"] for f in frames[3:]}, {int(A.column("command_row_after", 2)[2])})

    def test_a_trial_ended_at_its_first_control_holds_its_start(self):
        frames = rr.trial_frames(A, 3, 1)
        self.assertEqual({f["source"] for f in frames}, {-1})
        self.assertEqual({f["row"] for f in frames}, {20})
        self.assertEqual([f["ended_at"] for f in frames], [None] + [0] * 8)

    def test_a_trial_ended_by_a_finite_state_holds_that_state(self):
        done, available = A.column("done", 4), A.column("available", 4)
        self.assertTrue(done[4] and available[4] and not available[5])
        frames = rr.trial_frames(A, 4, 1)
        self.assertEqual([f["source"] for f in frames], [-1, 0, 1, 2, 3, 4, 4, 4, 4])
        self.assertEqual([f["ended_at"] for f in frames], [None] * 5 + [4] * 4)
        self.assertEqual(rr.panel_text("s1", 4, ("b", "c"), frames[7], CONTROL_DT)[2],
                         "after control 6 | 0.14 s | ended at control 4")  # fmt: skip
        self.assertEqual(rr.panel_text("s1", 4, ("b", "c"), frames[0], CONTROL_DT)[2], "start | 0.00 s")

    def test_a_missing_state_before_the_end_is_refused(self):
        available = field(A, "available")
        available[2, 0] = False
        rec = rr.Record(copy_record(A, os.path.join(ROOT, "gap"), available=available))
        with self.assertRaisesRegex(ValueError, "no recorded state after control 2 before its end"):
            rr.trial_frames(rec, 0, 1)

    def test_the_frame_rate_is_the_control_rate(self):
        self.assertEqual(rr.control_rate_fps(CONTROL_DT), "50")
        self.assertEqual(rr.control_rate_fps(1 / 30.5), "30.5")

    def test_a_record_read_for_some_trials(self):
        some = rr.Record(A.folder, [3, 1])
        self.assertEqual((some.chosen, some.trials, some.controls), ([3, 1], 5, CONTROLS))
        for t in (3, 1):
            for k in rr.CONTROL_FIELDS:
                np.testing.assert_array_equal(some.column(k, t), A.column(k, t))
            self.assertEqual(rr.trial_frames(some, t, 3), rr.trial_frames(A, t, 3))
        np.testing.assert_array_equal(some.column("state", 1), field(A, "state")[:, 1])
        with self.assertRaisesRegex(ValueError, "trial 0 is not among the trials read"):
            some.column("state", 0)
        with self.assertRaisesRegex(KeyError, "read per trial"):
            some["state"]
        with self.assertRaisesRegex(ValueError, r"trial 7 is not in \[0, 5\)"):
            rr.Record(A.folder, [7])


class TestStop(unittest.TestCase):
    def test_the_clip_ends_in_the_record(self):
        a_last, b_last = map(int, MOTION.clip_end)
        self.assertEqual((a_last, b_last), (25, 76))
        for t in range(E.trials):
            after = E.column("command_row_after", t)
            c = rr.clip_ended_at(E, t)
            self.assertEqual(after[c], E.column("clip_last_row", t)[c])
            self.assertTrue(c == 0 or after[c - 1] < after[c])  # reached at c, not before
            held = after[c:][after[c:] >= 0]
            self.assertEqual(set(held.tolist()), {after[c]})  # and held: the row no longer advances
        self.assertEqual([rr.clip_ended_at(E, t) for t in range(E.trials)], [2, 0, 1, 0, 3])
        self.assertEqual([rr.ended_at(E, t) for t in range(E.trials)], [None, None, 3, 0, 4])
        self.assertEqual([rr.clip_ended_at(A, t) for t in range(A.trials)], [None] * 5)
        # a row at the clip's last row that advances again (not written by the rollout tool) has not held
        after = field(E, "command_row_after")
        after[3, 0] -= 1
        edited = rr.Record(copy_record(E, os.path.join(ROOT, "advances_again"), command_row_after=after))
        self.assertEqual(rr.clip_ended_at(edited, 0), 4)

    def test_end_points(self):
        self.assertEqual(rr.end_point(A, 2), (3, "trial ended"))  # its command does not reach its clip's last row
        self.assertEqual(rr.end_point(A, 0), (None, "all controls"))  # neither
        self.assertEqual(rr.end_point(E, 0), (2, "clip ended"))
        self.assertEqual(rr.end_point(E, 2), (1, "clip ended"))  # before the trial's end at control 3
        self.assertEqual(rr.end_point(E, 3), (0, "trial ended"))  # the same control
        self.assertEqual(rr.end_point(E, 4), (3, "clip ended"))

    def test_seconds_to_controls(self):
        self.assertEqual(rr.tail_controls(0.06, CONTROL_DT), 3)  # 0.06 / 0.02 is 2.999... in floats
        self.assertEqual(rr.tail_controls(2.0, CONTROL_DT), 100)
        self.assertIsNone(rr.tail_controls(None, CONTROL_DT))
        self.assertEqual([rr.tail_seconds(x) for x in ("2", "0", "-1", "all")], [2.0, 0.0, None, None])
        for bad in ("soon", "nan", "inf"):
            with self.assertRaises(rr.argparse.ArgumentTypeError):
                rr.tail_seconds(bad)

    def stop(self, records, trial, tail):
        s = rr.video_stop(records, trial, tail)
        return s["control"], s["reason"]

    def test_the_stopping_control(self):
        self.assertEqual(self.stop([A], 2, 2), (5, "trial ended"))
        self.assertEqual(self.stop([E], 0, 2), (4, "clip ended"))
        self.assertEqual(self.stop([E], 3, 0), (0, "trial ended"))
        self.assertEqual(self.stop([A], 0, 2), (CONTROLS - 1, "all controls"))  # neither
        self.assertEqual(self.stop([A], 2, None), (CONTROLS - 1, "all controls"))  # every control
        self.assertEqual(self.stop([A], 2, 4), (7, "trial ended"))  # up to the last recorded control
        self.assertEqual(self.stop([A], 2, 5), (CONTROLS - 1, "all controls"))  # never beyond it
        self.assertEqual(rr.video_stop([E], 2, 1)["end_points"], [{"control": 1, "reason": "clip ended"}])

    def test_frames_up_to_the_stop(self):
        def controls(rec, trial, stride, tail):
            return [f["control"] for f in rr.trial_frames(rec, trial, stride, rr.video_stop([rec], trial, tail)["control"])]

        self.assertEqual(controls(A, 2, 1, 2), [-1, 0, 1, 2, 3, 4, 5])  # trial ended at 3
        self.assertEqual(controls(E, 0, 1, 1), [-1, 0, 1, 2, 3])  # clip ended at 2
        self.assertEqual(controls(E, 3, 1, 0), [-1, 0])  # both at 0
        self.assertEqual(controls(A, 0, 1, 2), list(range(-1, CONTROLS)))  # neither
        # a stride shows the first frame at or after the stop, at most the last recorded control
        self.assertEqual(controls(A, 2, 3, 2), [-1, 2, 5])
        self.assertEqual(controls(A, 2, 4, 2), [-1, 3, 7])
        self.assertEqual(controls(E, 0, 3, 0), [-1, 2])
        self.assertEqual(controls(E, 2, 2, 0), [-1, 1])
        self.assertEqual(controls(E, 4, 3, 1), [-1, 2, 5])
        self.assertEqual(controls(A, 0, 3, 2), [-1, 2, 5])  # as without a stop
        # a stop off the grid near the record's end: the last frame within the record comes before it
        self.assertEqual(self.stop([A], 4, 2), (6, "trial ended"))
        self.assertEqual(controls(A, 4, 3, 2), [-1, 2, 5])
        self.assertEqual(rr.trial_frames(A, 4, 1, CONTROLS - 1), rr.trial_frames(A, 4, 1))
        for bad in (-1, CONTROLS):
            with self.assertRaisesRegex(ValueError, "stop control"):
                rr.trial_frames(A, 0, 1, bad)

    def test_side_by_side_ends_after_the_later_panel(self):
        early = ended_copy(A, os.path.join(ROOT, "ends_at_1"), 2, 1)
        self.assertEqual((rr.end_point(early, 2), rr.end_point(A, 2)), ((1, "trial ended"), (3, "trial ended")))
        rr.check_same_trials(A, early)
        for pair in ((A, early), (early, A)):
            self.assertEqual(self.stop(pair, 2, 2), (5, "trial ended"))
            s = rr.video_stop(pair, 2, 2)
            self.assertEqual([p["control"] for p in s["end_points"]], [rr.end_point(r, 2)[0] for r in pair])
            plans = [rr.trial_frames(r, 2, 1, s["control"]) for r in pair]
            self.assertEqual([[f["time"] for f in p] for p in plans], [list(range(7))] * 2)  # one time axis
        # a clip end in one panel, a trial's end in the other: the later
        clip = ended_copy(E, os.path.join(ROOT, "e_ends_at_2"), 4, 2)
        self.assertEqual(rr.end_point(clip, 4), (2, "trial ended"))
        for pair in ((E, clip), (clip, E)):
            self.assertEqual(self.stop(pair, 4, 1), (4, "clip ended"))
        # a panel without an end point: every control
        late = ended_copy(A, os.path.join(ROOT, "ends_at_5"), 0, 5)
        for pair in ((A, late), (late, A)):
            self.assertEqual(self.stop(pair, 0, 0), (CONTROLS - 1, "all controls"))
        self.assertEqual(self.stop([late], 0, 0), (5, "trial ended"))


class TestChoiceAndRefusals(unittest.TestCase):
    def test_choice_and_order_of_trials(self):
        self.assertEqual(rr.choose_trials([3, 0, 2], 5), [3, 0, 2])
        for bad, msg in (([], "no trial"), ([5], "not in"), ([-1], "not in"), ([1, 0, 1], "twice")):
            with self.assertRaisesRegex(ValueError, msg):
                rr.choose_trials(bad, 5)

    def test_a_record_that_lacks_a_field_is_refused(self):
        folder = copy_record(A, os.path.join(ROOT, "no_available"), drop="available")
        with self.assertRaisesRegex(ValueError, r"lacks the fields \['available'\]"):
            rr.Record(folder)
        folder = os.path.join(ROOT, "no_control_dt")
        copy_record(A, folder)
        d = dict(A.description, settings={})
        with open(os.path.join(folder, "description.json"), "w") as fh:
            json.dump(d, fh)
        with self.assertRaisesRegex(ValueError, "settings.control_dt"):
            rr.Record(folder)

    def test_two_records_of_the_same_trials(self):
        self.assertEqual(rr.check_same_trials(A, B), 0.0)
        self.assertFalse(np.array_equal(field(A, "state"), field(B, "state"), equal_nan=True))
        # start states that differ within the tolerance
        start = A["start_state"].copy()
        start[0, 0] += 0.5 * rr.START_STATE_TOLERANCE
        c = rr.Record(copy_record(A, os.path.join(ROOT, "close_start"), start_state=start))
        self.assertLessEqual(rr.check_same_trials(A, c), rr.START_STATE_TOLERANCE)

    def test_two_records_of_other_trials_are_refused(self):
        other = dict(SCENE, trials=SCENE["trials"][:1] + [{"command_row": 26, "body_row": 61}] + SCENE["trials"][2:])
        c = write_record(os.path.join(ROOT, "other_body_row"), other, 0.5, LIBRARY, MOTION)
        with self.assertRaisesRegex(ValueError, "different scenes"):
            rr.check_same_trials(A, c)
        fewer = dict(SCENE, trials=SCENE["trials"][:3])
        d = write_record(os.path.join(ROOT, "fewer"), fewer, 0.5, LIBRARY, MOTION)
        with self.assertRaisesRegex(ValueError, "different scenes"):
            rr.check_same_trials(A, d)
        # the same scene content but other arrays
        e = rr.Record(copy_record(A, os.path.join(ROOT, "other_rows"), body_row=A["body_row"][::-1].copy()))
        with self.assertRaisesRegex(ValueError, "different body_rows"):
            rr.check_same_trials(A, e)
        command_row = A["command_row"].copy()
        command_row[2] += 1
        f = rr.Record(copy_record(A, os.path.join(ROOT, "other_command_row"), command_row=command_row))
        with self.assertRaisesRegex(ValueError, "different command_rows"):
            rr.check_same_trials(A, f)
        g = rr.Record(copy_record(A, os.path.join(ROOT, "three_trials"), keep=[0, 1, 2]))
        self.assertEqual(g.description["scene"], A.description["scene"])
        with self.assertRaisesRegex(ValueError, "holds 5 trials, .*three_trials 3"):
            rr.check_same_trials(A, g)
        start = A["start_state"].copy()
        start[4, 13] += 1e-3
        h = rr.Record(copy_record(A, os.path.join(ROOT, "other_start"), start_state=start))
        with self.assertRaisesRegex(ValueError, "start states .* differ by up to 0.001"):
            rr.check_same_trials(A, h)

    def test_another_library_is_refused(self):
        folder = os.path.join(ROOT, "more_clips")
        shutil.copytree(LIBRARY["motion_dir"], folder)
        write_clip(os.path.join(folder, "c_more.csv"), 16)
        more = rr.load_library(ROBOT, dict(LIBRARY, motion_dir=folder))
        with self.assertRaisesRegex(ValueError, "the motion library has 103 rows"):
            rr.check_library(more, A, [0])
        # the same rows in another order: b_run first
        folder = os.path.join(ROOT, "reordered")
        os.makedirs(folder)
        shutil.copy(os.path.join(LIBRARY["motion_dir"], "a_walk.csv"), os.path.join(folder, "c_walk.csv"))
        shutil.copy(os.path.join(LIBRARY["motion_dir"], "b_run.csv"), folder)
        reordered = rr.load_library(ROBOT, dict(LIBRARY, motion_dir=folder))
        self.assertEqual(reordered.num_frames, MOTION.num_frames)
        with self.assertRaisesRegex(ValueError, "reference bodies differ"):
            rr.check_library(reordered, A, [1])
        rr.check_library(MOTION, A, [0, 1, 2, 3, 4])

    def test_command_line_refusals(self):
        def run(*args):
            r = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)
            return r.returncode, r.stderr

        out = os.path.join(ROOT, "exists")
        os.makedirs(out, exist_ok=True)
        rc, err = run(A.folder, "--trials", "0", "--out", out)
        self.assertEqual(rc, 2)
        self.assertIn("exists", err)
        out = os.path.join(ROOT, "cli_out")
        for args, msg in (((A.folder, "--trials", "0", "--width", "161"), "even"),
                          ((A.folder, B.folder, "--trials", "0", "--labels", "s1"), "1 labels for 2 records"),
                          ((A.folder, B.folder, A.folder, "--trials", "0"), "one or two record folders"),
                          ((A.folder, "--trials", "0", "--tail", "soon"), "not a number of seconds"),
                          ((A.folder, "--trials", "0", "--ghost_at", "body"), "invalid choice")):  # fmt: skip
            rc, err = run(*args, "--out", out)
            self.assertEqual(rc, 2, err)
            self.assertIn(msg, err)
        self.assertFalse(os.path.exists(out))

    def test_no_opengl_context_is_refused_before_the_output_folder(self):
        out = os.path.join(ROOT, "no_gl")
        err = io.StringIO()
        with mock.patch.dict(os.environ), mock.patch("mujoco.Renderer", side_effect=RuntimeError("no context")), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:  # fmt: skip
            rr.main([A.folder, "--trials", "0", "--out", out])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("MuJoCo could not open an OpenGL context with MUJOCO_GL=", err.getvalue())
        self.assertIn("set the environment variable MUJOCO_GL", err.getvalue())
        self.assertFalse(os.path.exists(out))


class TestRender(unittest.TestCase):
    """Renders with MUJOCO_GL=egl when this machine has the backend; otherwise the tool must refuse, naming MUJOCO_GL."""

    def run_tool(self, out, *args, gl="egl"):
        cmd = [sys.executable, TOOL, *args, "--out", out, "--width", "160", "--height", "120"]
        return subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, MUJOCO_GL=gl), timeout=600)

    def refused(self, r, out, gl):
        print(f"\nrender: refused with MUJOCO_GL={gl}: {r.stderr.strip()[-400:]}", file=sys.stderr)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("MuJoCo could not", r.stderr)
        self.assertIn(f"with MUJOCO_GL={gl}", r.stderr)
        self.assertIn("set the environment variable MUJOCO_GL", r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_the_model_has_a_drawn_floor(self):
        import mujoco

        planes = [g for g in range(MODEL.ngeom) if MODEL.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        info = [(MODEL.geom_group[g], MODEL.geom_rgba[g].tolist(), MODEL.geom_matid[g]) for g in planes]
        alpha = lambda g: MODEL.mat_rgba[MODEL.geom_matid[g]][3] if MODEL.geom_matid[g] >= 0 else MODEL.geom_rgba[g][3]  # noqa: E731
        self.assertTrue(any(MODEL.geom_group[g] < 3 and alpha(g) > 0 for g in planes), info)

    def test_render_or_refusal_with_osmesa(self):
        out = os.path.join(ROOT, "videos_osmesa")
        r = self.run_tool(out, A.folder, "--trials", "0", "--stride", "8", gl="osmesa")
        if r.returncode != 0:
            self.refused(r, out, "osmesa")
            return
        print("\nrender: rendered with MUJOCO_GL=osmesa", file=sys.stderr)
        self.assertEqual(sorted(os.listdir(out)), ["trial_000.json", "trial_000.mp4", "trial_000.png"])

    def sheet(self, out, trial):
        return np.asarray(Image.open(os.path.join(out, f"trial_{trial:03d}.png")).convert("RGB")).astype(int)

    def test_render_or_refusal(self):
        out = os.path.join(ROOT, "videos")
        r = self.run_tool(out, A.folder, "--trials", "3", "1", "--stride", "4")
        if r.returncode != 0:
            self.refused(r, out, "egl")
            return
        print("\nrender: rendered with MUJOCO_GL=egl", file=sys.stderr)
        self.assertLess(r.stdout.index("trial 3:"), r.stdout.index("trial 1:"))
        self.assertEqual(sorted(os.listdir(out)), [f"trial_00{t}.{x}" for t in (1, 3) for x in ("json", "mp4", "png")])
        for t in (1, 3):
            with open(os.path.join(out, f"trial_00{t}.mp4"), "rb") as fh:
                mp4 = fh.read()
            self.assertEqual(mp4[4:8], b"ftyp")
            self.assertIn(b"avc1", mp4)  # H.264
            with open(os.path.join(out, f"trial_00{t}.json")) as fh:
                d = json.load(fh)
            self.assertEqual((d["trial"], d["frames"], d["stride"], d["fps"]), (t, 3, 4, 50.0))
            self.assertAlmostEqual(d["playback_speed"], 4.0)  # the control rate at a stride of 4
            self.assertEqual(d["records"], [A.folder])
            self.assertEqual(d["ended_at"], [0 if t == 3 else None])
            sheet = self.sheet(out, t)
            self.assertEqual(sheet.shape, (60, 3 * 80, 3))  # three frames at half size
            self.assertGreater(sheet.std(), 5.0)
        # the command pose is drawn: trial 1's body is 0.34 m from it
        bare = os.path.join(ROOT, "videos_without_command")
        r = self.run_tool(bare, A.folder, "--trials", "1", "--stride", "4", "--ghost_alpha", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        drawn, without = self.sheet(out, 1), self.sheet(bare, 1)
        changed = np.abs(drawn - without).sum(-1) > 0
        self.assertGreater(changed.sum(), 0)
        self.assertGreater(drawn[changed][:, 2].mean(), drawn[changed][:, 0].mean())  # translucent blue
        # side by side: one time axis, a header line above the two panels
        pair = os.path.join(ROOT, "videos_pair")
        r = self.run_tool(pair, A.folder, B.folder, "--labels", "a", "b", "--trials", "4", "--stride", "8")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(pair, "trial_004.json")) as fh:
            d = json.load(fh)
        self.assertEqual((d["records"], d["labels"], d["frames"], d["ended_at"]), ([A.folder, B.folder], ["a", "b"], 2, [4, 4]))
        self.assertEqual(d["start_state_gap"], 0.0)
        sheet = self.sheet(pair, 4)
        self.assertEqual(sheet.shape[1], 2 * 160)  # two frames of two panels at half size
        self.assertGreater(sheet.shape[0], 60)
        # a tail of 0.04 s: trial 2 ends at control 3, the video at control 5; the command pose at the robot's root
        short = os.path.join(ROOT, "videos_tail")
        r = self.run_tool(short, A.folder, "--trials", "2", "0", "--tail", "0.04", "--ghost_at", "robot")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(short, "trial_002.json")) as fh:
            d = json.load(fh)
        self.assertEqual((d["frames"], d["options"]["ghost_at"], d["options"]["tail_s"]), (7, "robot", 0.04))
        self.assertEqual(d["stop"], {"control": 5, "reason": "trial ended", "last_frame_control": 5,
                                     "end_points": [{"control": 3, "reason": "trial ended"}]})  # fmt: skip
        self.assertEqual(self.sheet(short, 2).shape, (2 * 60, 4 * 80, 3))  # seven frames: all on the sheet
        with open(os.path.join(short, "trial_000.json")) as fh:
            d = json.load(fh)
        self.assertEqual((d["frames"], d["stop"]["control"], d["stop"]["reason"]), (CONTROLS + 1, 7, "all controls"))


if __name__ == "__main__":
    unittest.main()
