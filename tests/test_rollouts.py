"""tools/rollouts.py on the CPU: the scene's checks, and the rollout loop driven by a stand-in for the simulator (the
synthetic library of synthetic.py, the release's MDP and kinematics)."""

import copy
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
sys.path[:0] = [HERE, TOOLS]
import rollouts  # noqa: E402
from synthetic import ROBOT, library  # noqa: E402

from blinkmimic2.dr import DomainRandCfg  # noqa: E402
from blinkmimic2.env import placed_state  # noqa: E402
from blinkmimic2.mdp import flatten_state, unflatten_state  # noqa: E402
from blinkmimic2.motion import GOAL_Q_OFFSET  # noqa: E402
from blinkmimic2.networks import Policy  # noqa: E402
from blinkmimic2.rotations import mat_to_quat  # noqa: E402

KIN, MOTION, MDP_ = library()
J, B, ROWS = ROBOT.num_joints, len(ROBOT.tracked_bodies), MOTION.num_frames
WALK, LOW, TILTED, LYING = 0, 6, 10, 14  # first rows of the synthetic clips
PER_TRIAL = {"start_state", "start_body_pos_w", "start_body_quat_w", "command_row", "body_row", "push_control",
             "push_velocity_xy"}  # fmt: skip


class StandIn:
    """Stand-in for rollouts.Simulator: per-world states placed as TrackingSim.reset places them in evaluation mode;
    a control moves each root by its velocity and the joints by the action; the report depends only on the command
    row; the worlds of ``nonfinite_at`` (world -> control) get a non-finite state at that control."""

    def __init__(self, T: int, nonfinite_at: dict | None = None):
        self.T, self.nonfinite_at, self.control, self.placed = T, nonfinite_at or {}, 0, []
        self.S = None

    def place(self, ids, command_row, body_row, uniforms):
        ref = tuple(x.clone() for x in MOTION.ref_state(body_row))
        p, R, v, w, q, qd = placed_state(ref, DomainRandCfg(), False, uniforms)
        placed = (p, R, v, w, torch.clamp(q, KIN.q_lo, KIN.q_hi), qd)
        if self.S is None:
            self.S = tuple(torch.zeros(self.T, *x.shape[1:]) for x in placed)
        for x, y in zip(self.S, placed):
            x[ids] = y
        self.placed.append(ids.tolist())

    def state(self):
        return tuple(x.clone() for x in self.S)

    def observe(self, command_row):
        return MDP_.obs(self.S, torch.zeros(self.T, J), torch.zeros(self.T, MDP_.xi_dim), command_row)

    def push(self, ids, velocity_xy):
        self.S[2][ids, :2] += velocity_xy

    def step(self, action, command_row):
        assert bool((command_row >= 0).all() and (command_row < ROWS).all())
        p, R, v, w, q, qd = self.S
        self.S = (p + 0.02 * v, R, v, w, q + 0.02 * action, qd)
        for world, c in self.nonfinite_at.items():
            if c == self.control:
                self.S[0][world] = float("nan")
        self.control += 1
        report = {"contacts": command_row % 5 + self.control, "constraint_rows": 2 * command_row,
                  "contacts_beyond_capacity": command_row % 2 == 0, "rows_beyond_capacity": command_row % 3 == 0}  # fmt: skip
        return self.state(), report


class RowPolicy:
    """A policy whose action of a world depends on that world's history and goal only."""

    hist = 3

    def __call__(self, hist, goal):
        return 0.1 * torch.tanh(hist[:, -1, :J] - hist[:, 0, :J]) + 0.01 * goal[:, GOAL_Q_OFFSET : GOAL_Q_OFFSET + J]


def trial(command_row, body_row=None, uniforms=None, push=None) -> dict:
    t = {"command_row": command_row}
    if body_row is not None:
        t["body_row"] = body_row
    if uniforms is not None:
        t["reset_uniforms"] = uniforms
    if push is not None:
        t["push"] = push
    return t


class SetAfterStep(StandIn):
    """StandIn whose step then puts each world's root ``below[i]`` m below the command root after the control and its
    up-axis cosine (R[2, 2]) ``tilted[i]`` below the command root's, by a roll about x."""

    def __init__(self, below: list, tilted: list):
        super().__init__(len(below))
        self.below, self.tilted = torch.tensor(below), torch.tensor(tilted)

    def step(self, action, command_row):
        _, report = super().step(action, command_row)
        after = torch.minimum(command_row + 1, MOTION.clip_end[MOTION.clip_id[command_row]])
        c = MOTION.root_R[after][:, 2, 2] - self.tilted
        s, one, zero = torch.sqrt(1 - c * c), torch.ones_like(c), torch.zeros_like(c)
        self.S[0][:, 2] = MOTION.root_p[after][:, 2] - self.below
        self.S[1][:] = torch.stack((one, zero, zero, zero, c, -s, zero, s, c), -1).view(-1, 3, 3)
        return self.state(), report


def run(trials, controls, root=False, nonfinite_at=None, policy=None, env=None):
    env = env or StandIn(len(trials), nonfinite_at)
    tr = rollouts.read_scene({"trials": trials}, ROWS, J, controls)
    return rollouts.roll_out(env, policy or RowPolicy(), 4, MDP_, tr, controls, root), env


def uniforms(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {k: (2 * torch.rand(n, generator=g, dtype=torch.float64) - 1).tolist()
            for k, n in (("position", 3), ("rotation", 3), ("velocity", 6), ("joints", J))}  # fmt: skip


class TestScene(unittest.TestCase):
    def test_defaults(self):
        tr = rollouts.read_scene({"trials": [trial(3)]}, ROWS, J, 10)
        self.assertEqual((tr["command_row"].tolist(), tr["body_row"].tolist(), tr["push_control"].tolist()), ([3], [3], [-1]))
        for k in rollouts.UNIFORMS + ("push_velocity_xy",):
            self.assertFalse(bool(tr[k].any()))

    def test_given_values(self):
        u = uniforms(1)
        tr = rollouts.read_scene({"trials": [trial(2, 15, u, {"control": 9, "velocity_xy": [0.5, -3]})]}, ROWS, J, 10)
        self.assertEqual((tr["body_row"].item(), tr["push_control"].item()), (15, 9))
        self.assertEqual(tr["push_velocity_xy"].tolist(), [[0.5, -3.0]])
        for k in rollouts.UNIFORMS:
            torch.testing.assert_close(tr[k][0], torch.tensor(u[k], dtype=torch.float32))

    def test_refusals(self):
        u = uniforms(2)
        bad_u = lambda k, x: {**u, k: x}  # noqa: E731
        scenes = [
            [], {"trials": []}, {"trials": [trial(0)], "seed": 1}, {"trials": [{"body_row": 1}]},
            {"trials": [{**trial(0), "clip": "walk"}]}, {"trials": [trial(ROWS)]}, {"trials": [trial(-1)]},
            {"trials": [trial(True)]}, {"trials": [trial(1.0)]}, {"trials": [trial(0, ROWS)]},
            {"trials": [trial(0, uniforms={k: x for k, x in u.items() if k != "joints"})]},
            {"trials": [trial(0, uniforms={**u, "joint_velocities": [0.0] * J})]},
            {"trials": [trial(0, uniforms=bad_u("position", [0.0, 0.0]))]},
            {"trials": [trial(0, uniforms=bad_u("rotation", [0.0, 1.5, 0.0]))]},
            {"trials": [trial(0, uniforms=bad_u("velocity", [float("nan")] * 6))]},
            {"trials": [trial(0, uniforms=bad_u("joints", [0.0] * (J - 1)))]},
            {"trials": [trial(0, push={"control": 10, "velocity_xy": [1.0, 0.0]})]},
            {"trials": [trial(0, push={"control": 2, "velocity_xy": [1.0]})]},
            {"trials": [trial(0, push={"control": 2, "velocity_xy": [float("inf"), 0.0]})]},
            {"trials": [trial(0, push={"control": 2, "velocity_xy": [1.0, 0.0], "duration": 0.2})]},
            {"trials": [5]}, {"trials": [trial(0, uniforms=[])]}, {"trials": [trial(0, push=3)]},
            {"trials": [trial(0, push={"control": -1, "velocity_xy": [1.0, 0.0]})]},
            {"trials": [trial(0, push={"control": True, "velocity_xy": [1.0, 0.0]})]},
            {"trials": [trial(0, push={"control": 2, "velocity_xy": [True, 0.0]})]},
            {"trials": [trial(0, uniforms=bad_u("joints", [1.5] + [0.0] * (J - 1)))]},
            {"trials": [trial(0, uniforms=bad_u("velocity", [0.0] * 5 + [-1.01]))]},
        ]  # fmt: skip
        for scene in scenes:
            with self.subTest(scene=scene), self.assertRaises(ValueError):
                rollouts.read_scene(scene, ROWS, J, 10)


class TestRollOut(unittest.TestCase):
    def test_fields(self):
        C, scene = 4, [trial(1), trial(2, LYING + 1, uniforms(3), {"control": 1, "velocity_xy": [1.0, 0.0]})]
        policy = Policy(MDP_.obs_dim, MOTION.goal_dim(4), J, hist=3, enc_hidden=(16,), dec_hidden=(16,))
        rec, _ = run(scene, C, policy=policy.eval())
        T, S = len(scene), 13 + 2 * J
        f, i, b = np.float32, np.int32, np.bool_
        want = {
            "body_pos_w": ((C, T, B, 3), f), "body_quat_w": ((C, T, B, 4), f), "command_body_pos_w": ((C, T, B, 3), f),
            "command_body_quat_w": ((C, T, B, 4), f), "command_row_before": ((C, T), i), "command_row_after": ((C, T), i),
            "clip_last_row": ((C, T), i), "available": ((C, T), b), "done": ((C, T), b), "nonfinite": ((C, T), b),
            "termination_height": ((C, T), b), "termination_tilt": ((C, T), b), "contacts": ((C, T), i),
            "constraint_rows": ((C, T), i), "contacts_beyond_capacity": ((C, T), b), "rows_beyond_capacity": ((C, T), b),
            "action": ((C, T, J), f), "state": ((C, T, S), f), "push_applied": ((C, T), b),
            "start_state": ((T, S), f), "start_body_pos_w": ((T, B, 3), f), "start_body_quat_w": ((T, B, 4), f),
            "command_row": ((T,), i), "body_row": ((T,), i), "push_control": ((T,), i), "push_velocity_xy": ((T, 2), f),
        }  # fmt: skip
        self.assertEqual(set(rec), set(want))
        for k, (shape, dtype) in want.items():
            self.assertEqual((rec[k].shape, rec[k].dtype), (shape, np.dtype(dtype)), k)

    def test_record_holds_the_stepped_state_and_the_command_bodies(self):
        rec, _ = run([trial(1, uniforms=uniforms(4)), trial(LOW)], 3)
        S = unflatten_state(torch.from_numpy(rec["state"]).reshape(-1, 13 + 2 * J))
        bp, bR = KIN.forward(*S)[:2]
        # the state's rotation passes through its quaternion here, hence a tolerance
        np.testing.assert_allclose(rec["body_pos_w"].reshape(-1, B, 3), bp.numpy(), atol=1e-6)
        np.testing.assert_allclose(rec["body_quat_w"].reshape(-1, B, 4), mat_to_quat(bR).numpy(), atol=1e-6)
        cbp, cbR = MOTION.ref_bodies(torch.from_numpy(rec["command_row_after"]).long().flatten())[:2]
        np.testing.assert_array_equal(rec["command_body_pos_w"].reshape(-1, B, 3), cbp.numpy())
        np.testing.assert_array_equal(rec["command_body_quat_w"].reshape(-1, B, 4), mat_to_quat(cbR).numpy())

    def test_uniforms_reach_the_placement(self):
        u = uniforms(5)
        rec, _ = run([trial(3, LYING + 2, u), trial(2)], 1)
        tu = {k: torch.tensor([u[k], [0.0] * len(u[k])], dtype=torch.float32) for k in u}
        ref = tuple(x.clone() for x in MOTION.ref_state(torch.tensor([LYING + 2, 2])))
        p, R, v, w, q, qd = placed_state(ref, DomainRandCfg(), False, tu)
        start = unflatten_state(torch.from_numpy(rec["start_state"]))
        torch.testing.assert_close(start[0], p)
        torch.testing.assert_close(start[1], R)
        torch.testing.assert_close(start[2], v)
        torch.testing.assert_close(start[3], w)
        torch.testing.assert_close(start[4], torch.clamp(q, KIN.q_lo, KIN.q_hi))
        np.testing.assert_array_equal(rec["start_state"][1], flatten_state(ref)[1].numpy())  # zero uniforms: the reference

    def test_command_row_advances_and_holds_at_the_clip_end(self):
        # each trial starts two rows before the last row of its own clip: walk, low and lying, in one batch
        lasts = [LOW - 1, TILTED - 1, ROWS - 1]
        for root in (False, True):
            rec, _ = run([trial(last - 2) for last in lasts], 5, root=root)
            for t, last in enumerate(lasts):
                self.assertEqual(rec["command_row_before"][:, t].tolist(), [last - 2, last - 1, last, last, last])
                self.assertEqual(rec["command_row_after"][:, t].tolist(), [last - 1, last, last, last, last])
                self.assertEqual(rec["clip_last_row"][:, t].tolist(), [last] * 5)
            self.assertFalse(rec["done"].any())
            self.assertTrue(rec["available"].all())

    def test_terminations(self):
        # upright at the command; 0.36 m low (height term only); rolled 90 degrees (tilt term only); lying (both)
        scene = [trial(1), trial(1, LOW + 1), trial(1, TILTED + 1), trial(1, LYING + 1)]
        nonfinite, _ = run(scene, 3, root=False)
        on_root, _ = run(scene, 3, root=True)
        terms = [[False, False], [True, False], [False, True], [True, True]]
        self.assertEqual([[bool(nonfinite["termination_height"][0, t]), bool(nonfinite["termination_tilt"][0, t])]
                          for t in range(4)], terms)  # fmt: skip
        self.assertFalse(nonfinite["done"].any())
        self.assertEqual(on_root["done"][0].tolist(), [False, True, True, True])
        self.assertFalse(on_root["done"][1:].any())
        self.assertTrue(on_root["available"][:, 0].all() and on_root["available"][0].all())
        self.assertFalse(on_root["available"][1:, 1:].any())
        # the record of the first control is the same under both terminations
        for k in ("state", "termination_height", "termination_tilt", "command_row_after"):
            np.testing.assert_array_equal(on_root[k][0], nonfinite[k][0])
        # the two terms are the release's termination (MDP.termination) against the command row after the control
        S = unflatten_state(torch.from_numpy(nonfinite["state"]).reshape(-1, 13 + 2 * J))
        release = MDP_.termination(S, torch.from_numpy(nonfinite["command_row_after"]).long().flatten())
        np.testing.assert_array_equal(release.numpy(), (nonfinite["termination_height"] | nonfinite["termination_tilt"]).flatten())

    def test_terminations_at_their_thresholds(self):
        # 0.24 and 0.26 m below the command root, 0.26 m above it, and up-axis cosines 0.79 and 0.81 below the command's
        below, tilted = [0.24, 0.26, -0.26, 0.0, 0.0], [0.0, 0.0, 0.0, 0.79, 0.81]
        rec, _ = run([trial(1)] * 5, 1, root=True, env=SetAfterStep(below, tilted))
        self.assertEqual(rec["termination_height"][0].tolist(), [False, True, True, False, False])
        self.assertEqual(rec["termination_tilt"][0].tolist(), [False, False, False, False, True])
        self.assertEqual(rec["done"][0].tolist(), [False, True, True, False, True])
        S = unflatten_state(torch.from_numpy(rec["state"][0]))
        release = MDP_.termination(S, torch.from_numpy(rec["command_row_after"][0]).long())
        self.assertEqual(release.tolist(), rec["done"][0].tolist())

    def test_a_nonfinite_state_ends_a_trial_under_both_terminations(self):
        for root in (False, True):
            rec, env = run([trial(0), trial(1), trial(2)], 6, root=root, nonfinite_at={1: 2})
            self.assertEqual(rec["done"][:, 1].tolist(), [False, False, True, False, False, False])
            self.assertEqual(rec["nonfinite"][:, 1].tolist(), [False, False, True, False, False, False])
            self.assertEqual(rec["available"][:, 1].tolist(), [True, True, False, False, False, False])
            self.assertEqual(rec["command_row_after"][:, 1].tolist(), [2, 3, 4, -1, -1, -1])
            # the step's report while a trial runs (the stand-in's values of the row before), -1 or false after its end
            running = rec["command_row_before"] >= 0
            self.assertEqual(running[:, 1].tolist(), [True, True, True, False, False, False])
            self.assertTrue(running[:, [0, 2]].all())
            before, control = rec["command_row_before"], np.arange(6)[:, None]
            np.testing.assert_array_equal(rec["contacts"], np.where(running, before % 5 + control + 1, -1))
            np.testing.assert_array_equal(rec["constraint_rows"], np.where(running, 2 * before, -1))
            np.testing.assert_array_equal(rec["contacts_beyond_capacity"], running & (before % 2 == 0))
            np.testing.assert_array_equal(rec["rows_beyond_capacity"], running & (before % 3 == 0))
            self.assertTrue(np.isnan(rec["body_pos_w"][2:, 1]).all() and np.isnan(rec["action"][3:, 1]).all())
            self.assertFalse(rec["done"][:, [0, 2]].any())
            # the ended trial is placed again at its start before every later control
            self.assertEqual(env.placed, [[0, 1, 2], [1], [1], [1], [1]])

    def test_an_ended_trial_leaves_the_others_unchanged(self):
        scene = [trial(0, uniforms=uniforms(6)), trial(1, LYING), trial(2, push={"control": 1, "velocity_xy": [0.3, 0.1]})]
        rec, _ = run(scene, 5, root=True, nonfinite_at={1: 1})
        alone, _ = run([scene[0], scene[2]], 5, root=True)
        self.assertTrue(rec["done"][0, 1])
        for k, v in alone.items():
            np.testing.assert_array_equal(np.delete(rec[k], 1, axis=0 if k in PER_TRIAL else 1), v, err_msg=k)

    def test_push(self):
        dv = [0.4, -0.25]
        scene = [trial(1, push={"control": 2, "velocity_xy": dv}), trial(2)]
        pushed, _ = run(scene, 4)
        still, _ = run([trial(1), trial(2)], 4)
        self.assertEqual(pushed["push_applied"].tolist(), [[False, False], [False, False], [True, False], [False, False]])
        np.testing.assert_array_equal(pushed["state"][:2], still["state"][:2])
        np.testing.assert_array_equal(pushed["state"][:, 1], still["state"][:, 1])
        np.testing.assert_allclose(pushed["state"][2:, 0, 7:9] - still["state"][2:, 0, 7:9], [dv, dv], atol=1e-6)
        np.testing.assert_array_equal(pushed["state"][2:, 0, 9:13], still["state"][2:, 0, 9:13])

    def test_a_push_after_the_end_is_not_applied(self):
        rec, _ = run([trial(1, LYING, push={"control": 2, "velocity_xy": [1.0, 0.0]})], 4, root=True)
        self.assertTrue(rec["done"][0, 0])
        self.assertFalse(rec["push_applied"].any())


class TestCommandLine(unittest.TestCase):
    def call(self, *extra):
        cmd = [sys.executable, os.path.join(TOOLS, "rollouts.py"), "--run_dir", "run", "--ckpt", "1", "--scene", "s.json",
               "--termination", "nonfinite", *extra]  # fmt: skip
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_an_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as out:
            r = self.call("--controls", "5", "--out", out)
        self.assertEqual(r.returncode, 2)
        self.assertIn("exists", r.stderr)

    def test_no_controls_is_refused(self):
        r = self.call("--controls", "0", "--out", "never_made")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--controls", r.stderr)
        self.assertFalse(os.path.exists("never_made"))

    def test_help(self):
        r = subprocess.run([sys.executable, os.path.join(TOOLS, "rollouts.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--termination", r.stdout)


if __name__ == "__main__":
    unittest.main()
