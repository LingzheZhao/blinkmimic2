"""The control period (train.py --control_period): the motion library's row period, which the MDP, the simulator and
the tools that load a run follow. At the default 0.02 s every quantity keeps the release's value; at 0.04 s the
durations in seconds stay the same.

train.py builds the simulator and trains once its options are parsed, so only its refusal runs here; the simulator's
physics steps per control and a training at another period need a GPU."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ROBOT  # noqa: E402
from test_motion_dir import write_clip  # noqa: E402

from blinkmimic2.dr import XiLayout  # noqa: E402
from blinkmimic2.env import EPISODE_S, PHYSICS_DT, decimation  # noqa: E402
from blinkmimic2.kinematics import Kinematics  # noqa: E402
from blinkmimic2.mdp import MDP  # noqa: E402
from blinkmimic2.motion import CONTROL_DT, MotionLib, run_control_dt  # noqa: E402

KIN = Kinematics(ROBOT, "cpu")


def library(d: str, control_dt: float, store: str = "full") -> MotionLib:
    return MotionLib.load(ROBOT, KIN, "cpu", d, "", None, store, control_dt)


def mdp_of(motion: MotionLib) -> MDP:
    return MDP(KIN, motion, torch.tensor(ROBOT.default_joint_pos_list()), XiLayout(ROBOT, "cpu").dim, ROBOT.sensing)


class TestPeriod(unittest.TestCase):
    def test_the_default_keeps_the_release_s_values(self):
        self.assertEqual(CONTROL_DT, 0.02)
        self.assertEqual(PHYSICS_DT, 0.02 / 4)  # the release's CONTROL_DT / DECIMATION
        self.assertEqual(decimation(CONTROL_DT), 4)
        self.assertEqual(round(EPISODE_S / CONTROL_DT), 500)  # the release's max_episode_steps
        with tempfile.TemporaryDirectory() as d:
            write_clip(os.path.join(d, "walk.csv"), 91)
            motion = library(d, CONTROL_DT)
        self.assertEqual((motion.dt, motion.bin_frames), (0.02, 50))
        self.assertEqual(mdp_of(motion).max_step_reward, (4 * 1.0 + 2 * 0.5) * 0.02)  # the release's MAX_STEP_REWARD

    def test_physics_steps_per_control(self):
        self.assertEqual([decimation(t) for t in (0.005, 0.02, 0.035, 0.04)], [1, 4, 7, 8])
        for t in (0.0, 0.0123, 0.0225):
            with self.assertRaisesRegex(ValueError, "not a whole number of 0.005 s physics steps"):
                decimation(t)

    def test_durations_in_seconds_stay_at_another_period(self):
        with tempfile.TemporaryDirectory() as d:
            write_clip(os.path.join(d, "walk.csv"), 91)  # 3 s at 30 fps, root moving at 0.5 m/s along x
            for store in ("full", "fp32"):
                m20, m40 = library(d, 0.02, store), library(d, 0.04, store)
                self.assertEqual(m20.clip_len.tolist(), [151])
                self.assertEqual(m40.clip_len.tolist(), [76])
                self.assertEqual((m40.dt, m40.bin_frames), (0.04, 25))
                # row k at 0.04 s is the time of row 2k at 0.02 s
                torch.testing.assert_close(m40.root_p[torch.arange(76)], m20.root_p[2 * torch.arange(76)])
                # velocities by differences over the period
                v = m40.ref_state(torch.arange(76))[2]
                torch.testing.assert_close(v[:, 0], torch.full((76,), 0.5))
                # one-second bins of 25 rows
                self.assertEqual(m40.frame_bin[torch.tensor([0, 24, 25, 50, 74])].tolist(), [0, 0, 1, 2, 2])

    def test_the_reward_is_a_rate_times_the_period(self):
        with tempfile.TemporaryDirectory() as d:
            write_clip(os.path.join(d, "walk.csv"), 91)
            for dt in (0.02, 0.04):
                motion = library(d, dt)
                mdp = mdp_of(motion)
                self.assertEqual(mdp.max_step_reward, 5.0 * dt)
                f = torch.arange(10, 20)
                a = torch.zeros(10, ROBOT.num_joints)
                # on the reference, with no action change: every tracking term is 1
                torch.testing.assert_close(mdp.reward(motion.ref_state(f), f, a, a), torch.full((10,), 5.0 * dt))

    def test_the_packed_library_is_not_resampled(self):
        with self.assertRaisesRegex(ValueError, "one row per 0.02 s, not per 0.04 s"):
            MotionLib.load(ROBOT, KIN, "cpu", None, "keep.txt", "lib_all.npz", "full", 0.04)

    def test_a_run_s_period(self):
        self.assertEqual(run_control_dt({"control_period": 0.04, "robot": "mjlab_g1"}), 0.04)
        self.assertEqual(run_control_dt({"control_period": 0.02}), 0.02)
        self.assertEqual(run_control_dt({"robot": "mjlab_g1"}), 0.02)  # args.json written before --control_period


class TestCommandLine(unittest.TestCase):
    def test_a_period_that_is_not_whole_physics_steps_is_refused(self):
        r = subprocess.run([sys.executable, "-m", "blinkmimic2.train", "--control_period", "0.0123"],
                           capture_output=True, text=True)  # fmt: skip
        self.assertEqual(r.returncode, 2)
        self.assertIn("--control_period: the control period 0.0123 s is not a whole number", r.stderr)

    def test_rollouts_refuse_a_period_other_than_the_run_s(self):
        tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
        for run, extra, trained in (({"control_period": 0.04}, (), 0.04), ({"control_period": 0.02}, ("--control_period", "0.04"), 0.02),
                                    ({}, ("--control_period", "0.04"), 0.02)):  # fmt: skip
            with tempfile.TemporaryDirectory() as d:
                json.dump(run | {"robot": "mjlab_g1"}, open(os.path.join(d, "args.json"), "w"))
                cmd = [sys.executable, os.path.join(tools, "rollouts.py"), "--run_dir", d, "--ckpt", "1", "--scene", "s.json",
                       "--termination", "nonfinite", "--controls", "5", "--out", os.path.join(d, "out"), *extra]  # fmt: skip
                r = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(r.returncode, 2, run)
                self.assertIn(f"the run trained at {trained} s", r.stderr)
                self.assertFalse(os.path.exists(os.path.join(d, "out")))


if __name__ == "__main__":
    unittest.main()
