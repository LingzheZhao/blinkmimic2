"""Root velocity pushes in training: TrackingSim._push as fixed-size masked updates of every world (so the captured
collection step can push), driven by a stand-in for the simulator on the CPU, and train.py's push options.

train.py builds the simulator and trains once its options are parsed, so only its refusals run here; a training with
pushes needs a GPU."""

import dataclasses
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ROBOT  # noqa: E402

from blinkmimic2.dr import DomainRandCfg  # noqa: E402
from blinkmimic2.env import TrackingSim  # noqa: E402
from blinkmimic2.motion import CONTROL_DT  # noqa: E402
from blinkmimic2.rotations import mat_to_quat, quat_to_mat, so3_exp  # noqa: E402

J = ROBOT.num_joints
HALF = (0.5, 0.5, 0.2, 0.52, 0.52, 0.78)  # x, y, z (m/s), roll, pitch, yaw (rad/s)
INTERVAL = (1.0, 3.0)


def stand_in(timer: torch.Tensor, seed: int = 0, dt: float = CONTROL_DT):
    """The attributes TrackingSim._push and TrackingSim.state read and write, without the simulator: N worlds with
    random root poses and velocities, the free joint first in qpos and qvel, and the control period dt."""
    N = len(timer)
    g = torch.Generator().manual_seed(seed)
    quat = mat_to_quat(so3_exp(2 * torch.rand(N, 3, generator=g) - 1))
    qpos = torch.cat((torch.rand(N, 3, generator=g), quat, torch.rand(N, J, generator=g)), -1)
    qvel = torch.randn(N, 6 + J, generator=g)
    env = SimpleNamespace(
        N=N, device="cpu", dt=dt, push_timer=timer.clone(), push_half=torch.tensor(HALF),
        spec=dataclasses.replace(ROBOT, dr=DomainRandCfg(push_interval_s=INTERVAL, push_velocity=HALF)),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel)),
        robot=SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7), free_joint_v_adr=torch.arange(6)),
                              data=SimpleNamespace(joint_pos=qpos[:, 7:], joint_vel=qvel[:, 6:])),
        scene=SimpleNamespace(env_origins=torch.zeros(N, 3)))  # fmt: skip
    env.state = lambda: TrackingSim.state(env)
    env._redraw_push_timer = lambda mask: TrackingSim._redraw_push_timer(env, mask)
    return env


def world_velocity(env) -> torch.Tensor:
    """World-frame root linear and angular velocity, as TrackingSim.state gives them."""
    _, _, v, w, _, _ = TrackingSim.state(env)
    return torch.cat((v, w), -1)


class TestPush(unittest.TestCase):
    def test_due_worlds_get_the_kick_in_the_world_frame_and_a_new_timer(self):
        timer = torch.tensor([0.01, 0.5, 0.02, -0.1, 0.019, 2.0])
        done = torch.tensor([False, False, True, False, False, False])
        env = stand_in(timer)
        qpos0, qvel0, vel0 = env.sim.data.qpos.clone(), env.sim.data.qvel.clone(), world_velocity(env)
        torch.manual_seed(3)
        due = TrackingSim._push(env, done)
        # the draws in their order: the kick of every world, then the timer of every world
        torch.manual_seed(3)
        kick = (2 * torch.rand(6, 6) - 1) * torch.tensor(HALF)
        u = torch.rand(6)
        want_due = torch.tensor([True, False, False, True, True, False])  # timer - control period <= 0, episode goes on
        self.assertTrue(torch.equal(due, want_due))
        vel1, qvel1 = world_velocity(env), env.sim.data.qvel
        torch.testing.assert_close(vel1[due], vel0[due] + kick[due])
        self.assertTrue(torch.equal(vel1[due, :3], vel0[due, :3] + kick[due, :3]))  # the free joint's linear velocity
        self.assertTrue(torch.equal(qvel1[~due], qvel0[~due]))  # other worlds bitwise unchanged
        self.assertTrue(torch.equal(qvel1[:, 6:], qvel0[:, 6:]))  # joint velocities unchanged
        self.assertTrue(torch.equal(env.sim.data.qpos, qpos0))  # positions unchanged
        lo, hi = INTERVAL
        self.assertTrue(torch.equal(env.push_timer[due], (lo + (hi - lo) * u)[due]))
        self.assertTrue(torch.equal(env.push_timer[~due], (timer - CONTROL_DT)[~due]))

    def test_without_a_due_world_no_velocity_changes(self):
        timer = torch.tensor([0.5, 1.0, 2.9])
        env = stand_in(timer, seed=1)
        qvel0 = env.sim.data.qvel.clone()
        due = TrackingSim._push(env, torch.zeros(3, dtype=torch.bool))
        self.assertFalse(due.any())
        self.assertTrue(torch.equal(env.sim.data.qvel, qvel0))
        self.assertTrue(torch.equal(env.push_timer, timer - CONTROL_DT))

    def test_the_timer_counts_seconds(self):
        timer = torch.tensor([0.03, 0.05, 1.0])
        for dt, due in ((0.02, [False, False, False]), (0.04, [True, False, False])):
            env = stand_in(timer, dt=dt)
            got = TrackingSim._push(env, torch.zeros(3, dtype=torch.bool))
            self.assertEqual(got.tolist(), due)
            self.assertTrue(torch.equal(env.push_timer[~got], (timer - dt)[~got]))

    def test_kicks_stay_within_the_half_ranges(self):
        env = stand_in(torch.zeros(4096), seed=2)
        vel0 = world_velocity(env)
        due = TrackingSim._push(env, torch.zeros(4096, dtype=torch.bool))
        self.assertTrue(due.all())
        d = world_velocity(env) - vel0
        self.assertTrue((d.abs() <= torch.tensor(HALF) + 1e-5).all())
        self.assertTrue((d.abs().amax(0) > 0.9 * torch.tensor(HALF)).all())
        self.assertTrue(((env.push_timer >= INTERVAL[0]) & (env.push_timer <= INTERVAL[1])).all())


class TestTimer(unittest.TestCase):
    def test_the_redraw_takes_a_new_interval_in_the_masked_worlds_only(self):
        # as reset and reset_masked call it for the worlds whose episode starts
        timer = torch.tensor([-0.01, 0.5, 2.0, 0.0])
        mask = torch.tensor([True, False, True, False])
        env = stand_in(timer)
        torch.manual_seed(5)
        TrackingSim._redraw_push_timer(env, mask)
        torch.manual_seed(5)
        lo, hi = INTERVAL
        want = lo + (hi - lo) * torch.rand(4)
        self.assertTrue(torch.equal(env.push_timer[mask], want[mask]))
        self.assertTrue(torch.equal(env.push_timer[~mask], timer[~mask]))


class TestCommandLine(unittest.TestCase):
    def call(self, *extra):
        cmd = [sys.executable, "-m", "blinkmimic2.train", *extra]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_incomplete_or_invalid_push_options_are_refused(self):
        for extra, msg in ((("--push_interval_s", "1", "3"), "together"),
                           (("--push_velocity", *map(str, HALF)), "together"),
                           (("--push_interval_s", "3", "1", "--push_velocity", *map(str, HALF)), "0 < LO <= HI"),
                           (("--push_interval_s", "0", "1", "--push_velocity", *map(str, HALF)), "0 < LO <= HI"),
                           (("--push_interval_s", "1", "inf", "--push_velocity", *map(str, HALF)), "0 < LO <= HI < inf"),
                           (("--push_interval_s", "nan", "3", "--push_velocity", *map(str, HALF)), "0 < LO <= HI < inf"),
                           (("--push_interval_s", "1", "3", "--push_velocity", "0.5", "0.5", "-0.2", "0", "0", "0"),
                            "finite and not negative"),
                           (("--push_interval_s", "1", "3", "--push_velocity", "nan", "0.5", "0.2", "0", "0", "0"),
                            "finite and not negative"),
                           (("--push_interval_s", "1", "3", "--push_velocity", "0.5", "0.5", "0.2", "inf", "0", "0"),
                            "finite and not negative")):  # fmt: skip
            r = self.call(*extra)
            self.assertEqual(r.returncode, 2, extra)
            self.assertIn(msg, r.stderr)


if __name__ == "__main__":
    unittest.main()
