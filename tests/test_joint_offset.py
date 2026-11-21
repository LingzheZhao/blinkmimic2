"""Default joint position offset ranges per joint name pattern (DomainRandCfg.joint_offset, train.py --joint_offset): the
sim context keeps its layout, the ranges reach the per-episode draws, and train.py's refusals.

train.py builds the simulator and trains once its options are parsed, so only its refusals run here; a training with
other ranges needs a GPU."""

import dataclasses
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from blinkmimic2.dr import DomainRandCfg, DomainRandomizer, XiLayout, _match
from blinkmimic2.robot import get_robot

ROBOT = get_robot("mjlab_g1")
ANKLES = {"left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"}
RANGES = {".*": 0.01, ".*_ankle_.*": 0.1}


def with_ranges(ranges: dict):
    return dataclasses.replace(ROBOT, dr=dataclasses.replace(ROBOT.dr, joint_offset=ranges))


class TestRanges(unittest.TestCase):
    def test_the_default_is_0_01_rad_on_every_joint(self):
        self.assertEqual(DomainRandCfg().joint_offset, {".*": 0.01})
        self.assertEqual(_match(ROBOT.dr.joint_offset, ROBOT.joint_names), [0.01] * ROBOT.num_joints)

    def test_the_last_matching_pattern_gives_the_range(self):
        half = _match(RANGES, ROBOT.joint_names)
        self.assertEqual({n for n, h in zip(ROBOT.joint_names, half) if h == 0.1}, ANKLES)
        self.assertTrue(all(h == 0.01 for n, h in zip(ROBOT.joint_names, half) if n not in ANKLES))
        self.assertEqual(_match({".*_ankle_.*": 0.1, ".*": 0.01}, ROBOT.joint_names), [0.01] * ROBOT.num_joints)

    def test_the_sim_context_keeps_its_layout(self):
        base, wide = XiLayout(ROBOT, "cpu"), XiLayout(with_ranges(RANGES), "cpu")
        self.assertEqual(wide.dim, base.dim)
        self.assertEqual(wide.slices, base.slices)
        self.assertTrue(torch.equal(wide.center, base.center))
        # only the normalization of the ankle offsets changes: offset / half-range stays in [-1, 1]
        sl = base.slices["joint_offset"]
        ankle = torch.tensor([n in ANKLES for n in ROBOT.joint_names])
        self.assertTrue(torch.equal(wide.scale[sl][~ankle], base.scale[sl][~ankle]))
        torch.testing.assert_close(wide.scale[sl][ankle], torch.full((4,), 0.1))
        outside = torch.ones(base.dim, dtype=torch.bool)
        outside[sl] = False
        self.assertTrue(torch.equal(wide.scale[outside], base.scale[outside]))

    def test_the_per_episode_draws_take_the_ranges(self):
        robot = with_ranges(RANGES)
        layout = XiLayout(robot, "cpu")
        half = torch.tensor(_match(robot.dr.joint_offset, robot.joint_names))
        rand = SimpleNamespace(eval_mode=False, layout=layout, J=robot.num_joints, joint_offset_half=half)
        xi, mask = torch.zeros(20000, layout.dim), torch.ones(20000, dtype=torch.bool)
        DomainRandomizer.reset_masked(rand, xi, mask)
        off = xi[:, layout.slices["joint_offset"]].abs().amax(0)
        self.assertTrue((off <= half).all())
        self.assertTrue((off > 0.99 * half).all())


class TestCommandLine(unittest.TestCase):
    def call(self, *extra):
        return subprocess.run([sys.executable, "-m", "blinkmimic2.train", *extra], capture_output=True, text=True)

    def test_malformed_ranges_are_refused(self):
        for items in (["ankle"], [".*=-0.1"], [".*=nan"], [".*=0.01", ".*=0.1"], ["(=0.1"], ["=0.1"]):
            r = self.call("--joint_offset", *items)
            self.assertEqual(r.returncode, 2, items)
            self.assertIn("needs PATTERN=HALF", r.stderr)

    def test_a_pattern_matching_no_joint_is_refused(self):
        r = self.call("--joint_offset", ".*=0.01", ".*_ancle_.*=0.1")
        self.assertEqual(r.returncode, 2)
        self.assertIn(".*_ancle_.* matches no joint of mjlab_g1", r.stderr)


if __name__ == "__main__":
    unittest.main()
