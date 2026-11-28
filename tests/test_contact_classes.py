"""The contact wrench recording's split of MuJoCo's constraint force, and the contact classes of recorded controls
(CPU; the trainer's mjlab G1 model)."""

import os
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from blinkmimic2.rigid_body import trainer_model  # noqa: E402
from blinkmimic2.robot import get_robot  # noqa: E402
from contact_wrench_recording import Bodies, substep_loads  # noqa: E402
from rigid_body_comparison import random_states  # noqa: E402
from world_model_error_by_contact_class import FEET, contact_classes  # noqa: E402

ROBOT = get_robot("mjlab_g1")
MODEL = trainer_model(ROBOT)


class TestConstraintSplit(unittest.TestCase):
    def test_split_is_exact_and_covers_every_part(self):
        """qfrc_constraint = J^T W_floor + tau_self + tau_limit at rounding, on states with floor contacts, robot-robot
        contacts and joint-limit forces (some joints placed past their range)."""
        m, d = MODEL, mujoco.MjData(MODEL)
        B = Bodies(m)
        rng = np.random.default_rng(3)
        qpos, qvel, _ = random_states(m, rng, 60, ROBOT.com_range)
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
        past = rng.random((60, m.nu)) < 0.2
        qpos[:, 7:] = np.where(past, np.where(rng.random((60, m.nu)) < 0.5, lo - 0.05, hi + 0.05), qpos[:, 7:])
        f6, jp, jr = np.zeros(6), np.zeros((3, m.nv)), np.zeros((3, m.nv))
        seen = {"floor": 0, "self": 0, "limit": 0}
        for s in range(len(qpos)):
            mujoco.mj_resetData(m, d)
            d.qpos[:], d.qvel[:] = qpos[s], qvel[s]
            mujoco.mj_forward(m, d)
            Wf, Ws, tf, ts, tl, qc = substep_loads(m, d, B, f6, jp, jr)
            scale = max(1.0, float(np.abs(qc).max()))
            self.assertLess(float(np.abs(qc - tf - ts - tl).max()) / scale, 1e-10)
            # robot-robot contacts act on two bodies with opposite forces
            np.testing.assert_allclose(Ws[:, :3].sum(0), 0.0, atol=1e-9 * scale)
            seen["floor"] += int(np.abs(Wf).max() > 0)
            seen["self"] += int(np.abs(Ws).max() > 0)
            seen["limit"] += int(np.abs(tl).max() > 0)
        for k, n in seen.items():
            self.assertGreater(n, 0, k)


class TestContactClasses(unittest.TestCase):
    def test_flags_and_combinations(self):
        nb, nv, k = 30, 35, 4
        feet = [7, 13]
        N = 6
        wf, ws, tl = np.zeros((N, k, nb, 6)), np.zeros((N, k, nb, 6)), np.zeros((N, k, nv))
        wf[1, 2, 7, 2] = 300.0  # 1: one foot on the floor
        wf[2, 0, 7, 2], wf[2, 3, 20, 2] = 200.0, 50.0  # 2: a foot and a hand
        wf[3, 1, 0, 2], tl[3, 0, 12] = 100.0, 5.0  # 3: pelvis on the floor and a joint limit
        ws[4, 2, 18, 0], ws[4, 2, 24, 0], tl[4, 3, 30] = 7.0, -7.0, 1.0  # 4: robot-robot contact and a joint limit
        wf[5, 0, 13, 2] = 1e-9  # 5: below the threshold: no contact
        f, keys = contact_classes(wf, ws, tl, feet, threshold=1e-6)
        np.testing.assert_array_equal(f["joint_limit"], [0, 0, 0, 1, 1, 0])
        np.testing.assert_array_equal(f["robot_robot"], [0, 0, 0, 0, 1, 0])
        np.testing.assert_array_equal(f["floor_non_foot"], [0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(f["feet_only_floor"], [0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(f["no_contact"], [1, 0, 0, 0, 0, 1])
        self.assertEqual(list(keys), ["no_contact", "feet_only_floor", "floor_non_foot", "joint_limit+floor_non_foot",
                                      "joint_limit+robot_robot", "no_contact"])

    def test_feet_are_the_ankle_roll_links(self):
        names = Bodies(MODEL).names
        self.assertEqual([names[i] for i in [names.index(n) for n in FEET]], list(FEET))
        self.assertEqual(sum(1 for b in names if "ankle_roll" in b), 2)


if __name__ == "__main__":
    unittest.main()
