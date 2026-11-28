"""Comparison of blinkmimic2.rigid_body with MuJoCo C on the trainer's mjlab G1 model (CPU, float64)."""

import os
import sys
import unittest

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from blinkmimic2.rigid_body import RigidBody, from_mujoco, from_trainer, to_trainer, trainer_model  # noqa: E402
from blinkmimic2.robot import get_robot  # noqa: E402
from rigid_body_comparison import compare, ctrl_for, mujoco_reference, random_states  # noqa: E402

ROBOT = get_robot("mjlab_g1")
MODEL = trainer_model(ROBOT)


class TestRigidBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = MODEL
        cls.rb = RigidBody(m, ROBOT.joint_names)
        rng = np.random.default_rng(1)
        cls.qpos, cls.qvel, cls.com = random_states(m, rng, 24, ROBOT.com_range)
        act_joint = np.array([m.jnt_dofadr[m.actuator_trnid[k, 0]] - 6 for k in range(m.nu)])
        ctrl = ctrl_for(m, rng, cls.qpos, act_joint)
        cls.ref = mujoco_reference(m, cls.qpos, cls.qvel, ctrl, cls.com, m.body("robot/" + ROBOT.com_body).id)
        cls.ctrl_j = np.zeros((len(cls.qpos), m.nu))
        cls.ctrl_j[:, act_joint] = ctrl
        cls.res = compare(cls.rb, m, cls.qpos, cls.qvel, cls.ctrl_j, cls.com, cls.ref, torch.float64)

    def test_model_constants(self):
        rb = self.rb
        self.assertEqual((rb.nb, rb.nv), (30, 35))
        self.assertAlmostEqual(rb.total_mass, float(MODEL.body_mass[2:].sum()))
        self.assertTrue(bool((rb.armature[6:] > 0).all()) and float(rb.armature[:6].abs().max()) == 0.0)
        self.assertEqual(int(MODEL.opt.integrator), int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST))
        torch.testing.assert_close(rb.kp * torch.tensor(ROBOT.action_scale_list(), dtype=rb.dtype),
                                   0.25 * rb.f_hi)  # BeyondMimic's action scale 0.25 effort / kp

    def test_comparison_with_mujoco(self):
        r = self.res
        for k in ("M", "qfrc_bias", "jacp", "jacr", "qfrc_actuator", "physics_substep_qvel", "physics_substep_pos", "physics_substep_joint_pos", "tree_M",
                  "tree_bias_minus_wrench_vs_dense", "tree_cholesky_physics_substep_qvel", "tree_ltdl_physics_substep_qvel",
                  "tree_columns_physics_substep_qvel"):
            self.assertLess(r[k]["max_rel"], 1e-11, k)
        self.assertLess(r["physics_substep_rot_max_abs"], 1e-12)
        self.assertEqual(r["qfrc_passive"]["max_abs"], 0.0)
        # the states saturate actuators, and keeping kd for them would be visible: MuJoCo drops clamped actuators
        self.assertGreater(r["physics_substep_qvel_if_clamped_kd_kept"]["max_rel"], 1e-4)

    def test_conversions(self):
        p, R, q, nu = from_mujoco(torch.tensor(self.qpos), torch.tensor(self.qvel))
        S = to_trainer(p, R, q, nu)
        for a, b in zip(from_trainer(S), (p, R, q, nu)):
            torch.testing.assert_close(a, b)
        # world angular velocity of the trainer = R w_local
        torch.testing.assert_close(S[3], (R @ nu[:, 3:6].unsqueeze(-1)).squeeze(-1))

    def test_wrench_force_is_jacobian_transpose(self):
        rb = self.rb
        p, R, q, nu = from_mujoco(torch.tensor(self.qpos), torch.tensor(self.qvel))
        kin = rb.kinematics(p, R, q)
        Jp, Jr = rb.jacobians(kin)
        links = rb.index(["left_ankle_roll_link", "right_wrist_yaw_link"])
        w = torch.randn(len(p), 2, 6, dtype=rb.dtype)
        tau = rb.wrench_force(Jp, Jr, links, w)
        # virtual power: tau . nu = sum_l F . v_l + n . w_l
        v = torch.einsum("bldi,bd->bli", Jp[:, links], nu)
        om = torch.einsum("bldi,bd->bli", Jr[:, links], nu)
        torch.testing.assert_close((tau * nu).sum(-1), (w[..., :3] * v + w[..., 3:] * om).sum((-1, -2)))


if __name__ == "__main__":
    unittest.main()
