"""DynaMimic's link inputs to the world model (train.py --wm_link_inputs, networks.link_features) on the CPU: the
features against MuJoCo's link positions and velocities on hand-built states, their invariance to the base's heading
and planar position, the option's absence leaving the world model's parameters and the random draws unchanged, a
checkpoint round trip, and tools/world_model_error.py reading such a checkpoint."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import world_model_error as W  # noqa: E402
from synthetic import ROBOT  # noqa: E402

from blinkmimic2.dr import XiLayout  # noqa: E402
from blinkmimic2.kinematics import Kinematics  # noqa: E402
from blinkmimic2.mdp import flatten_state  # noqa: E402
from blinkmimic2.networks import WorldModel, build_world_model, link_features  # noqa: E402
from blinkmimic2.rotations import mat_to_quat, quat_to_mat  # noqa: E402

J = ROBOT.num_joints
LINKS = ["pelvis", "left_knee_link", "left_ankle_roll_link", "right_wrist_yaw_link", "torso_link"]
KIN = Kinematics(ROBOT, "cpu")
XI_DIM = XiLayout(ROBOT, "cpu").dim


def random_state(n, seed=0, yaw=None):
    g = torch.Generator().manual_seed(seed)
    quat = torch.randn(n, 4, generator=g)
    p = torch.randn(n, 3, generator=g) * 0.3 + torch.tensor([0.0, 0.0, 0.6])
    v, w = torch.randn(n, 3, generator=g), torch.randn(n, 3, generator=g)
    q, qd = torch.randn(n, J, generator=g) * 0.3, torch.randn(n, J, generator=g)
    return (p, quat_to_mat(quat), v, w, q, qd)


def about_z(S, angle, shift):
    """The states ``S`` turned by ``angle`` about the world z axis and moved by the planar ``shift``."""
    c, s = np.cos(angle), np.sin(angle)
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    p, R, v, w, q, qd = S
    return (p @ Rz.T + torch.tensor([shift[0], shift[1], 0.0]), Rz @ R, v @ Rz.T, w @ Rz.T, q, qd)


class TestLinkFeatures(unittest.TestCase):
    def test_against_mujoco(self):
        m = mujoco.MjModel.from_xml_path(ROBOT.mjcf_path)
        d = mujoco.MjData(m)
        S = random_state(3, seed=1)
        index = [KIN.body_names.index(b) for b in LINKS]
        f = link_features(KIN, index, S).view(3, len(LINKS), 7).double()
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        qa, va = m.jnt_qposadr[free], m.jnt_dofadr[free]
        p, R, v, w, q, qd = (x.double() for x in S)
        for i in range(3):
            d.qpos[:] = 0
            d.qvel[:] = 0
            d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = p[i].numpy(), mat_to_quat(R[i:i + 1])[0].numpy()
            d.qvel[va:va + 3], d.qvel[va + 3:va + 6] = v[i].numpy(), (R[i].T @ w[i]).numpy()
            for j, name in enumerate(ROBOT.joint_names):
                jid = m.joint(name).id
                d.qpos[m.jnt_qposadr[jid]], d.qvel[m.jnt_dofadr[jid]] = q[i, j], qd[i, j]
            mujoco.mj_forward(m, d)
            fx, fy = R[i, 0, 0], R[i, 1, 0]
            n = float(np.hypot(float(fx), float(fy)))
            H = np.array([[fx / n, -fy / n, 0], [fy / n, fx / n, 0], [0, 0, 1]])
            for k, name in enumerate(LINKS):
                b = m.body(name).id
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)  # body frame origin: [angular, linear], world frame
                expected = np.concatenate(([d.xpos[b][2]], H.T @ (d.xpos[b] - p[i].numpy()), H.T @ vel[3:]))
                np.testing.assert_allclose(f[i, k].numpy(), expected, atol=2e-4, err_msg=f"{name} of state {i}")

    def test_invariant_to_heading_and_planar_position(self):
        S = random_state(4, seed=2)
        index = [KIN.body_names.index(b) for b in LINKS]
        f0 = link_features(KIN, index, S)
        for angle, shift in ((0.7, (1.0, -2.0)), (-2.5, (0.0, 3.0))):
            f1 = link_features(KIN, index, about_z(S, angle, shift))
            self.assertTrue(torch.allclose(f0, f1, atol=1e-4), (f0 - f1).abs().max())


class TestMirror(unittest.TestCase):
    def test_support_links_are_mirror_symmetric_bodies_with_joints(self):
        from blinkmimic2.g1.g1_constants import SUPPORT_LINKS

        self.assertEqual(len(SUPPORT_LINKS), 18)
        partner = lambda n: n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")  # noqa: E731
        self.assertEqual(sorted(partner(n) for n in SUPPORT_LINKS), sorted(SUPPORT_LINKS))
        self.assertTrue(all(n in KIN.body_names for n in SUPPORT_LINKS))

    def test_features_of_the_mirrored_state_are_the_mirrored_features(self):
        # train.py fits the world model on mirrored windows (mirror.state); the inputs are computed from the mirrored
        # state, so the link features must be those of the partner links with the lateral components negated
        from blinkmimic2.g1.g1_constants import SUPPORT_LINKS
        from blinkmimic2.mirror import Mirror

        mirror = Mirror(ROBOT, XiLayout(ROBOT, "cpu").slices, "cpu")
        links = list(SUPPORT_LINKS)
        partner = lambda n: n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")  # noqa: E731
        index = [KIN.body_names.index(b) for b in links]
        partner_index = [KIN.body_names.index(partner(b)) for b in links]
        S = random_state(4, seed=7)
        mirrored = link_features(KIN, index, mirror.state(S)).view(4, len(links), 7)
        expected = link_features(KIN, partner_index, S).view(4, len(links), 7).clone()
        expected[..., 2] *= -1  # lateral position relative to the base
        expected[..., 5] *= -1  # lateral linear velocity
        self.assertTrue(torch.allclose(mirrored, expected, atol=1e-4), (mirrored - expected).abs().max())


class TestOption(unittest.TestCase):
    def test_absent_leaves_parameters_and_draws_unchanged(self):
        torch.manual_seed(5)
        a = build_world_model(ROBOT, XI_DIM, {"wm_hidden": "64,64"}, KIN)
        after_a = torch.rand(4)
        torch.manual_seed(5)
        b = WorldModel(J, XI_DIM, use_prev_action=ROBOT.has_action_delay, hidden=(64, 64), structured=True, dt=0.02)
        after_b = torch.rand(4)
        self.assertTrue(torch.equal(after_a, after_b))
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(list(sa), list(sb))
        self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))
        self.assertEqual(a.link_bodies, ())
        with_links = build_world_model(ROBOT, XI_DIM, {"wm_hidden": "64,64", "wm_link_inputs": LINKS}, KIN)
        self.assertEqual(with_links.in_dim, a.in_dim + 7 * len(LINKS))
        with self.assertRaises(ValueError):
            build_world_model(ROBOT, XI_DIM, {"wm_hidden": "64,64", "wm_link_inputs": LINKS})

    def test_inputs_hold_the_link_features_last(self):
        model = build_world_model(ROBOT, XI_DIM, {"wm_hidden": "64,64", "wm_link_inputs": LINKS}, KIN)
        S = random_state(2, seed=3)
        a, xi = torch.zeros(2, J), torch.zeros(2, XI_DIM)
        x = model.inputs(S, a, a, xi)
        self.assertEqual(x.shape, (2, model.in_dim))
        index = [KIN.body_names.index(b) for b in LINKS]
        self.assertTrue(torch.equal(x[:, -7 * len(LINKS):], link_features(KIN, index, S)))

    def test_train_py_records_the_option_only_when_given(self):
        def train(log_dir, *flags):
            cmd = [sys.executable, "-m", "blinkmimic2.train", "--log_dir", log_dir, *flags]
            return subprocess.run(cmd, capture_output=True, text=True)

        with tempfile.TemporaryDirectory() as d:
            missing = ["--bones_list", os.path.join(d, "missing.txt")]
            r = train(os.path.join(d, "a"), "--wm_link_inputs", *LINKS, *missing)
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            r = train(os.path.join(d, "b"), *missing)
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            a = json.load(open(os.path.join(d, "a", "args.json")))
            b = json.load(open(os.path.join(d, "b", "args.json")))
        self.assertEqual(a["wm_link_inputs"], LINKS)
        self.assertNotIn("wm_link_inputs", b)
        self.assertEqual({k: v for k, v in a.items() if k not in ("wm_link_inputs", "log_dir")},
                         {k: v for k, v in b.items() if k != "log_dir"})


class TestCheckpoint(unittest.TestCase):
    def test_round_trip_and_the_error_tool(self):
        args = {"robot": ROBOT.name, "wm_hidden": "64,64", "wm_link_inputs": LINKS}
        torch.manual_seed(6)
        model = build_world_model(ROBOT, XI_DIM, args, KIN)
        with torch.no_grad():
            for prm in model.net.parameters():
                prm.normal_(0, 0.05)
            model.in_norm.mean.normal_(0, 0.1)
        S = random_state(5, seed=4)
        a = torch.randn(5, J) * 0.2
        xi_n = XiLayout(ROBOT, "cpu").normalize(W.evaluation_xi(XiLayout(ROBOT, "cpu"))).expand(5, -1)
        with torch.no_grad():
            expected = model.step(S, a, a, xi_n)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "last.pt")
            torch.save({"wm": model.state_dict(), "args": args, "iteration": 1}, path)
            loaded, ck, robot, layout = W.load_world_model(path)
        self.assertEqual(loaded.link_bodies, tuple(LINKS))
        with torch.no_grad():
            got = loaded.step(S, a, a, xi_n)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, got)))
        # the error tool's open loop over a record of the model's own states: zero error
        T, C = 5, 4
        states, prev, Sc = [], torch.zeros(T, J), S
        actions = torch.randn(C, T, J) * 0.2
        with torch.no_grad():
            for c in range(C):
                Sc = loaded.step(Sc, actions[c], prev, xi_n)
                states.append(flatten_state(Sc))
                prev = actions[c]
        start, state = flatten_state(S).numpy(), torch.stack(states).numpy()
        trial, first, seq, prv = W.segments(start, state, actions.numpy(), np.ones((C, T), bool), np.zeros((C, T), bool), 2)
        e, _, _ = W.open_loop(loaded, trial, first, seq, actions.numpy(), prv, xi_n[:1], 2)
        self.assertEqual(e.shape, (len(trial), 2))
        self.assertLess(float(np.abs(e).max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
