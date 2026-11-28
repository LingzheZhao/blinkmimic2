"""tools/world_model_error.py on the CPU with synthetic records and a small world model (the class
WorldModel): the segments a record gives (recorded, finite, not across a trial's end, not past a first-row clip), the
previous actions, the low-posture and get-up splits, the evaluation simulator's xi, the open-loop error in train.py's
normalization, and the world model built as train.py builds it."""

import os
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import world_model_error as W  # noqa: E402
from synthetic import ROBOT  # noqa: E402

from blinkmimic2.dr import XiLayout  # noqa: E402
from blinkmimic2.mdp import flatten_state, unflatten_state  # noqa: E402
from blinkmimic2.networks import (WorldModel, build_world_model, world_model_error_floor,  # noqa: E402
                                  world_model_error_scale, world_model_normalized_error)

J = ROBOT.num_joints
D = 13 + 2 * J  # flattened state
DE = 12 + 2 * J  # components of the state error


def upright_states(n, z=0.8):
    s = np.zeros((n, D), np.float32)
    s[:, 2] = z
    s[:, 3] = 1.0
    return s


class TestSegments(unittest.TestCase):
    def test_indices_ends_and_clip_limit(self):
        C, T, H = 10, 3, 3
        state = np.stack([upright_states(T) for _ in range(C)])
        start = upright_states(T)
        available = np.ones((C, T), bool)
        done = np.zeros((C, T), bool)
        available[5, 1] = False  # trial 1: state after control 5 missing
        state[5, 1] = np.nan
        done[6, 2] = True  # trial 2 ends at control 6 (its state after 6 recorded), later states missing
        available[7:, 2] = False
        state[7:, 2] = np.nan
        action = np.arange(C * T * J, dtype=np.float32).reshape(C, T, J)
        trial, first, seq, prev = W.segments(start, state, action, available, done, H)
        got = {i: sorted(first[trial == i].tolist()) for i in range(T)}
        self.assertEqual(got[0], list(range(0, C - H + 1)))  # t = 0 .. 7
        # a segment of trial 1 needs the states before controls t .. t+3 (seq[t .. t+3]); seq[6] is missing
        self.assertEqual(got[1], [0, 1, 2, 7])
        # trial 2: done at control 6 may be the segment's last control (t = 4), not an earlier one
        self.assertEqual(got[2], [0, 1, 2, 3, 4])
        self.assertTrue(np.array_equal(prev[0], np.zeros((T, J))))
        self.assertTrue(np.array_equal(prev[4], action[3]))
        self.assertTrue(np.array_equal(seq[0], start) and np.array_equal(seq[3, 0], state[2, 0]))
        trial, first, _, _ = W.segments(start, state, action, available, done, H, last_control=np.array([5, 9, 9]))
        self.assertEqual(sorted(first[trial == 0].tolist()), [0, 1, 2, 3])  # t + H - 1 <= 5

    def test_low_posture_states_and_get_up_windows(self):
        s = upright_states(4)
        s[1, 2] = 0.49
        s[2, 3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # rolled 90 degrees: up-axis z 0
        s[3, 3:7] = [np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0]  # 45 degrees: up-axis z 0.707
        self.assertEqual(W.low_posture(s).tolist(), [False, True, True, False])
        trial, first = np.array([0, 0, 0, 1, 1]), np.array([4, 5, 20, 5, 30])
        per = [[dict(window_controls=[5, 10]), dict(window_controls=[18, 25])], [dict(window_controls=[0, 2])]]
        self.assertEqual(W.in_get_up_windows(trial, first, per).tolist(), [False, True, True, False, False])

    def test_evaluation_xi(self):
        layout = XiLayout(ROBOT, "cpu")
        xi = W.evaluation_xi(layout)
        self.assertEqual(xi.shape, (layout.dim,))
        self.assertEqual(float(xi[layout.slices["friction"]].item()), 1.0)
        rest = torch.cat([xi[sl] for n, sl in layout.slices.items() if n != "friction"])
        self.assertTrue(bool((rest == 0).all()))


class TestOpenLoop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        layout = XiLayout(ROBOT, "cpu")
        self.model = WorldModel(J, layout.dim, use_prev_action=True, hidden=(32, 32), structured=True)
        with torch.no_grad():
            for p in self.model.net.parameters():
                p.normal_(0, 0.05)
            self.model.out_norm.var.fill_(1e-4)
            self.model.inc_norm.var.fill_(4e-6)  # std 2e-3: below the position floors 3e-3, above none of the others
        self.xi_n = layout.normalize(W.evaluation_xi(layout)).unsqueeze(0)

    def record(self, C=6, T=2):
        """States the model itself predicts, from upright starts with random actions."""
        g = torch.Generator().manual_seed(1)
        action = torch.randn(C, T, J, generator=g) * 0.3
        S = unflatten_state(torch.as_tensor(upright_states(T)))
        start = flatten_state(S)
        states, prev = [], torch.zeros(T, J)
        with torch.no_grad():
            for c in range(C):
                S = self.model.step(S, action[c], prev, self.xi_n.expand(T, -1))
                states.append(flatten_state(S))
                prev = action[c]
        return start.numpy(), torch.stack(states).numpy(), action.numpy()

    def test_zero_error_on_its_own_states_and_one_perturbed_step(self):
        start, state, action = self.record()
        C, T = state.shape[:2]
        trial, first, seq, prev = W.segments(start, state, action, np.ones((C, T), bool), np.zeros((C, T), bool), 3)
        e, first_groups, last_groups = W.open_loop(self.model, trial, first, seq, action, prev, self.xi_n, 3)
        self.assertEqual(e.shape, (len(trial), 3))
        self.assertEqual(first_groups.shape, (len(trial), 6))
        self.assertLess(float(np.abs(e).max()), 1e-6)
        # the open loop continues from its own prediction: a change of one recorded state is seen at that step only
        d = 0.01
        seq2 = seq.copy()
        seq2[2, 0, 13] += d  # joint 0 of trial 0 after control 1
        e2, _, _ = W.open_loop(self.model, trial, first, seq2, action, prev, self.xi_n, 3)
        w = int(np.nonzero((trial == 0) & (first == 0))[0][0])
        scale = max(2e-3, 3e-3)  # the joint-position lower bound
        self.assertLess(abs(e2[w, 0]), 1e-6)
        self.assertAlmostEqual(float(e2[w, 1]), d**2 / scale**2 / DE, delta=1e-3 * d**2 / scale**2 / DE)
        self.assertLess(abs(e2[w, 2]), 1e-6)

    def test_stride_and_error_scale(self):
        start, state, action = self.record()
        C, T = state.shape[:2]
        ones, zeros = np.ones((C, T), bool), np.zeros((C, T), bool)
        trial, first, seq, prev = W.segments(start, state, action, ones, zeros, 2)
        t2, f2, _, _ = W.segments(start, state, action, ones, zeros, 2, stride=2)
        self.assertEqual(sorted(set(f2.tolist())), [t for t in sorted(set(first.tolist())) if t % 2 == 0])
        seq2 = seq.copy()
        seq2[2, 0, 13] += 0.01
        own, _, _ = W.open_loop(self.model, trial, first, seq2, action, prev, self.xi_n, 2)
        same, _, _ = W.open_loop(self.model, trial, first, seq2, action, prev, self.xi_n, 2,
                                 scale=world_model_error_scale(self.model))
        self.assertTrue(np.array_equal(own, same))
        double, _, _ = W.open_loop(self.model, trial, first, seq2, action, prev, self.xi_n, 2,
                                   scale=2 * world_model_error_scale(self.model))
        self.assertTrue(np.allclose(double, own / 4, rtol=1e-5, atol=1e-12))

    def test_state_groups(self):
        start, state, action = self.record()
        C, T = state.shape[:2]
        trial, first, seq, prev = W.segments(start, state, action, np.ones((C, T), bool), np.zeros((C, T), bool), 3)
        d = 0.01
        seq2 = seq.copy()
        seq2[3, 0, 13] += d  # joint 0 of trial 0 after control 2: the last control of the segment from control 0
        seq2[1, 0, 2] += d  # root height of trial 0 after control 0: the first control of that segment
        e, first_groups, last_groups = W.open_loop(self.model, trial, first, seq2, action, prev, self.xi_n, 3)
        w = int(np.nonzero((trial == 0) & (first == 0))[0][0])
        names = [n for n, _ in W.STATE_GROUPS]
        # the root position error is taken in the heading frame: a change of the height is its z component
        expected_first = np.zeros(6)
        expected_first[names.index("root_position")] = d**2 / 3e-3**2 / 3
        expected_last = np.zeros(6)
        expected_last[names.index("joint_positions")] = d**2 / 3e-3**2 / J
        self.assertTrue(np.allclose(first_groups[w], expected_first, rtol=1e-3, atol=1e-5), first_groups[w])
        self.assertTrue(np.allclose(last_groups[w], expected_last, rtol=1e-3, atol=1e-5), last_groups[w])
        # the groups' means weighted by their widths give the total error
        widths = np.array([width(J) for _, width in W.STATE_GROUPS])
        self.assertAlmostEqual(float((first_groups[w] * widths).sum() / widths.sum()), float(e[w, 0]), places=5)
        out = W.summary(e, np.arange(len(e)) == w, first_groups, last_groups)
        self.assertAlmostEqual(out["state_groups"]["joint_positions"]["error_after_3_controls_mean"],
                               float(last_groups[w, names.index("joint_positions")]), places=6)

    def test_the_trainers_error_form(self):
        S = unflatten_state(torch.as_tensor(upright_states(3)))
        S1 = tuple(x.clone() for x in S)
        S1[5][:, 2] += 0.1
        std = torch.maximum(self.model.inc_norm.std(), world_model_error_floor(J))
        ref = ((self.model.state_error(S1, S) / std) ** 2).mean(-1)  # train.py _wm_err, per sample and control
        self.assertTrue(torch.allclose(world_model_normalized_error(self.model, S1, S), ref))
        self.assertAlmostEqual(float(ref[0]), (0.1 / 5e-2) ** 2 / DE, places=5)

    def test_world_model_as_train_py_builds_it(self):
        model = build_world_model(ROBOT, XiLayout(ROBOT, "cpu").dim, {"wm_hidden": "64,64"})
        self.assertTrue(model.structured)
        self.assertFalse(model.contact_modes)
        self.assertEqual(model.use_prev_action, ROBOT.has_action_delay)
        self.assertEqual([m.out_features for m in model.net if isinstance(m, torch.nn.Linear)][:2], [64, 64])
        src = open(os.path.join(REPO, "src", "blinkmimic2", "train.py")).read()
        self.assertIn("wm = build_world_model(robot, layout.dim, vars(args), kin)", src)
        self.assertIn("ERR_FLOOR = world_model_error_floor(J)", src)
        self.assertEqual(world_model_error_floor(J).shape, (DE,))


class TestNonFootContact(unittest.TestCase):
    """The non-foot-contact split on hand-built states of mjlab's compiled G1 (MuJoCo C on the CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.contacts = W.Contacts(ROBOT)

    def state(self, z, rotation=(1.0, 0.0, 0.0, 0.0)):
        s = np.zeros(D, np.float64)
        s[2], s[3:7] = z, rotation
        s[13:13 + J] = ROBOT.default_joint_pos_list()
        return s

    def test_airborne_lying_and_feet_only(self):
        lying = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)  # rolled 90 degrees about x
        self.assertEqual(self.contacts(np.stack([self.state(2.0), self.state(0.12, lying)])).tolist(), [False, True])
        # the standing pose lowered until its first penetrating contact: only the feet touch the floor
        import mujoco

        m, d = self.contacts.m, self.contacts.d
        for z in np.arange(1.2, 0.3, -0.002):
            s = self.state(z)
            d.qpos[self.contacts.free_adr:self.contacts.free_adr + 7] = s[:7]
            d.qpos[self.contacts.q_adr] = s[13:13 + J]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            touching = [c for c in d.contact[: d.ncon] if c.dist < 0]
            if touching:
                break
        self.assertTrue(touching)
        feet = {m.body("robot/" + f).id for f in W.FEET}
        robot_geoms = [g for c in touching for g in (c.geom1, c.geom2) if not self.contacts.floor[g]]
        self.assertTrue(robot_geoms and all(m.geom_bodyid[g] in feet for g in robot_geoms))
        self.assertEqual(self.contacts(s[None]).tolist(), [False])


if __name__ == "__main__":
    unittest.main()
