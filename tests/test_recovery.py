"""Recovery training (src/blinkmimic2/recovery.py, RECOVERY.md) on the CPU: the row tables of a generated get-up
library, the regime rule of one control, the start draws, the termination of a recovering control, the value bound,
the reward and critic features with and without recovery, the replay buffer's fields, and the simulator's step and
resets driven through stand-ins for the simulator, and train.py's recovery options with their refusals and presets.

train.py builds the simulator and trains once its options are checked, so only the checks run here (in a subprocess);
the imagined step and the log columns need a GPU."""

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ROBOT, getup_library  # noqa: E402
from test_placement import Recorder  # noqa: E402

import blinkmimic2.env as env_module  # noqa: E402
from blinkmimic2 import recovery  # noqa: E402
from blinkmimic2.buffer import ReplayBuffer, fields  # noqa: E402
from blinkmimic2.env import TrackingSim, placed_state  # noqa: E402
from blinkmimic2.mdp import CONTACT_Z, MAX_STEP_REWARD, inv_dim, invariant_features  # noqa: E402
from blinkmimic2.motion import CONTROL_DT, CompactMotionLib, MotionLib  # noqa: E402
from blinkmimic2.rotations import mat_to_6d, mat_to_quat, mm3, mv3, rot_angle_sq, so3_exp, yaw_mat  # noqa: E402


class Compact32(CompactMotionLib):
    STORE_DTYPE = torch.float32  # as --motion_store fp32, the training default


def options(**changed):
    """Parsed recovery options of train.py for the generated library: hidden getup_a and getup_b, command-excluded
    also excluded; a run of 3 upright rows ends a get-up, starts at most 4 rows before their end, commands with 4
    upright rows ahead; the setting's share, scale and durations (2.0 s and 1.5 s)."""
    values = dict(recovery_share=0.5, recovery_hidden_clips=["getup_a", "getup_b"],
                  recovery_command_excluded_clips=["getup_a", "getup_b", "excluded"], recovery_low_pelvis=[0.5, 0.5],
                  recovery_rows=[3, 4, 4], recovery_reward_scale=2.5, recovery_seconds=[2.0, 1.5])
    values.update(changed)
    return argparse.Namespace(**values)


SETTINGS = recovery.settings_from_args(options())
_, MOTION, MDP_ = getup_library()
MDP_.feet = [ROBOT.tracked_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]  # as train.py
REC = recovery.build(MOTION, SETTINGS)
J = ROBOT.num_joints


def state_of(rows, z=None, xy=None, grad=False):
    """The reference states of ``rows``, with the pelvis height ``z`` and planar position ``xy`` when given."""
    S = tuple(x.clone() for x in MOTION.ref_state(torch.tensor(rows)))
    if z is not None:
        S[0][:, 2] = torch.tensor(z)
    if xy is not None:
        S[0][:, :2] = torch.tensor(xy)
    if grad:
        S[0].requires_grad_(True)
    return S


ROLLED_90 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # 90 degrees about x


def tilted(rows):
    """The reference states of ``rows`` with the pelvis up axis 90 degrees from the reference's: rolled where the
    reference is upright, upright where it is not; the pelvis height is the reference's."""
    S = state_of(rows)
    S[1][:] = torch.where((S[1][:, 2, 2] > 0.5)[:, None, None], ROLLED_90, torch.eye(3))
    return S


def regime_of(hidden, recovering, errors=0, left=0):
    n = len(hidden)
    return {"hidden_frame": torch.tensor(hidden), "recovering": torch.tensor(recovering),
            "error_controls": torch.full((n,), errors), "blend_left": torch.full((n,), left),
            "command_offset": torch.zeros(n, 2)}


class TestSettings(unittest.TestCase):
    def test_the_durations_become_controls(self):
        self.assertEqual((SETTINGS.stop_controls, SETTINGS.blend_controls), (100, 75))
        self.assertEqual(SETTINGS.share, 0.5)

    def test_no_option_gives_none(self):
        # train.py leaves an option it was not given out of its arguments
        self.assertIsNone(recovery.settings_from_args(argparse.Namespace()))

    def test_refusals(self):
        cases = {"go together; missing \\['--recovery_rows', '--recovery_seconds'\\]":
                 options(recovery_rows=None, recovery_seconds=None),
                 "not a whole number": options(recovery_seconds=[2.01, 1.5]),
                 "must also be command-excluded": options(recovery_command_excluded_clips=["getup_a", "excluded"]),
                 "share 0 must lie in": options(recovery_share=0),
                 "must be positive": options(recovery_rows=[0, 4, 4]),
                 "reward scale nan": options(recovery_reward_scale=float("nan"))}
        for message, args in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                recovery.settings_from_args(args)


class TestRows(unittest.TestCase):
    def test_tables_and_counts_on_both_library_classes(self):
        end = [6] * 7 + [7, 8, 9] + [-1] * 7 + [22] * 6 + list(range(23, 30)) + [32] * 3 + list(range(33, 42))
        for cls in (MotionLib, Compact32):
            rec = recovery.build(getup_library(cls)[1], SETTINGS)
            self.assertEqual(rec.get_up_end.tolist(), end)
            self.assertEqual(rec.start_rows.tolist(), [2, 3])
            self.assertEqual(rec.command_rows.tolist(), list(range(30, 37)))
            self.assertEqual(rec.counts, dict(low_rows_of_hidden_clips=7, without_a_get_up_end_in_the_clip=3,
                                              beyond_the_longest_span=2, start_rows=2, get_ups=1, command_rows=7))

    def test_refusals(self):
        cases = {"\\['crawl'\\] are not clips": options(recovery_hidden_clips=["crawl"],
                                                       recovery_command_excluded_clips=["crawl"]),
                 "gives 0 and 7": options(recovery_hidden_clips=["getup_b"]),
                 "gives 2 and 0": options(recovery_rows=[3, 4, 20])}
        for message, args in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                recovery.build(MOTION, recovery.settings_from_args(args))


class TestAdvance(unittest.TestCase):
    def test_the_hidden_row_advances_by_one_and_holds_at_the_get_up_end(self):
        # the body on the hidden row each control; the command (row 17, lying) is not within reach at the get-up end
        reg, rows = regime_of([2], [True]), []
        for _ in range(8):
            target = min(int(reg["hidden_frame"]) + 1, 6)
            reg, row, scale, stop, switch = recovery.advance(MDP_, REC, state_of([target]), torch.tensor([17]), reg)
            rows.append(int(row))
            self.assertTrue(bool(reg["recovering"]) and not bool(switch) and not bool(stop))
            self.assertEqual(float(scale), 2.5)
        self.assertEqual(rows, [3, 4, 5, 6, 6, 6, 6, 6])
        # at the get-up end the command row's thresholds count: errors from the control that reached row 6
        self.assertEqual(int(reg["error_controls"]), 5)

    def test_the_stop_at_control_100_and_not_99(self):
        reg = regime_of([2], [True])
        for control in range(1, 101):
            row = min(int(reg["hidden_frame"]) + 1, 6)
            reg, _, _, stop, _ = recovery.advance(MDP_, REC, state_of([row], z=[2.0]), torch.tensor([35]), reg)
            self.assertEqual(int(reg["error_controls"]), control)
            self.assertEqual(bool(stop), control == 100)

    def test_the_stop_by_the_tilt_alone(self):
        # at each target row's height with the up axis 90 degrees from that row's: an error at every control
        reg = regime_of([2], [True])
        for control in range(1, 101):
            row = min(int(reg["hidden_frame"]) + 1, 6)
            reg, _, _, stop, _ = recovery.advance(MDP_, REC, tilted([row]), torch.tensor([35]), reg)
            self.assertEqual(int(reg["error_controls"]), control)
            self.assertEqual(bool(stop), control == 100)

    def test_a_non_finite_body_is_a_target_error(self):
        # the switching state of test_the_switch_needs_the_body_within_both_rows with a non-finite height or up axis
        for which in ("height", "up axis"):
            with self.subTest(which):
                S1 = state_of([6], z=[0.63])
                if which == "height":
                    S1[0][:, 2] = float("nan")
                else:
                    S1[1][:, 2, 2] = float("nan")
                reg, _, _, stop, switch = recovery.advance(MDP_, REC, S1, torch.tensor([25]),
                                                           regime_of([5], [True], errors=99))
                self.assertFalse(bool(switch))
                self.assertTrue(bool(stop) and bool(reg["recovering"]))
                self.assertEqual(int(reg["error_controls"]), 100)
                self.assertTrue(torch.equal(reg["command_offset"], torch.zeros(1, 2)))

    def test_a_control_without_error_resets_the_count(self):
        reg = regime_of([2], [True], errors=99)
        reg, _, _, stop, _ = recovery.advance(MDP_, REC, state_of([3]), torch.tensor([35]), reg)
        self.assertEqual(int(reg["error_controls"]), 0)
        self.assertFalse(bool(stop))
        reg, _, _, stop, _ = recovery.advance(MDP_, REC, state_of([4], z=[2.0]), torch.tensor([35]), reg)
        self.assertEqual(int(reg["error_controls"]), 1)

    def test_the_switch_needs_the_body_within_both_rows(self):
        # target row 6 at 0.76 m, command row 25 at 0.50 m, both upright; the thresholds are 0.25 m and 0.8
        for z, switches in ((0.63, True), (0.76, False), (0.50, False), (0.10, False)):
            with self.subTest(z=z):
                reg, row, _, stop, switch = recovery.advance(MDP_, REC, state_of([6], z=[z]), torch.tensor([25]),
                                                             regime_of([5], [True], errors=7))
                self.assertEqual(int(row), 6)
                self.assertEqual(bool(switch), switches)
                self.assertEqual(bool(reg["recovering"]), not switches)
                self.assertEqual(int(reg["error_controls"]), 0 if switches else 8)
                self.assertEqual(int(reg["blend_left"]), 75 if switches else 0)
        # the tilt alone: within both heights (0.63 m), the body rolled 90 degrees against the upright target and
        # command rows; then upright on the target row 6 against a rolled command row (row 3 at 0.76 m)
        for S1, command in ((tilted([6]), 25), (state_of([6]), 3)):
            with self.subTest(command=command):
                if command == 25:
                    S1[0][:, 2] = 0.63
                reg, _, _, _, switch = recovery.advance(MDP_, REC, S1, torch.tensor([command]),
                                                        regime_of([5], [True], errors=7))
                self.assertFalse(bool(switch))
                self.assertEqual((bool(reg["recovering"]), int(reg["error_controls"])), (True, 8))
        # before the get-up end the command row is not tested
        reg, _, _, _, switch = recovery.advance(MDP_, REC, state_of([4]), torch.tensor([17]), regime_of([3], [True]))
        self.assertFalse(bool(switch))
        self.assertEqual(int(reg["error_controls"]), 0)

    def test_the_offset_at_the_switch_is_detached_and_kept(self):
        S1 = state_of([6], z=[0.63], xy=[[0.3, -0.2]], grad=True)
        reg, _, _, _, _ = recovery.advance(MDP_, REC, S1, torch.tensor([35]), regime_of([5], [True]))
        # the pelvis minus the command row's pelvis (row 35: x = 0.5 m/s x 5 rows x 0.02 s)
        torch.testing.assert_close(reg["command_offset"], torch.tensor([[0.3 - 0.05, -0.2]]))
        self.assertFalse(reg["command_offset"].requires_grad)
        reg, row, _, _, _ = recovery.advance(MDP_, REC, state_of([36], xy=[[1.0, 1.0]]), torch.tensor([36]), reg)
        torch.testing.assert_close(reg["command_offset"], torch.tensor([[0.3 - 0.05, -0.2]]))
        self.assertEqual(int(row), 36)

    def test_the_reward_scale_of_the_switch_and_the_blend(self):
        reg, _, scale, _, switch = recovery.advance(MDP_, REC, state_of([6], z=[0.63]), torch.tensor([30]),
                                                    regime_of([5], [True]))
        self.assertTrue(bool(switch))
        scales = [float(scale)]
        for _ in range(76):  # far from the command: outside recovery nothing is tested
            reg, row, scale, stop, switch = recovery.advance(MDP_, REC, state_of([31], z=[2.0]), torch.tensor([31]), reg)
            scales.append(float(scale))
            self.assertFalse(bool(stop) or bool(switch) or bool(reg["recovering"]))
            self.assertEqual(int(row), 31)
        self.assertEqual(scales[0], 2.5)
        torch.testing.assert_close(torch.tensor(scales[1:]), torch.tensor([k / 75 for k in range(1, 76)] + [1.0]))
        self.assertEqual(int(reg["blend_left"]), 0)

    def test_regimes(self):
        rec = torch.tensor([True, True, False, False])
        left = torch.tensor([0, 3, 2, 0])
        self.assertEqual(recovery.regime(rec, left).tolist(),
                         [recovery.RECOVERY, recovery.RECOVERY, recovery.BLEND, recovery.TRACKING])


class TestStarts(unittest.TestCase):
    def test_draw_order_and_tables(self):
        n = 400
        frames = torch.randint(0, MOTION.num_frames, (n,))
        bins = MOTION.frame_bin[frames]
        torch.manual_seed(4)
        command, cbins, body = recovery.draw(REC, frames, bins, MOTION.frame_bin)
        after = torch.get_rng_state()
        torch.manual_seed(4)
        u = [torch.rand(n) for _ in range(3)]
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        pick = u[0] < 0.5
        self.assertTrue(torch.equal(body[pick], REC.start_rows[(u[1] * 2).long()][pick]))
        self.assertTrue(torch.equal(command[pick], REC.command_rows[(u[2] * 7).long()][pick]))
        self.assertTrue(torch.equal(command[~pick], frames[~pick]) and torch.equal(body[~pick], frames[~pick]))
        self.assertTrue(torch.equal(cbins, MOTION.frame_bin[command]))
        self.assertTrue(150 < int(pick.sum()) < 250)
        state = recovery.start_state(body, command)
        self.assertTrue(torch.equal(state["recovering"], pick))
        self.assertTrue(torch.equal(state["hidden_frame"], body))
        self.assertEqual(float(state["command_offset"].abs().sum()) + int(state["error_controls"].sum())
                         + int(state["blend_left"].sum()), 0)


class TestTerminationAndBound(unittest.TestCase):
    def test_a_recovering_control_ends_only_by_the_stop(self):
        start = torch.tensor([True, True, False, False])
        stop = torch.tensor([True, False, False, True])
        # the simulator's termination against the command row
        self.assertEqual(recovery.termination(start, stop, torch.tensor([False, True, True, False])).tolist(),
                         [True, False, True, False])
        # imagination's soft termination: a lying body under an upright command row is almost surely terminated
        soft = MDP_.soft_termination(state_of([0, 0, 0, 0]), torch.tensor([35] * 4))
        self.assertTrue(bool((soft > 0.99).all()))
        d = recovery.termination(start, stop, soft)
        self.assertEqual(d.dtype, soft.dtype)
        self.assertEqual(d[:2].tolist(), [1.0, 0.0])
        self.assertTrue(torch.equal(d[2:], soft[2:]))

    def test_the_value_bound(self):
        release = (MAX_STEP_REWARD + max(0.25, 0.0) * 2 * CONTROL_DT) / (1 - 0.99)
        self.assertEqual(recovery.value_bound(0.99, 0.25, 1.0), release)
        self.assertAlmostEqual(release, 11.0, places=9)
        self.assertAlmostEqual(recovery.value_bound(0.99, 0.25, 2.5), 26.0, places=9)
        # below 1 the tracking and blend controls (scale up to 1) keep the release's bound
        self.assertEqual(recovery.value_bound(0.99, 0.25, 0.5), release)


class TestFailurePenaltyAtTheTargetRow(unittest.TestCase):
    def test_the_soft_termination_at_the_target_row_times_the_weight(self):
        lying = state_of([0, 0])
        rows = torch.tensor([0, 35])  # a lying body: at its own (hidden) row, and under an upright command row
        pen = recovery.failure_penalty(MDP_, lying, rows, 3.0)
        self.assertTrue(torch.equal(pen, 3.0 * MDP_.soft_termination(lying, rows, scale=1.0)))
        self.assertLess(float(pen[0]), 0.01)
        self.assertGreater(float(pen[1]), 2.97)
        # outside recovery the target row is the command row: the release's penalty (--fail_penalty) form
        S = state_of([35])
        self.assertTrue(torch.equal(recovery.failure_penalty(MDP_, S, torch.tensor([35]), 3.0),
                                    3.0 * MDP_.soft_termination(S, torch.tensor([35]))))

    def test_it_has_a_gradient_through_the_state(self):
        z = float(MOTION.root_p[35][2]) - 0.25  # at the height threshold of row 35
        S = state_of([35], z=[z], grad=True)
        recovery.failure_penalty(MDP_, S, torch.tensor([35]), 3.0).sum().backward()
        self.assertGreater(float(S[0].grad[0, 2].abs()), 1.0)


def release_reward(m, S, f, a, a_prev, terms=False, w_contact=0.0):
    """MDP.reward of the release (BlinkMimic 2 v1.0.0, src/blinkmimic2/mdp.py), for the MDP ``m``."""
    c, mo = m.cfg, m.motion
    p, R, v, w, q, qd = S
    bp, bR, bv, bw = m.kin.forward(p, R, v, w, q, qd)
    rbp, rbR, rbv, rbw = mo.ref_bodies(f)
    rp, rR = mo.root_p[f], mo.root_R[f]
    dR = yaw_mat(mm3(R, rR.transpose(-1, -2)))
    anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
    ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), rbp - rp.unsqueeze(-2))
    ref_bR = mm3(dR.unsqueeze(-3), rbR)
    e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
    e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
    e_lv = ((rbv - bv) ** 2).sum(-1).mean(-1)
    e_av = ((rbw - bw) ** 2).sum(-1).mean(-1)
    e_rp = ((rp - p) ** 2).sum(-1)
    e_ro = rot_angle_sq(rR, R)
    r_track = (
        torch.exp(-e_pos / c.body_pos_std**2)
        + torch.exp(-e_ori / c.body_ori_std**2)
        + torch.exp(-e_lv / c.body_linvel_std**2)
        + torch.exp(-e_av / c.body_angvel_std**2)
        + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
        + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
    )
    r_rate = ((a - a_prev) ** 2).sum(-1)
    r_lim = (torch.relu(q - m.soft_hi) + torch.relu(m.soft_lo - q)).sum(-1)
    r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT
    if w_contact > 0:
        fz, rz = bp[..., m.feet, 2], rbp[..., m.feet, 2]
        c_sim = torch.sigmoid((CONTACT_Z - fz) / 0.005)
        rc = getattr(mo, "ref_contact", None)
        c_ref = rc[f] if rc is not None else (rz < CONTACT_Z).float()
        slip = (bv[..., m.feet, :2] ** 2).sum(-1)
        r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * torch.exp(-slip / 0.3**2)
        r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
    if terms:
        return r, {"track": r_track * CONTROL_DT, "e_body_pos": e_pos, "e_body_ori": e_ori, "e_root_pos": e_rp,
                   "action_rate": r_rate, "joint_limit": r_lim}
    return r


def release_critic_features(m, S, f, xi):
    """MDP.critic_features of the release, for the MDP ``m``."""
    mo = m.motion
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Ht = H.transpose(-1, -2)
    rp, rR = mo.root_p[f], mo.root_R[f]
    bp = m.kin.forward(p, R, v, w, q, qd)[0]
    dR = yaw_mat(mm3(R, rR.transpose(-1, -2)))
    anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
    ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), mo.ref_bodies(f)[0] - rp.unsqueeze(-2))
    e_body = mv3(Ht.unsqueeze(-3), ref_bp - bp).flatten(-2)
    e_root = mv3(Ht, rp - p)
    rel = mat_to_6d(mm3(R.transpose(-1, -2), rR))
    return torch.cat((invariant_features(S), mo.q[f] - q, e_body, e_root, rel, xi), -1)


ROWS = [0, 3, 6, 12, 20, 33, 40]


def perturbed(seed: int, shift=None):
    """States near the reference rows ROWS, their pelvis moved in the plane by ``shift`` when given."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    p, R, v, w, q, qd = state_of(ROWS)
    n = len(ROWS)
    p = p + 0.1 * r(n, 3)
    if shift is not None:
        p = p + torch.nn.functional.pad(shift, (0, 1))
    return p, so3_exp(0.3 * r(n, 3)) @ R, v + r(n, 3), w + r(n, 3), q + 0.2 * r(n, J), qd + r(n, J)


def controls(seed: int):
    g = torch.Generator().manual_seed(seed)
    n = len(ROWS)
    return (torch.tensor(ROWS) + 1, torch.randn(n, J, generator=g), torch.randn(n, J, generator=g),
            torch.randn(n, MDP_.xi_dim, generator=g))


def neutral(scale=1.0, height_only=False, offset=None):
    n = len(ROWS)
    return (torch.full((n,), scale), torch.full((n,), height_only),
            torch.zeros(n, 2) if offset is None else offset)


class TestRewardAndCritic(unittest.TestCase):
    def test_without_recovery_the_release_values(self):
        S, (f, a, a_prev, xi) = perturbed(1), controls(2)
        for w_contact in (0.0, 0.25):
            r, terms = MDP_.reward(S, f, a, a_prev, terms=True, w_contact=w_contact)
            r0, terms0 = release_reward(MDP_, S, f, a, a_prev, terms=True, w_contact=w_contact)
            self.assertTrue(torch.equal(r, r0))
            for k in terms0:
                self.assertTrue(torch.equal(terms[k], terms0[k]), k)
        self.assertTrue(torch.equal(MDP_.critic_features(S, f, xi), release_critic_features(MDP_, S, f, xi)))

    def test_neutral_recovery_arguments_change_nothing(self):
        S, (f, a, a_prev, xi) = perturbed(3), controls(4)
        r, terms = MDP_.reward(S, f, a, a_prev, terms=True, w_contact=0.25)
        r1, terms1 = MDP_.reward(S, f, a, a_prev, terms=True, w_contact=0.25, recovery=neutral())
        self.assertTrue(torch.equal(r, r1))
        self.assertTrue(all(torch.equal(terms[k], terms1[k]) for k in terms))
        self.assertTrue(torch.equal(MDP_.critic_features(S, f, xi), MDP_.critic_features(S, f, xi, torch.zeros(7, 2))))

    def test_the_scale_multiplies_the_tracking_terms_only(self):
        S, (f, a, a_prev, _) = perturbed(5), controls(6)
        r1, terms1 = MDP_.reward(S, f, a, a_prev, terms=True, w_contact=0.25, recovery=neutral())
        for scale in (2.5, 1 / 75):
            r, terms = MDP_.reward(S, f, a, a_prev, terms=True, w_contact=0.25, recovery=neutral(scale))
            # the action rate, the joint limits and the contact term are not scaled
            torch.testing.assert_close(r - r1, (scale - 1) * terms1["track"])
            torch.testing.assert_close(terms["track"], scale * terms1["track"])

    def test_the_root_height_alone_in_recovery(self):
        shift = torch.tensor([[0.7, -0.4]]).expand(7, 2)
        f, a, a_prev, _ = controls(8)
        _, terms = MDP_.reward(perturbed(7), f, a, a_prev, terms=True, recovery=neutral(height_only=True))
        _, moved = MDP_.reward(perturbed(7, shift), f, a, a_prev, terms=True, recovery=neutral(height_only=True))
        torch.testing.assert_close(terms["e_root_pos"], (MOTION.root_p[f][:, 2] - perturbed(7)[0][:, 2]) ** 2)
        torch.testing.assert_close(moved["e_root_pos"], terms["e_root_pos"])
        _, full = MDP_.reward(perturbed(7), f, a, a_prev, terms=True)
        self.assertTrue(bool((full["e_root_pos"] > terms["e_root_pos"]).all()))

    def test_the_offset_moves_the_reference_root_only(self):
        # a body moved in the plane by the offset gets the reward and critic features of the body that is not moved
        shift = torch.tensor([[0.7, -0.8], [-1.0, 2.0], [1.0, 1.0], [0.0, 1.5], [2.0, 0.0], [-1.0, -1.0], [0.6, 0.9]])
        S, moved = perturbed(9), perturbed(9, shift)
        f, a, a_prev, xi = controls(10)
        r, terms = MDP_.reward(S, f, a, a_prev, terms=True, w_contact=0.25, recovery=neutral())
        rm, terms_m = MDP_.reward(moved, f, a, a_prev, terms=True, w_contact=0.25, recovery=neutral(offset=shift))
        torch.testing.assert_close(rm, r)
        for k in terms:
            torch.testing.assert_close(terms_m[k], terms[k])
        torch.testing.assert_close(MDP_.critic_features(moved, f, xi, shift), MDP_.critic_features(S, f, xi))
        # without the offset only the root position error of the moved body changes
        _, terms_0 = MDP_.reward(moved, f, a, a_prev, terms=True, recovery=neutral())
        torch.testing.assert_close(terms_0["e_body_pos"], terms["e_body_pos"])
        self.assertTrue(bool((terms_0["e_root_pos"] > terms["e_root_pos"] + 0.1).all()))
        i = inv_dim(J) + J + 3 * len(ROBOT.tracked_bodies)
        unmoved, offset_off = MDP_.critic_features(S, f, xi), MDP_.critic_features(moved, f, xi)
        torch.testing.assert_close(offset_off[:, :i], unmoved[:, :i])
        torch.testing.assert_close(offset_off[:, i + 3:], unmoved[:, i + 3:])


RELEASE_FIELDS = ["s", "s1", "a", "a_prev", "xi", "r", "f", "ep_step", "done", "cut"]


class TestBuffer(unittest.TestCase):
    def test_fields(self):
        self.assertEqual(list(fields(J, MDP_.xi_dim)), RELEASE_FIELDS)
        self.assertEqual(list(fields(J, MDP_.xi_dim, recovery=True)), RELEASE_FIELDS + list(recovery.KEYS))

    def test_a_start_carries_the_regime_stored_with_its_transition(self):
        T, N = 4, 3
        for with_recovery in (False, True):
            buf = ReplayBuffer(T, N, "cpu", J, MDP_.xi_dim, recovery=with_recovery)
            buf.use_device_ptr(device_size=True)
            for t in range(T):
                tr = {k: torch.zeros((N,) + ((shape,) if isinstance(shape, int) else shape), dtype=dt)
                      for k, (shape, dt) in buf.fields.items()}
                tr["f"] = 100 * t + torch.arange(N)
                if with_recovery:
                    env = torch.arange(N)
                    tr.update(hidden_frame=10 * t + env, recovering=(t + env) % 2 == 0, error_controls=t + env,
                              blend_left=2 * env, command_offset=torch.stack((torch.full((N,), float(t)), env.float()), -1))
                buf.add_device(tr)
                buf.advance_host()
            age, env = torch.tensor([0, 3, 2, 1]), torch.tensor([1, 0, 2, 2])
            start, _ = buf.starts_at(age, env, 2)
            self.assertTrue(torch.equal(start["f"], 100 * age + env))
            if not with_recovery:
                self.assertEqual(set(start), {"s", "a_prev", "xi", "f"})
                continue
            self.assertTrue(torch.equal(start["hidden_frame"], 10 * age + env))
            self.assertTrue(torch.equal(start["recovering"], (age + env) % 2 == 0))
            self.assertTrue(torch.equal(start["error_controls"], age + env))
            self.assertTrue(torch.equal(start["blend_left"], 2 * env))
            self.assertTrue(torch.equal(start["command_offset"], torch.stack((age.float(), env.float()), -1)))


def nothing(*args, **kwargs):
    return None


def step_stand_in(S, S1, f, regime=None):
    """The attributes TrackingSim.step_core reads and writes, with a simulator whose state is S before the control and
    S1 after it."""
    N, states = len(f), iter([S, S1])
    return SimpleNamespace(
        pipe=SimpleNamespace(stateful=False), _obs=None, _obs_state=None, state=lambda: next(states), f=f.clone(),
        xi=torch.zeros(N, MDP_.xi_dim), a_prev=torch.zeros(N, J), ep_step=torch.zeros(N, dtype=torch.long),
        default_q=MDP_.default_q, J=J, alpha=torch.ones(J), robot=SimpleNamespace(set_joint_position_target=nothing),
        scene=SimpleNamespace(write_data_to_sim=nothing, update=nothing), sim=SimpleNamespace(step=nothing), mdp=MDP_,
        motion=MOTION, max_episode_steps=500, push_enabled=False, gamma=0.99, ep_disc_ret=torch.zeros(N),
        ep_ret=torch.zeros(N), recovery=None if regime is None else REC, regime=regime)  # fmt: skip


def zero_regime(N):
    zero = torch.zeros(N, dtype=torch.long)
    return recovery.start_state(zero, zero)


def reset_stand_in(N, with_recovery, frames):
    """The attributes TrackingSim.reset reads and writes; sampled resets get the rows ``frames``."""
    z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype)  # noqa: E731
    env = SimpleNamespace(
        motion=MOTION, mdp=MDP_, spec=ROBOT, reset_noise=True, eval_mode=False, aprev_ref=False,
        recovery=REC if with_recovery else None, sim=SimpleNamespace(reset=nothing),
        scene=SimpleNamespace(reset=nothing, env_origins=z(N, 3)), robot=Recorder(),
        randomizer=SimpleNamespace(reset=nothing), xi=None, f=z(N, dtype=torch.long), ep_step=z(N, dtype=torch.long),
        a_prev=z(N, J), start_bin=z(N, dtype=torch.long), ep_disc_ret=z(N), ep_ret=z(N), obs_fresh=z(N, dtype=torch.bool),
        sample_frames=lambda n: (frames, MOTION.frame_bin[frames]))  # fmt: skip
    if with_recovery:
        env.regime, env.recovery_episode = zero_regime(N), z(N, dtype=torch.bool)
    return env


def masked_stand_in(N, with_recovery, frames):
    """The attributes TrackingSim.reset_masked reads and writes, with the simulator's arrays; sampled resets get the
    rows ``frames``."""
    z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype)  # noqa: E731
    targets = {k: z(N, J) for k in ("joint_pos_target", "joint_vel_target", "joint_effort_target")}
    targets.update({k: z(N, 0) for k in ("tendon_len_target", "tendon_vel_target", "tendon_effort_target",
                                         "site_effort_target")})
    indexing = SimpleNamespace(free_joint_q_adr=torch.arange(7), free_joint_v_adr=torch.arange(6),
                               joint_q_adr=torch.arange(7, 7 + J), joint_v_adr=torch.arange(6, 6 + J))
    sim = SimpleNamespace(_reset_mask=z(N, dtype=torch.bool), wp_device="cpu", use_cuda_graph=True, reset_graph=object(),
                          data=SimpleNamespace(qpos=z(N, 7 + J), qvel=z(N, 6 + J)))
    env = SimpleNamespace(
        eval_mode=False, audit=None, pipe=SimpleNamespace(stateful=False), push_enabled=False,
        randomizer=SimpleNamespace(_terms=[], reset_masked=nothing), spec=ROBOT, N=N, device="cpu", motion=MOTION,
        sample_frames=lambda n, gate: (frames, MOTION.frame_bin[frames]), det=None, exam=None, reset_noise=True,
        det_nominal=False, mdp=MDP_, sim=sim, robot=SimpleNamespace(data=SimpleNamespace(**targets), indexing=indexing),
        scene=SimpleNamespace(env_origins=z(N, 3)), f=torch.full((N,), 41), ep_step=z(N, dtype=torch.long), a_prev=z(N, J),
        aprev_ref=False, start_bin=z(N, dtype=torch.long), ep_disc_ret=z(N), ep_ret=z(N), xi=None,
        obs_fresh=z(N, dtype=torch.bool), recovery=REC if with_recovery else None)  # fmt: skip
    if with_recovery:
        env.regime, env.recovery_episode = zero_regime(N), z(N, dtype=torch.bool)
        env.regime["hidden_frame"].fill_(-7)  # outside the mask nothing is written
    return env


def placed(rows):
    """The training reset's placement of the rows (reset noise drawn), joints clamped as the reset clamps them."""
    p, R, v, w, q, qd = placed_state(tuple(x.clone() for x in MOTION.ref_state(rows)), ROBOT.dr, True)
    return p, R, v, w, torch.clamp(q, MDP_.kin.q_lo, MDP_.kin.q_hi), qd


# MuJoCo Warp's reset of the masked worlds, which reset_masked replays as a captured Warp graph
NO_WARP = SimpleNamespace(ScopedDevice=lambda device: contextlib.nullcontext(), capture_launch=nothing)


class TestSimulator(unittest.TestCase):
    def test_step_without_recovery_is_the_release_step(self):
        f, a = torch.tensor([31, 33]), torch.randn(2, J)
        S1 = state_of([32, 0])  # world 1 lying
        env = step_stand_in(state_of([31, 33]), S1, f)
        before = torch.get_rng_state()
        out = TrackingSim.step_core(env, a)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(set(out), {"s", "a", "a_prev", "f", "s1", "r", "term", "done", "ep_step", "xi", "terms", "cut"})
        self.assertTrue(torch.equal(out["r"], release_reward(MDP_, S1, f + 1, a, torch.zeros(2, J))))
        self.assertEqual(out["term"].tolist(), [False, True])

    def test_step_with_recovery(self):
        # world 0 recovering on its hidden row, world 1 tracking, world 2 recovering at its 100th error
        f, a = torch.tensor([30, 32, 34]), torch.randn(3, J)
        regime = zero_regime(3)
        regime["hidden_frame"].copy_(torch.tensor([2, 32, 3]))
        regime["recovering"].copy_(torch.tensor([True, False, True]))
        regime["error_controls"].copy_(torch.tensor([0, 0, 99]))
        pointers = {k: v.data_ptr() for k, v in regime.items()}
        S1 = state_of([3, 33, 4])
        S1[0][2, 2] = 2.0
        env = step_stand_in(state_of([2, 32, 3]), S1, f, regime)
        out = TrackingSim.step_core(env, a)
        rows, recovering = torch.tensor([3, 33, 4]), torch.tensor([True, False, True])
        want = MDP_.reward(S1, rows, a, torch.zeros(3, J),
                           recovery=(torch.tensor([2.5, 1.0, 2.5]), recovering, torch.zeros(3, 2)))
        torch.testing.assert_close(out["r"], want)
        # the release terminates world 0 against its command row; recovering, it ends only by the stop
        self.assertEqual(MDP_.termination(S1, f + 1).tolist(), [True, False, True])
        self.assertEqual(out["term"].tolist(), [False, False, True])
        self.assertEqual(out["recovery_stop"].tolist(), [False, False, True])
        self.assertEqual(out["done"].tolist(), [False, False, True])
        # out holds the regime of the state the control starts from; the regime is advanced in place
        self.assertEqual(out["hidden_frame"].tolist(), [2, 32, 3])
        self.assertEqual(out["error_controls"].tolist(), [0, 0, 99])
        self.assertEqual(env.regime["hidden_frame"].tolist(), [3, 33, 4])
        self.assertEqual(env.regime["error_controls"].tolist(), [0, 0, 100])
        self.assertEqual({k: v.data_ptr() for k, v in env.regime.items()}, pointers)
        self.assertEqual(env.f.tolist(), [31, 33, 35])

    def test_bins_take_tracking_episodes_only(self):
        def tail(with_recovery):
            resets = []
            env = SimpleNamespace(
                bin_fix=True, motion=MOTION, gamma=0.99, ep_step=torch.tensor([10, 20, 30]),
                ep_ret=torch.tensor([0.5, 1.0, 1.5]), ep_disc_ret=torch.tensor([0.4, 0.8, 1.2]),
                start_bin=torch.tensor([0, 3, 1]), bin_G=torch.zeros(MOTION.num_bins),
                bin_fail=torch.zeros(MOTION.num_bins), track_fail=True, det=None, exam=None,
                recovery=REC if with_recovery else None, recovery_episode=torch.tensor([True, False, False]),
                reset_masked=resets.append)  # fmt: skip
            out = {"done": torch.tensor([True, True, False]), "term": torch.tensor([True, False, False]),
                   "f": torch.tensor([5, 35, 20])}
            out = TrackingSim.step_tail_masked(env, out)
            self.assertEqual(len(resets), 1)
            return env, out
        release, _ = tail(False)
        env, out = tail(True)
        self.assertEqual((release.bin_G != 0).tolist(), [True, False, False, True])
        self.assertEqual((env.bin_G != 0).tolist(), [False, False, False, True])
        self.assertEqual(float(env.bin_G[3]), float(release.bin_G[3]))
        self.assertTrue(float(release.bin_fail[0]) > 0)
        self.assertEqual(float(env.bin_fail.abs().sum()), 0.0)
        # the episode counts and sums of the log keep every ended episode
        self.assertEqual((float(out["n_done"]), float(out["ep_len_sum"])), (2.0, 30.0))

    def test_sampled_reset_draws(self):
        N, ids = 40, torch.arange(0, 40, 2)
        frames = 30 + torch.arange(20) % 12
        for with_recovery in (False, True):
            env = reset_stand_in(N, with_recovery, frames)
            torch.manual_seed(6)
            TrackingSim.reset(env, ids)
            after = torch.get_rng_state()
            torch.manual_seed(6)
            command, body = frames, frames
            if with_recovery:
                command, _, body = recovery.draw(REC, frames, MOTION.frame_bin[frames], MOTION.frame_bin)
            p, R, v, w, q, qd = placed(body)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            wr = env.robot.writes
            self.assertTrue(torch.equal(wr["write_root_link_pose_to_sim"][0], torch.cat((p, mat_to_quat(R)), -1)))
            self.assertTrue(torch.equal(wr["write_joint_position_to_sim"][0], q))
            self.assertTrue(torch.equal(env.f[ids], command))
            if with_recovery:
                starts = body != command
                self.assertTrue(0 < int(starts.sum()) < 20)
                self.assertTrue(torch.equal(env.regime["hidden_frame"][ids], body))
                self.assertTrue(torch.equal(env.regime["recovering"][ids], starts))
                self.assertTrue(torch.equal(env.recovery_episode[ids], starts))
                self.assertFalse(bool(env.recovery_episode[1::2].any()))

    def test_masked_reset(self):
        N = 16
        frames, mask = 30 + torch.arange(N) % 12, torch.arange(N) % 4 != 1
        with mock.patch.object(env_module, "wp", NO_WARP):
            # without recovery: no recovery draw, the release's placement
            env = masked_stand_in(N, False, frames)
            torch.manual_seed(8)
            with mock.patch.object(recovery, "draw", side_effect=AssertionError("drawn")):
                TrackingSim.reset_masked(env, mask)
            torch.manual_seed(8)
            p, R, v, w, q, qd = placed(frames)
            self.assertTrue(torch.equal(env.sim.data.qpos[mask, :3], p[mask]))
            self.assertTrue(torch.equal(env.f, torch.where(mask, frames, 41)))
            # with recovery: the recovery draws after the reset rows, then the reset noise
            env = masked_stand_in(N, True, frames)
            torch.manual_seed(8)
            TrackingSim.reset_masked(env, mask)
            torch.manual_seed(8)
            command, _, body = recovery.draw(REC, frames, MOTION.frame_bin[frames], MOTION.frame_bin)
            p, R, v, w, q, qd = placed(body)
        starts = mask & (body != command)
        self.assertTrue(0 < int(starts.sum()) < int(mask.sum()))
        self.assertTrue(torch.equal(env.sim.data.qpos[mask, :3], p[mask]))
        self.assertTrue(torch.equal(env.sim.data.qpos[mask, 7:], q[mask]))
        self.assertTrue(torch.equal(env.f, torch.where(mask, command, 41)))
        self.assertTrue(torch.equal(env.regime["hidden_frame"], torch.where(mask, body, -7)))
        self.assertTrue(torch.equal(env.regime["recovering"], starts))
        self.assertTrue(torch.equal(env.recovery_episode, starts))


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFUSED = ["--exam_frac != 0", "--front_share != 0", "--fail_penalty != 0", "--wstream_frac != 0",
           "--start_prac_frac >= 0", "a non-empty --phase", "--pi_ref_skip", "--reset_aprev_ref"]


def preset(name: str) -> list:
    """The flags of PRESET[name] in scripts/presets.sh."""
    cmd = ["bash", "-c", f'source "{REPO}/scripts/presets.sh" && echo "${{PRESET[{name}]}}"']
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()


def train(log_dir: str, *flags):
    cmd = [sys.executable, "-m", "blinkmimic2.train", "--log_dir", log_dir, *flags]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestOptions(unittest.TestCase):
    def test_a_partial_set_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = train(os.path.join(d, "run"), "--recovery_share", "0.5", "--recovery_seconds", "2.0", "1.5")
            self.assertEqual(r.returncode, 2)
            self.assertIn("the recovery options go together; missing ['--recovery_hidden_clips', "
                          "'--recovery_command_excluded_clips', '--recovery_low_pelvis', '--recovery_rows', "
                          "'--recovery_reward_scale']", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "run")))

    def test_each_refused_option_alone(self):
        one = {"--exam_frac != 0": ["--exam_frac", "0.5"], "--front_share != 0": ["--front_share", "1"],
               "--fail_penalty != 0": ["--fail_penalty", "3"], "--wstream_frac != 0": ["--wstream_frac", "0.25"],
               "--start_prac_frac >= 0": ["--start_prac_frac", "0"], "a non-empty --phase": ["--phase", "0:fail_penalty=0"],
               "--pi_ref_skip": ["--pi_ref_skip"], "--reset_aprev_ref": ["--reset_aprev_ref"]}
        self.assertEqual(list(one), REFUSED)
        for name, flags in one.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                r = train(os.path.join(d, "run"), *preset("recovery"), *flags)
                self.assertEqual(r.returncode, 2, r.stderr[-2000:])
                self.assertIn(f"the recovery options are refused with {name}: each reads", r.stderr)

    def test_the_options_with_the_default_recipe_are_refused(self):
        control, setting = preset("recovery_control"), preset("recovery")
        self.assertEqual(setting[: len(control)], control)
        with tempfile.TemporaryDirectory() as d:
            r = train(os.path.join(d, "run"), *setting[len(control):])
            self.assertEqual(r.returncode, 2)
            self.assertIn("the recovery options are refused with " + ", ".join(REFUSED), r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "run")))

    def test_the_failure_penalty_at_the_target_row(self):
        with tempfile.TemporaryDirectory() as d:
            r = train(os.path.join(d, "a"), *preset("recovery_control"), "--recovery_fail_penalty", "3", "2000")
            self.assertEqual(r.returncode, 2)
            self.assertIn("--recovery_fail_penalty needs the recovery options, a weight above 0", r.stderr)
            for values in (("0", "2000"), ("3", "-1"), ("3", "2000.5")):
                r = train(os.path.join(d, "b"), *preset("recovery"), "--recovery_fail_penalty", *values)
                self.assertEqual(r.returncode, 2, values)
                self.assertIn("--recovery_fail_penalty needs the recovery options", r.stderr)
            r = train(os.path.join(d, "c"), *preset("recovery"), "--recovery_fail_penalty", "3", "2000", "--fail_penalty", "3")
            self.assertEqual(r.returncode, 2)
            self.assertIn("refused with --fail_penalty != 0", r.stderr)
            missing = ["--bones_list", os.path.join(d, "missing.txt")]
            r = train(os.path.join(d, "run"), *preset("recovery"), "--recovery_fail_penalty", "3", "2000", *missing)
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            run = json.load(open(os.path.join(d, "run", "args.json")))
            r = train(os.path.join(d, "plain"), *preset("recovery"), *missing)
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            plain = json.load(open(os.path.join(d, "plain", "args.json")))
        self.assertEqual(run["recovery_fail_penalty"], [3.0, 2000.0])
        self.assertEqual(run["fail_penalty"], 0.0)
        self.assertNotIn("recovery_fail_penalty", plain)
        self.assertEqual({k: v for k, v in run.items() if k not in ("recovery_fail_penalty", "log_dir")},
                         {k: v for k, v in plain.items() if k != "log_dir"})
        src = open(os.path.join(REPO, "src", "blinkmimic2", "train.py")).read()
        self.assertIn("r = r - recovery.failure_penalty(mdp, S_all, flat(Rows), RECOVERY_PENALTY_WEIGHT).view(HZ, B)", src)
        self.assertIn("RECOVERY_PENALTY_WEIGHT.fill_(recovery.penalty_weight(it, *RECOVERY_PENALTY))", src)

    def test_the_penalty_starts_where_the_release_schedule_starts_its_failure_penalty(self):
        # the release's schedule (train.py FINAL_PHASE, the default of --phase): the first entry with fail_penalty > 0
        src = open(os.path.join(REPO, "src", "blinkmimic2", "train.py")).read()
        schedule = "".join(re.findall(r'"([^"]*)"', src[src.index("FINAL_PHASE = ("):src.index(")", src.index("FINAL_PHASE = ("))]))
        starts = [int(e.split(":")[0]) for e in schedule.split(";")
                  if any(kv.split("=")[0] == "fail_penalty" and float(kv.split("=")[1]) > 0 for kv in e.split(":")[1].split(","))]
        self.assertEqual(starts[0], 2000)
        self.assertEqual([recovery.penalty_weight(it, 3.0, 2000) for it in (0, 1999, 2000, 6999)], [0.0, 0.0, 3.0, 3.0])
        self.assertEqual(recovery.penalty_weight(0, 3.0, 0), 3.0)
        # a scalar tensor weight, as train.py passes it to the captured update
        S = state_of([0])
        pen = recovery.failure_penalty(MDP_, S, torch.tensor([35]), torch.tensor(0.0))
        self.assertEqual(float(pen), 0.0)

    def test_the_presets(self):
        self.assertEqual(preset("recovery_control"), preset("baseline") + ["--no-pi_ref_skip"])
        # the options that are kept, at the release's values
        kept = ["--fail_reset", "0.3", "--contact_reward", "0.25", "--exam_pre_every", "500", "--exam_mix", "0.6,0.2",
                "--share_ramp", "0.5,0.9,50,0.1"]
        with tempfile.TemporaryDirectory() as d:
            # past the checks the run writes args.json, then needs CUDA (or, on a GPU, the missing clip list)
            r = train(os.path.join(d, "control"), *preset("recovery_control"), "--bones_list", os.path.join(d, "missing.txt"))
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            control = json.load(open(os.path.join(d, "control", "args.json")))
            r = train(os.path.join(d, "run"), *preset("recovery"), *kept, "--bones_list", os.path.join(d, "missing.txt"))
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            self.assertNotIn("refused", r.stderr)
            run = json.load(open(os.path.join(d, "run", "args.json")))
        # without the options args.json holds none of them
        self.assertEqual(set(run) - set(control), set(recovery.OPTIONS))
        self.assertFalse([k for k in control if k.startswith("recovery")])
        self.assertEqual(run["recovery_share"], 0.5)
        hidden = ["fallAndGetUp1_subject1", "fallAndGetUp1_subject5", "fallAndGetUp2_subject2", "fallAndGetUp2_subject3",
                  "fallAndGetUp3_subject1"]
        self.assertEqual(run["recovery_hidden_clips"], hidden)
        self.assertEqual(run["recovery_command_excluded_clips"], hidden + ["fallAndGetUp1_subject4"])
        self.assertEqual(run["recovery_low_pelvis"], [0.5, 0.5])
        self.assertEqual(run["recovery_rows"], [50, 375, 500])
        self.assertEqual(run["recovery_reward_scale"], 2.5)
        self.assertEqual(run["recovery_seconds"], [2.0, 1.5])
        settings = recovery.settings_from_args(argparse.Namespace(**run))
        self.assertEqual((settings.stop_controls, settings.blend_controls), (100, 75))


if __name__ == "__main__":
    unittest.main()
