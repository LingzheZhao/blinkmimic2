"""What ends an episode: the root's height and tilt by default (the release), or with train.py --termination
torso_and_end_effectors (mdp.body_termination) the rule of BeyondMimic's released code, the heights of the torso link,
the ankle roll links and the wrist yaw links and the torso link's tilt against the reference. The rule gives the simulator's termination,
the danger ratio and the soft termination of imagined rollouts.

train.py builds the simulator and trains once its options are parsed, so only its refusals run here; a training with
this termination needs a GPU."""

import os
import subprocess
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ROBOT, library  # noqa: E402
from test_motion_dir import write_clip  # noqa: E402

from blinkmimic2.dr import XiLayout  # noqa: E402
from blinkmimic2.mdp import MDP, TERM_ANCHOR, TERM_END_EFFECTORS, RewardCfg, body_termination  # noqa: E402
from blinkmimic2.motion import MotionLib  # noqa: E402
from blinkmimic2.rotations import so3_exp  # noqa: E402

KIN, SYNTH, _ = library()
J_IDX = {n: i for i, n in enumerate(ROBOT.joint_names)}
BODIES = (TERM_ANCHOR, *TERM_END_EFFECTORS)


def libraries():
    """The synthetic library (every per-frame quantity stored) and a library of two CSV clips with root pose and joints
    stored in fp32, as train.py loads by default."""
    with tempfile.TemporaryDirectory() as d:
        write_clip(os.path.join(d, "a.csv"), 31)
        write_clip(os.path.join(d, "b.csv"), 46)
        compact = MotionLib.load(ROBOT, KIN, "cpu", d, "", None, "fp32")
    return {"full": SYNTH, "fp32": compact}


def mdp_of(motion, cfg: RewardCfg) -> MDP:
    return MDP(KIN, motion, torch.tensor(ROBOT.default_joint_pos_list()), XiLayout(ROBOT, "cpu").dim, ROBOT.sensing, cfg)


def release(S, f, motion):
    """The release's termination, danger ratio and soft termination (thresholds 0.25 m and 0.8)."""
    p, R = S[0], S[1]
    rp, rR = motion.root_p[f], motion.root_R[f]
    e_z = (rp[..., 2] - p[..., 2]).abs()
    e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
    alive = (1 - torch.sigmoid((e_z - 0.25) / 0.02)) * (1 - torch.sigmoid((e_tilt - 0.8) / 0.04))
    return (e_z > 0.25) | (e_tilt > 0.8), torch.maximum(e_z / 0.25, e_tilt / 0.8), 1 - alive


def body_rule(S, f, motion, h: float = 0.25):
    """The errors of BeyondMimic's released code computed here: heights of BODIES and the difference of the z components of the torso
    link's projected gravity R^T (0, 0, -1), from forward kinematics of every body. Returns (height errors [..., 5],
    tilt error, termination)."""
    i = [KIN.body_names.index(n) for n in BODIES]
    bp, bR = KIN.forward(*S, all_bodies=True)[:2]
    rbp, rbR = KIN.forward(*motion.ref_state(f), all_bodies=True)[:2]
    down = torch.tensor([0.0, 0.0, -1.0])
    gz = lambda Rb: (Rb.transpose(-1, -2) @ down)[..., 2]  # noqa: E731
    e_z = (rbp[:, i, 2] - bp[:, i, 2]).abs()
    e_tilt = (gz(rbR[:, i[0]]) - gz(bR[:, i[0]])).abs()
    return e_z, e_tilt, (e_z > h).any(-1) | (e_tilt > 0.8)


def random_states(motion, n: int, seed: int):
    """Reference states of random frames with the root moved by up to 0.4 m, turned by up to 0.75 rad about each axis
    and the joints moved by up to 0.5 rad."""
    g = torch.Generator().manual_seed(seed)
    u = lambda *shape: 2 * torch.rand(*shape, generator=g) - 1  # noqa: E731
    f = torch.randint(0, motion.num_frames, (n,), generator=g)
    p, R, v, w, q, qd = (x.clone() for x in motion.ref_state(f))
    return (p + 0.4 * u(n, 3), so3_exp(0.75 * u(n, 3)) @ R, v, w, q + 0.5 * u(n, q.shape[-1]), qd), f


def at_frame(motion, f: int, joints: dict | None = None, pitch: float = 0.0):
    """The reference state of frame f with the named joints set (clamped to their limits) and the root turned by
    ``pitch`` about the world y axis; a batch of one."""
    fr = torch.tensor([f])
    p, R, v, w, q, qd = (x.clone() for x in motion.ref_state(fr))
    for name, val in (joints or {}).items():
        j = J_IDX[name]
        q[:, j] = torch.clamp(torch.tensor(val), KIN.q_lo[j], KIN.q_hi[j])
    R = so3_exp(torch.tensor([[0.0, pitch, 0.0]])) @ R
    return (p, R, v, w, q, qd), fr


class TestDefault(unittest.TestCase):
    def test_the_default_equals_the_release_s_root_formulas_bitwise(self):
        # RewardCfg(), which train.py uses without --termination, against the release's formulas written out in release()
        for store, motion in libraries().items():
            S, f = random_states(motion, 512, 0)
            want = release(S, f, motion)
            self.assertTrue(0 < int(want[0].sum()) < 512, store)
            mdp = mdp_of(motion, RewardCfg())
            for got, w in zip((mdp.termination(S, f), mdp.danger(S, f), mdp.soft_termination(S, f)), want):
                self.assertTrue(torch.equal(got, w), store)

    def test_the_root_termination_does_not_depend_on_the_joints(self):
        S, f = at_frame(SYNTH, 2, {"left_hip_pitch_joint": -2.0})
        q = S[4].clone().requires_grad_(True)
        S = (*S[:4], q, S[5])
        mdp = mdp_of(SYNTH, RewardCfg())
        self.assertFalse(bool(mdp.termination(S, f)))
        self.assertFalse(mdp.soft_termination(S, f).requires_grad)


class TestBodyTermination(unittest.TestCase):
    def test_it_tests_the_torso_link_and_the_ankle_roll_and_wrist_yaw_links(self):
        cfg = body_termination(RewardCfg(term_height=0.3), ROBOT.tracked_bodies)
        names = [ROBOT.tracked_bodies[i] for i in cfg.term_bodies]
        self.assertEqual(names, ["torso_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link",
                                 "right_wrist_yaw_link"])  # fmt: skip
        self.assertEqual((cfg.term_height, cfg.term_tilt), (0.3, 0.8))

    def test_bodies_the_robot_does_not_track_are_refused(self):
        tracked = [n for n in ROBOT.tracked_bodies if n != "left_wrist_yaw_link"]
        with self.assertRaisesRegex(ValueError, "left_wrist_yaw_link"):
            body_termination(RewardCfg(), tracked)

    def test_the_errors_heights_and_projected_gravity_of_the_five_bodies(self):
        for store, motion in libraries().items():
            for h in (0.25, 0.1):
                mdp = mdp_of(motion, body_termination(RewardCfg(term_height=h), ROBOT.tracked_bodies))
                S, f = random_states(motion, 512, 1)
                e_z, e_tilt, want = body_rule(S, f, motion, h)
                got_z, got_tilt = mdp.body_errors(S, f)
                torch.testing.assert_close(got_z, e_z, rtol=0, atol=1e-6)
                torch.testing.assert_close(got_tilt, e_tilt, rtol=0, atol=1e-6)
                # states within 1e-5 of a threshold are left out of the exact comparison
                clear = ((e_z - h).abs() > 1e-5).all(-1) & ((e_tilt - 0.8).abs() > 1e-5)
                self.assertGreater(int(clear.sum()), 500)
                term = mdp.termination(S, f)
                self.assertTrue(torch.equal(term[clear], want[clear]), (store, h))
                self.assertTrue(0 < int(want.sum()) < 512)
                if h == 0.25:  # the rule ends episodes the root's would not
                    self.assertGreater(int((term & ~release(S, f, motion)[0]).sum()), 0)
                torch.testing.assert_close(mdp.danger(S, f), torch.maximum((e_z / h).amax(-1), e_tilt / 0.8),
                                           rtol=1e-6, atol=1e-6)  # fmt: skip
                alive = (1 - torch.sigmoid((e_z - h) / 0.02)).prod(-1) * (1 - torch.sigmoid((e_tilt - 0.8) / 0.04))
                torch.testing.assert_close(mdp.soft_termination(S, f), 1 - alive, rtol=1e-5, atol=1e-6)

    def test_a_raised_foot_or_hand_ends_the_episode_with_the_root_on_the_reference(self):
        body = mdp_of(SYNTH, body_termination(RewardCfg(), ROBOT.tracked_bodies))
        root = mdp_of(SYNTH, RewardCfg())
        for joint, val, end_effector in (("left_hip_pitch_joint", -2.0, "left_ankle_roll_link"),
                                         ("right_shoulder_pitch_joint", -2.5, "right_wrist_yaw_link")):  # fmt: skip
            S, f = at_frame(SYNTH, 2, {joint: val})
            e_z = body_rule(S, f, SYNTH)[0][0]
            self.assertGreater(float(e_z[BODIES.index(end_effector)]), 0.3, joint)  # the state raises that body
            self.assertTrue(bool(body.termination(S, f)), joint)
            self.assertFalse(bool(root.termination(S, f)), joint)
            self.assertGreaterEqual(float(body.danger(S, f)), 1.0)
            self.assertGreater(float(body.soft_termination(S, f)), 0.9)
            # on the reference itself nothing ends
            S0, f0 = at_frame(SYNTH, 2)
            self.assertFalse(bool(body.termination(S0, f0)))
            self.assertLess(float(body.soft_termination(S0, f0)), 1e-3)

    def test_the_torso_link_s_tilt_ends_the_episode_with_the_pelvis_below_the_threshold(self):
        # heights left out (10 m) so that only the tilt is tested: the pelvis turned by 1.2 rad about y
        # (0.64 < 0.8 of the z component) and the waist pitched the same way, which tilts the torso link further
        body = mdp_of(SYNTH, body_termination(RewardCfg(term_height=10.0), ROBOT.tracked_bodies))
        root = mdp_of(SYNTH, RewardCfg(term_height=10.0))
        cases = [at_frame(SYNTH, 2, {"waist_pitch_joint": s * 0.52}, pitch=s * 1.2) for s in (1.0, -1.0)]
        S, f = max(cases, key=lambda c: float(body_rule(*c, SYNTH)[1][0]))
        p_tilt = float((SYNTH.root_R[f][0, 2, 2] - S[1][0, 2, 2]).abs())
        t_tilt = float(body_rule(S, f, SYNTH)[1][0])
        self.assertLess(p_tilt, 0.8)
        self.assertGreater(t_tilt, 0.8)
        self.assertTrue(bool(body.termination(S, f)))
        self.assertFalse(bool(root.termination(S, f)))

    def test_the_soft_termination_has_a_gradient_from_the_end_effector_heights(self):
        # the soft termination of the release depends on the root only; with the rule a joint that moves an
        # end-effector near its threshold changes it
        body = mdp_of(SYNTH, body_termination(RewardCfg(), ROBOT.tracked_bodies))
        S, f = at_frame(SYNTH, 2, {"left_hip_pitch_joint": -0.9})
        q = S[4].clone().requires_grad_(True)
        body.soft_termination((*S[:4], q, S[5]), f).sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad[0, J_IDX["left_hip_pitch_joint"]].abs()), 0.0)


class TestCommandLine(unittest.TestCase):
    def call(self, *extra):
        cmd = [sys.executable, "-m", "blinkmimic2.train", *extra]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_an_unknown_termination_is_refused(self):
        r = self.call("--termination", "end_effectors")
        self.assertEqual(r.returncode, 2)
        self.assertIn("invalid choice", r.stderr)


if __name__ == "__main__":
    unittest.main()
