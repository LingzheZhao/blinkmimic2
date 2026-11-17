"""The reset placement (env.placed_state, TrackingSim.reset): the random draws in the release's order and values, the
unit uniforms scaled by the half-widths, and a body row apart from the command row."""

import dataclasses
import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ROBOT, library  # noqa: E402

from blinkmimic2.dr import DomainRandCfg  # noqa: E402
from blinkmimic2.env import TrackingSim, placed_state  # noqa: E402
from blinkmimic2.rotations import mat_to_quat, so3_exp  # noqa: E402

KIN, MOTION, MDP_ = library()


def release_draws(ref, c: DomainRandCfg):
    """The reset noise of the training reset, drawn by these calls in this order: position, rotation, linear velocity,
    angular velocity, joints, then joint velocities if reset_joint_vel > 0."""
    p, R, v, w, q, qd = ref
    n, dev = p.shape[0], p.device
    u = lambda half: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in half], -1)  # noqa: E731
    p = p + u(c.reset_pos)
    R = so3_exp(u(c.reset_rot)) @ R
    v = v + u(c.reset_lin_vel)
    w = w + u(c.reset_ang_vel)
    q = q + torch.empty_like(q).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
    if c.reset_joint_vel > 0:
        qd = qd + torch.empty_like(qd).uniform_(-c.reset_joint_vel, c.reset_joint_vel)
    return p, R, v, w, q, qd


def ref_of(rows):
    return tuple(x.clone() for x in MOTION.ref_state(torch.tensor(rows)))


def unit(n: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    sizes = {"position": 3, "rotation": 3, "velocity": 6, "joints": ROBOT.num_joints}
    return {k: 2 * torch.rand(n, s, generator=g) - 1 for k, s in sizes.items()}


class Recorder:
    """The robot entity's writes of TrackingSim.reset."""

    def __init__(self):
        self.writes = {}

    def __getattr__(self, name):
        return lambda x, env_ids: self.writes.__setitem__(name, (x.clone(), env_ids.clone()))


def stand_in_sim(N: int, noise: bool, dr: DomainRandCfg = DomainRandCfg()):
    """The attributes TrackingSim.reset reads and writes, without the simulator."""
    z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype)  # noqa: E731
    return SimpleNamespace(
        motion=MOTION, mdp=MDP_, spec=dataclasses.replace(ROBOT, dr=dr), reset_noise=noise, eval_mode=not noise,
        aprev_ref=False, sim=SimpleNamespace(reset=lambda ids: None),
        scene=SimpleNamespace(reset=lambda ids: None, env_origins=z(N, 3)), robot=Recorder(),
        randomizer=SimpleNamespace(reset=lambda xi, ids: None), xi=None, f=z(N, dtype=torch.long),
        ep_step=z(N, dtype=torch.long), a_prev=z(N, ROBOT.num_joints), start_bin=z(N, dtype=torch.long),
        ep_disc_ret=z(N), ep_ret=z(N), obs_fresh=z(N, dtype=torch.bool))  # fmt: skip


class TestPlacedState(unittest.TestCase):
    def test_draws_equal_the_release_in_order_and_value(self):
        for dr in (DomainRandCfg(), DomainRandCfg(reset_joint_vel=0.3)):
            for rows in ([0, 7, 12], list(range(18))):  # n reset ids, or every world as reset_masked draws
                torch.manual_seed(11)
                got = placed_state(ref_of(rows), dr, True)
                after = torch.get_rng_state()
                torch.manual_seed(11)
                want = release_draws(ref_of(rows), dr)
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                for g, w in zip(got, want):
                    self.assertTrue(torch.equal(g, w))

    def test_without_noise_nothing_is_drawn_or_changed(self):
        ref = ref_of([1, 9])
        before = torch.get_rng_state()
        got = placed_state(ref, DomainRandCfg(), False)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for g, r in zip(got, ref):
            self.assertTrue(torch.equal(g, r))

    def test_uniforms_scale_the_half_widths_and_draw_nothing(self):
        c, ref, u = DomainRandCfg(), ref_of([2, 8, 15]), unit(3)
        before = torch.get_rng_state()
        p, R, v, w, q, qd = placed_state(ref, c, True, u)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        h = lambda x: torch.tensor(x)  # noqa: E731
        torch.testing.assert_close(p - ref[0], u["position"] * h((0.05, 0.05, 0.01)))
        torch.testing.assert_close(v - ref[2], u["velocity"][:, :3] * h((0.5, 0.5, 0.2)))
        torch.testing.assert_close(w - ref[3], u["velocity"][:, 3:] * h((0.52, 0.52, 0.78)))
        torch.testing.assert_close(q - ref[4], u["joints"] * 0.1)
        self.assertTrue(torch.equal(qd, ref[5]))
        # a world-frame rotation vector premultiplied on the root rotation
        torch.testing.assert_close(R, so3_exp(u["rotation"] * h((0.1, 0.1, 0.2))) @ ref[1])

    def test_zero_uniforms_give_the_reference_exactly(self):
        ref = ref_of([0, 6, 10, 14])
        zeros = {k: torch.zeros_like(x) for k, x in unit(4).items()}
        for g, r in zip(placed_state(ref, DomainRandCfg(), False, zeros), ref):
            self.assertTrue(torch.equal(g, r))

    def test_uniforms_with_joint_velocity_noise_are_refused(self):
        with self.assertRaisesRegex(ValueError, "joint velocities"):
            placed_state(ref_of([0]), DomainRandCfg(reset_joint_vel=0.1), False, unit(1))


class TestReset(unittest.TestCase):
    def test_reset_without_the_added_arguments_writes_the_release_draws(self):
        env, ids, frames = stand_in_sim(5, noise=True), torch.tensor([0, 2, 4]), torch.tensor([1, 7, 11])
        torch.manual_seed(5)
        TrackingSim.reset(env, ids, frames)
        torch.manual_seed(5)
        p, R, v, w, q, qd = release_draws(ref_of(frames.tolist()), DomainRandCfg())
        q = torch.clamp(q, KIN.q_lo, KIN.q_hi)
        wr = env.robot.writes
        self.assertTrue(torch.equal(wr["write_root_link_pose_to_sim"][0], torch.cat((p, mat_to_quat(R)), -1)))
        self.assertTrue(torch.equal(wr["write_root_link_velocity_to_sim"][0], torch.cat((v, w), -1)))
        self.assertTrue(torch.equal(wr["write_joint_position_to_sim"][0], q))
        self.assertTrue(torch.equal(wr["write_joint_velocity_to_sim"][0], qd))
        self.assertTrue(torch.equal(env.f[ids], frames))

    def test_body_row_apart_from_the_command_row(self):
        env, ids = stand_in_sim(4, noise=False), torch.tensor([3, 1])
        command, body, u = torch.tensor([2, 3]), torch.tensor([15, 16]), unit(2, seed=3)
        before = torch.get_rng_state()
        TrackingSim.reset(env, ids, command, u, body)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        p, R, v, w, q, qd = placed_state(ref_of(body.tolist()), DomainRandCfg(), False, u)
        wr = env.robot.writes
        self.assertTrue(torch.equal(wr["write_root_link_pose_to_sim"][0], torch.cat((p, mat_to_quat(R)), -1)))
        self.assertTrue(torch.equal(wr["write_root_link_velocity_to_sim"][0], torch.cat((v, w), -1)))
        self.assertTrue(torch.equal(wr["write_joint_position_to_sim"][0], torch.clamp(q, KIN.q_lo, KIN.q_hi)))
        self.assertTrue(torch.equal(wr["write_root_link_pose_to_sim"][1], ids))
        self.assertTrue(torch.equal(env.f[ids], command))
        self.assertTrue(torch.equal(env.start_bin[ids], MOTION.frame_bin[command]))


if __name__ == "__main__":
    unittest.main()
