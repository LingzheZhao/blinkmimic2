"""The mixture-of-experts (MoE) actor (train.py --moe, src/blinkmimic2/moe.py, RECOVERY.md) on the CPU: the path without the
option, the parameter counts, what the mixing network reads, the mixing targets and the 0.8 rule in the simulator's resets and
step, the loss values on hand-built cases, the gradients of the losses, the checkpoint round trip with the EMA and
tools/avg_ckpt.py, and the policy as tools/rollouts.py builds and drives it.

The imagined step with the experts, the captured update and the log columns need a GPU."""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path[:0] = [HERE, TOOLS]
import rollouts  # noqa: E402
import test_recovery as tr  # noqa: E402
import test_rollouts  # noqa: E402
from synthetic import ROBOT  # noqa: E402

import blinkmimic2.env as env_module  # noqa: E402
from blinkmimic2 import moe, recovery  # noqa: E402
from blinkmimic2.buffer import ReplayBuffer, fields  # noqa: E402
from blinkmimic2.env import TrackingSim  # noqa: E402
from blinkmimic2.mdp import obs_dim, proprio_dim  # noqa: E402
from blinkmimic2.networks import Policy, mlp, policy_kwargs  # noqa: E402
from blinkmimic2.optim import MuonAdam  # noqa: E402

J = ROBOT.num_joints
MOTION, MDP_, REC = tr.MOTION, tr.MDP_, tr.REC
OBS, GOAL, HIST = MDP_.obs_dim, MOTION.goal_dim(10), 10
SMALL = {"enc_hidden": (32, 16), "dec_hidden": (64, 32)}
TRAIN_ARGS = {"pi_ref_skip": False, "goal_frames": 10, "hist": HIST, "enc_hidden": "32,16", "dec_hidden": "64,32",
              "amp": "none"}  # fmt: skip


def moe_policy(seed=0, **kw):
    torch.manual_seed(seed)
    return Policy(OBS, GOAL, J, hist=HIST, **(SMALL | kw), **policy_kwargs(TRAIN_ARGS | {"moe": True}, ROBOT))


def inputs(B=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, HIST, OBS, generator=g), torch.randn(B, GOAL, generator=g)


def with_random_mixing_output(pi, seed=2):
    """Gives the mixing network's last layer random weights (it starts at zero)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        pi.mixing_network[-1].weight.copy_(torch.randn(pi.mixing_network[-1].weight.shape, generator=g))
    return pi


def count(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


class TestOffPath(unittest.TestCase):
    def test_policy_kwargs(self):
        self.assertEqual(policy_kwargs({"pi_ref_skip": False}, ROBOT), {"ref_skip": None})
        self.assertEqual(policy_kwargs({"pi_ref_skip": False, "moe": True}, ROBOT),
                         {"ref_skip": None, "moe_prop_dim": proprio_dim(J)})

    def test_the_policy_is_the_release_policy(self):
        # the release's construction (networks.Policy before the option), from the same seed
        torch.manual_seed(3)
        enc, dec = mlp(GOAL, [32, 16], 32), mlp(32 + OBS * HIST, [64, 32], J)
        with torch.no_grad():
            dec[-1].weight.mul_(0.01)
            dec[-1].bias.zero_()
        after_release = torch.get_rng_state()
        torch.manual_seed(3)
        pi = Policy(OBS, GOAL, J, hist=HIST, **SMALL, **policy_kwargs({"pi_ref_skip": False}, ROBOT))
        self.assertTrue(torch.equal(torch.get_rng_state(), after_release))
        want = {f"enc.{k}": v for k, v in enc.state_dict().items()} | {f"dec.{k}": v for k, v in dec.state_dict().items()}
        sd = pi.state_dict()
        self.assertEqual(list(sd), list(want) + ["obs_norm.mean", "obs_norm.var", "obs_norm.count", "goal_norm.mean",
                                                 "goal_norm.var", "goal_norm.count"])
        for k, v in want.items():
            self.assertTrue(torch.equal(sd[k], v), k)
        self.assertEqual(pi.mlps(), [pi.enc, pi.dec])
        o, g = inputs()
        z = torch.tanh(enc(pi.goal_norm(g)))
        self.assertTrue(torch.equal(pi(o, g), dec(torch.cat((z, pi.obs_norm(o).flatten(-2)), -1))))
        self.assertFalse(hasattr(pi, "mixing_network"))

    def test_the_world_model_keeps_its_regression_targets(self):
        """WorldModel.targets is read by train.py's collection step (a GPU-only path)."""
        from blinkmimic2.networks import WorldModel

        wm = WorldModel(J, MDP_.xi_dim, hidden=(8,), structured=True)
        S, S1 = tr.state_of([31, 32]), tr.state_of([32, 33])
        self.assertEqual(wm.targets(S, S1).shape, (2, wm.out_dim))

    def test_buffer_fields(self):
        self.assertEqual(list(fields(J, MDP_.xi_dim, recovery=True)), tr.RELEASE_FIELDS + list(recovery.KEYS))
        self.assertEqual(list(fields(J, MDP_.xi_dim, recovery=True, tracking_target_flag=True)),
                         tr.RELEASE_FIELDS + list(recovery.KEYS) + [moe.FLAG])

    def test_simulator_step_and_resets_without_the_flag(self):
        f, a = torch.tensor([30, 32]), torch.randn(2, J)
        env = tr.step_stand_in(tr.state_of([2, 32]), tr.state_of([3, 33]), f, tr.zero_regime(2))
        env.keeps_tracking_target = None  # as TrackingSim without tracking_target_flag
        out = TrackingSim.step_core(env, a)
        self.assertNotIn(moe.FLAG, out)
        # the resets draw the same numbers with and without the flag
        N, ids, frames = 12, torch.arange(0, 12, 2), 30 + torch.arange(6) % 12
        states = []
        for flag in (False, True):
            env = tr.reset_stand_in(N, True, frames)
            env.keeps_tracking_target = torch.zeros(N, dtype=torch.bool) if flag else None
            torch.manual_seed(4)
            TrackingSim.reset(env, ids)
            states.append((torch.get_rng_state(), env.robot.writes["write_root_link_pose_to_sim"][0]))
        self.assertTrue(torch.equal(states[0][0], states[1][0]))
        self.assertTrue(torch.equal(states[0][1], states[1][1]))


class TestOption(unittest.TestCase):
    def test_moe_without_recovery_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = tr.train(os.path.join(d, "run"), *tr.preset("recovery_control"), "--moe")
            self.assertEqual(r.returncode, 2, r.stderr[-2000:])
            self.assertIn("--moe needs the recovery options", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "run")))

    def test_args_json(self):
        with tempfile.TemporaryDirectory() as d:
            missing = ["--bones_list", os.path.join(d, "missing.txt")]
            r = tr.train(os.path.join(d, "single"), *tr.preset("recovery"), *missing)
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            r = tr.train(os.path.join(d, "moe"), *tr.preset("recovery"), "--moe", *missing)
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            self.assertNotIn("refused", r.stderr)
            single = json.load(open(os.path.join(d, "single", "args.json")))
            run = json.load(open(os.path.join(d, "moe", "args.json")))
        self.assertNotIn("moe", single)
        self.assertEqual(set(run) - set(single), {"moe"})
        self.assertIs(run["moe"], True)
        strip = lambda a_: {k: v for k, v in a_.items() if k not in ("moe", "log_dir")}  # noqa: E731
        self.assertEqual(strip(run), strip(single))


class TestExpertWidths(unittest.TestCase):
    """--moe_expert_hidden: the experts' hidden widths; absent, every --dec_hidden width halved."""

    FULL = (2048, 1024, 512, 256)

    def test_policy_kwargs(self):
        a_ = {"pi_ref_skip": False, "moe": True}
        self.assertNotIn("expert_hidden", policy_kwargs(a_, ROBOT))
        self.assertEqual(policy_kwargs(a_ | {"moe_expert_hidden": "2048,1024,512,256"}, ROBOT)["expert_hidden"], self.FULL)

    def test_absent_is_the_half_widths(self):
        widths = {"enc_hidden": (64, 32), "dec_hidden": self.FULL}
        torch.manual_seed(5)
        default = Policy(OBS, GOAL, J, hist=HIST, **widths, moe_prop_dim=proprio_dim(J))
        after_default = torch.get_rng_state()
        torch.manual_seed(5)
        half = Policy(OBS, GOAL, J, hist=HIST, **widths, moe_prop_dim=proprio_dim(J), expert_hidden=(1024, 512, 256, 128))
        self.assertTrue(torch.equal(torch.get_rng_state(), after_default))
        a, b = default.state_dict(), half.state_dict()
        self.assertEqual(list(a), list(b))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_full_widths(self):
        widths = {"enc_hidden": (2048, 1024, 512, 256), "dec_hidden": self.FULL}
        single = Policy(OBS, GOAL, J, hist=HIST, **widths)
        full = Policy(OBS, GOAL, J, hist=HIST, **widths, moe_prop_dim=proprio_dim(J), expert_hidden=self.FULL)
        self.assertEqual([m.out_features for m in full.expert_r if isinstance(m, torch.nn.Linear)], list(self.FULL) + [J])
        self.assertEqual(count(full.expert_m), count(single.dec))
        print(f"\nparameter counts with full-width experts: actor {count(full)}, each expert {count(full.expert_m)}",
              file=sys.stderr)

    def test_wider_single_network_decoder(self):
        """--dec_hidden 3200,1600,800,400: the narrowest widths in the default's 2:1 ratios (multiples of 8) at which
        the single network has at least as many parameters as the MoE actor with full-width experts."""
        enc = (2048, 1024, 512, 256)
        moe_full = count(Policy(OBS, GOAL, J, hist=HIST, enc_hidden=enc, dec_hidden=self.FULL,
                                moe_prop_dim=proprio_dim(J), expert_hidden=self.FULL))
        size = lambda w: count(Policy(OBS, GOAL, J, hist=HIST, enc_hidden=enc, dec_hidden=(w, w // 2, w // 4, w // 8)))  # noqa: E731
        wider = size(3200)
        print(f"\nparameter counts: single network with decoder 3200,1600,800,400: {wider}; MoE actor with full-width "
              f"experts: {moe_full}; with decoder 3192,1596,798,399: {size(3192)}", file=sys.stderr)
        self.assertGreaterEqual(wider, moe_full)
        self.assertLess(size(3192), moe_full)
        with tempfile.TemporaryDirectory() as d:
            r = tr.train(os.path.join(d, "wider"), *tr.preset("recovery"), "--dec_hidden", "3200,1600,800,400",
                         "--bones_list", os.path.join(d, "missing.txt"))
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            self.assertEqual(json.load(open(os.path.join(d, "wider", "args.json")))["dec_hidden"], "3200,1600,800,400")

    def test_widths_without_the_moe_actor_are_refused(self):
        with self.assertRaises(ValueError):
            Policy(OBS, GOAL, J, hist=HIST, **SMALL, expert_hidden=(8, 8))

    def test_the_option(self):
        with tempfile.TemporaryDirectory() as d:
            r = tr.train(os.path.join(d, "alone"), *tr.preset("recovery"), "--moe_expert_hidden", "2048,1024,512,256")
            self.assertEqual(r.returncode, 2, r.stderr[-2000:])
            self.assertIn("--moe_expert_hidden needs --moe", r.stderr)
            r = tr.train(os.path.join(d, "full"), *tr.preset("recovery"), "--moe", "--moe_expert_hidden", "2048,1024,512,256",
                         "--bones_list", os.path.join(d, "missing.txt"))
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            run = json.load(open(os.path.join(d, "full", "args.json")))
        self.assertEqual(run["moe_expert_hidden"], "2048,1024,512,256")


class TestNetworks(unittest.TestCase):
    def test_parameter_counts(self):
        """The actor at train.py's widths on the 40-clip library's dimensions (goal of 10 rows): the encoder is shared;
        the MoE actor has fewer parameters than the single network."""
        widths = {"enc_hidden": (2048, 1024, 512, 256), "dec_hidden": (2048, 1024, 512, 256)}
        single = Policy(OBS, GOAL, J, hist=HIST, **widths)
        two = Policy(OBS, GOAL, J, hist=HIST, **widths, moe_prop_dim=proprio_dim(J))
        n_single, n_moe = count(single), count(two)
        counts = {"obs": OBS, "proprio": proprio_dim(J), "goal": GOAL, "encoder": count(single.enc),
                  "decoder": count(single.dec), "expert": count(two.expert_m), "mixing_network": count(two.mixing_network),
                  "single": n_single, "moe": n_moe}  # fmt: skip
        print(f"\nparameter counts: {json.dumps(counts)}", file=sys.stderr)
        self.assertEqual(count(two.enc), count(single.enc))
        self.assertEqual(n_moe, count(two.enc) + 2 * count(two.expert_m) + count(two.mixing_network))
        self.assertLess(n_moe, n_single)
        self.assertEqual([m.out_features for m in two.expert_m if isinstance(m, torch.nn.Linear)], [1024, 512, 256, 128, J])
        self.assertEqual([m.out_features for m in two.mixing_network if isinstance(m, torch.nn.Linear)], [256, 128, 2])
        self.assertEqual(two.mixing_network[0].in_features, HIST * proprio_dim(J))
        self.assertEqual(two.expert_m[0].in_features, single.dec[0].in_features)

    def test_initialization(self):
        pi = moe_policy()
        o, g = inputs()
        a, z, on, log_w = pi.moe_forward(o, g)
        self.assertTrue(torch.equal(log_w.exp(), torch.full((5, 2), 0.5)))
        mu_m, mu_r = pi.experts(z, on)
        torch.testing.assert_close(a, 0.5 * mu_m + 0.5 * mu_r)
        self.assertFalse(torch.equal(mu_m, mu_r))  # two experts initialized apart
        self.assertTrue(torch.equal(pi(o, g), a))

    def test_the_observation_layout(self):
        # the observation's last 6 entries are the only ones that read the command row
        self.assertEqual(proprio_dim(J), obs_dim(J) - 6)
        S, xi, a_prev = tr.state_of([31, 33]), torch.zeros(2, MDP_.xi_dim), torch.randn(2, J)
        o1, o2 = MDP_.obs(S, a_prev, xi, torch.tensor([31, 33])), MDP_.obs(S, a_prev, xi, torch.tensor([3, 13]))
        self.assertTrue(torch.equal(o1[:, : proprio_dim(J)], o2[:, : proprio_dim(J)]))
        self.assertFalse(torch.equal(o1[:, proprio_dim(J):], o2[:, proprio_dim(J):]))

    def test_the_mixing_network_reads_proprioception_only(self):
        pi = with_random_mixing_output(moe_policy())
        pi.obs_norm.update(torch.randn(100, OBS))
        o, g = inputs()
        a, _, _, log_w = pi.moe_forward(o, g)
        # another goal: the same mixing weights, another action
        a2, _, _, log_w2 = pi.moe_forward(o, g + 1.0)
        self.assertTrue(torch.equal(log_w, log_w2))
        self.assertFalse(torch.equal(a, a2))
        # another reference-relative orientation (the observation's last 6 entries) in every frame: the same weights
        o3 = o.clone()
        o3[..., proprio_dim(J):] += 1.0
        self.assertTrue(torch.equal(pi.moe_forward(o3, g)[3], log_w))
        # another latent: the same weights
        self.assertTrue(torch.equal(pi.moe_forward(o, g, z=torch.zeros(5, 32))[3], log_w))
        # every proprioceptive entry of every frame reaches the mixing network
        o4 = o.clone().requires_grad_(True)
        pi.moe_forward(o4, g)[3][:, 1].sum().backward()
        self.assertTrue(bool((o4.grad[..., : proprio_dim(J)] != 0).all()))
        self.assertTrue(bool((o4.grad[..., proprio_dim(J):] == 0).all()))

    def test_optimizer_groups(self):
        single = Policy(OBS, GOAL, J, hist=HIST, **SMALL)
        pi = moe_policy()
        hidden = lambda m: [l.weight for l in m if isinstance(l, torch.nn.Linear)][1:-1]  # noqa: E731
        for p_, mlps in ((single, [single.enc, single.dec]), (pi, [pi.enc, pi.expert_m, pi.expert_r, pi.mixing_network])):
            opt = MuonAdam(p_, p_.mlps(), lr=1e-3, fused=False)
            muon = {id(p) for g in opt.muon.param_groups for p in g["params"]}
            self.assertEqual(muon, {id(w) for m in mlps for w in hidden(m)})
            self.assertEqual(len(muon) + sum(len(g["params"]) for g in opt.adam.param_groups), len(list(p_.parameters())))


class TestTargets(unittest.TestCase):
    def test_targets_by_regime(self):
        BL = 75
        left = torch.tensor([0, 0, 75, 74, 38, 1, 75])
        recovering = torch.tensor([False, True, False, False, False, False, True])
        label, y_r = moe.mixing_targets(recovering, left, BL, torch.zeros(7, dtype=torch.bool))
        T, R, B = recovery.TRACKING, recovery.RECOVERY, recovery.BLEND
        self.assertEqual(label.tolist(), [T, R, B, B, B, B, R])
        torch.testing.assert_close(y_r, torch.tensor([0.0, 1.0, 74 / 75, 73 / 75, 37 / 75, 0.0, 1.0]))
        # on blend controls the tracking target is the blend control's reward weight (recovery.advance)
        scale = (BL - left + 1) / BL
        b = label == B
        torch.testing.assert_close(1 - y_r[b], scale[b].float())
        # a flagged episode keeps the tracking target in every regime
        label, y_r = moe.mixing_targets(recovering, left, BL, torch.ones(7, dtype=torch.bool))
        self.assertEqual(label.tolist(), [T] * 7)
        self.assertEqual(y_r.tolist(), [0.0] * 7)

    def test_targets_over_a_rollout(self):
        # (H, B) regimes with a per-rollout flag broadcast over time, as train.py's policy_update
        recovering = torch.tensor([[True, True], [True, True], [False, False]])
        left = torch.tensor([[0, 0], [0, 0], [75, 75]])
        label, y_r = moe.mixing_targets(recovering, left, 75, torch.tensor([False, True]).unsqueeze(0))
        self.assertEqual(label[:, 0].tolist(), [recovery.RECOVERY, recovery.RECOVERY, recovery.BLEND])
        self.assertEqual(label[:, 1].tolist(), [recovery.TRACKING] * 3)
        self.assertEqual(y_r[:, 1].tolist(), [0.0] * 3)

    def test_the_rule_at_reset(self):
        recovering = torch.tensor([True, True, True, False])
        body, command = torch.tensor([0.62, 0.60, 0.70, 0.70]), torch.full((4,), 0.76)
        self.assertEqual(moe.tracking_target_at_reset(recovering, body, command).tolist(), [True, False, True, False])

    def test_resets_set_the_flag(self):
        N, ids, frames = 40, torch.arange(0, 40, 2), 30 + torch.arange(20) % 12
        for ratio in (0.8, 0.1):  # at 0.1 every recovery start of the generated library is above the ratio
            with self.subTest(ratio=ratio), mock.patch.object(moe, "RESET_HEIGHT_RATIO", ratio):
                env = tr.reset_stand_in(N, True, frames)
                env.keeps_tracking_target = torch.ones(N, dtype=torch.bool)
                torch.manual_seed(6)
                TrackingSim.reset(env, ids)
                p = env.robot.writes["write_root_link_pose_to_sim"][0][:, :3]
                rec = env.regime["recovering"][ids]
                self.assertTrue(0 < int(rec.sum()) < 20)
                want = rec & (p[:, 2] >= ratio * MOTION.root_p[env.f[ids]][:, 2])
                self.assertTrue(torch.equal(env.keeps_tracking_target[ids], want))
                if ratio == 0.1:  # every placed recovery start is above 0.1 of its command row's height
                    self.assertTrue(torch.equal(want, rec))
                self.assertTrue(bool(env.keeps_tracking_target[1::2].all()))  # worlds not reset keep their flag

    def test_masked_reset_sets_the_flag(self):
        N = 16
        frames, mask = 30 + torch.arange(N) % 12, torch.arange(N) % 4 != 1
        with mock.patch.object(env_module, "wp", tr.NO_WARP), mock.patch.object(moe, "RESET_HEIGHT_RATIO", 0.1):
            env = tr.masked_stand_in(N, True, frames)
            env.keeps_tracking_target = torch.ones(N, dtype=torch.bool)
            torch.manual_seed(8)
            TrackingSim.reset_masked(env, mask)
        rec = env.regime["recovering"]
        p = env.sim.data.qpos[:, :3]
        want = rec & (p[:, 2] >= 0.1 * MOTION.root_p[env.f][:, 2])
        self.assertTrue(0 < int(want.sum()) < int(mask.sum()))
        self.assertTrue(torch.equal(env.keeps_tracking_target, torch.where(mask, want, True)))

    def test_the_step_stores_the_flag(self):
        f, a = torch.tensor([30, 32]), torch.randn(2, J)
        env = tr.step_stand_in(tr.state_of([2, 32]), tr.state_of([3, 33]), f, tr.zero_regime(2))
        env.keeps_tracking_target = torch.tensor([True, False])
        out = TrackingSim.step_core(env, a)
        self.assertEqual(out[moe.FLAG].tolist(), [True, False])
        self.assertNotEqual(out[moe.FLAG].data_ptr(), env.keeps_tracking_target.data_ptr())

    def test_a_start_carries_the_flag(self):
        T, N = 3, 4
        buf = ReplayBuffer(T, N, "cpu", J, MDP_.xi_dim, recovery=True, tracking_target_flag=True)
        buf.use_device_ptr(device_size=True)
        for t in range(T):
            row = {k: torch.zeros((N,) + ((shape,) if isinstance(shape, int) else shape), dtype=dt)
                   for k, (shape, dt) in buf.fields.items()}  # fmt: skip
            row[moe.FLAG] = (torch.arange(N) + t) % 3 == 0
            buf.add_device(row)
            buf.advance_host()
        age, env = torch.tensor([0, 2, 1, 2]), torch.tensor([0, 1, 2, 3])
        start, _ = buf.starts_at(age, env, 2)
        self.assertEqual(start[moe.FLAG].tolist(), ((env + age) % 3 == 0).tolist())
        plain = ReplayBuffer(T, N, "cpu", J, MDP_.xi_dim, recovery=True)
        plain.use_device_ptr(device_size=True)
        self.assertNotIn(moe.FLAG, plain.starts_at(age, env, 2)[0])


T_, R_, B_ = recovery.TRACKING, recovery.RECOVERY, recovery.BLEND


class TestLosses(unittest.TestCase):
    def case(self):
        """H = 3 controls of B = 2 rollouts: rollout 0 tracking, tracking, recovery; rollout 1 recovery, blend, blend."""
        w_r = torch.tensor([[0.2, 0.6], [0.5, 0.7], [0.9, 0.3]])
        log_w = torch.stack((torch.log(1 - w_r), torch.log(w_r)), -1)
        label = torch.tensor([[T_, R_], [T_, B_], [R_, B_]])
        y_r = torch.tensor([[0.0, 1.0], [0.0, 0.5], [1.0, 0.25]])
        mu_m = torch.zeros(3, 2, 4)
        mu_r = torch.zeros(3, 2, 4)
        mu_r[1, 1] = torch.tensor([1.0, 1.0, 0.0, 0.0])  # blend control: mean squared difference 0.5
        mu_r[2, 1] = torch.tensor([2.0, 0.0, 0.0, 0.0])  # blend control: 1.0
        mu_r[0, 0] = 5.0  # a tracking control: no alignment
        return w_r, log_w, label, y_r, mu_m, mu_r

    def test_values(self):
        w_r, log_w, label, y_r, mu_m, mu_r = self.case()
        ce, cons, align = moe.mixing_losses(log_w, y_r, label, mu_m, mu_r, torch.ones(3, 2))
        L = math.log
        per = [[-L(0.8), -L(0.6)], [-L(0.5), -(0.5 * L(0.3) + 0.5 * L(0.7))], [-L(0.9), -(0.75 * L(0.7) + 0.25 * L(0.3))]]
        weights = [[1, 1], [1, 4], [1, 4]]
        want_ce = sum(per[t][b] * weights[t][b] for t in range(3) for b in range(2)) / 6
        self.assertAlmostEqual(float(ce), want_ce, places=5)
        # equal-label pairs: rollout 0 (0, 1); rollout 1 (1, 2)
        self.assertAlmostEqual(float(cons), ((0.5 - 0.2) ** 2 + (0.3 - 0.7) ** 2) / 2, places=6)
        self.assertAlmostEqual(float(align), (0.5 + 1.0) / 2, places=6)
        self.assertAlmostEqual(float(moe.total(ce, cons, align)), 0.1 * float(ce) + 0.01 * float(cons) + 0.01 * float(align),
                               places=6)

    def test_survival_weights(self):
        w_r, log_w, label, y_r, mu_m, mu_r = self.case()
        s = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])  # the last blend control has not survived
        ce, cons, align = moe.mixing_losses(log_w, y_r, label, mu_m, mu_r, s)
        self.assertAlmostEqual(float(align), 0.5, places=6)
        self.assertAlmostEqual(float(cons), (0.5 - 0.2) ** 2, places=6)
        full = moe.mixing_losses(log_w, y_r, label, mu_m, mu_r, torch.ones(3, 2))[0]
        self.assertNotAlmostEqual(float(ce), float(full), places=4)

    def test_no_blend_control_and_no_pair(self):
        log_w = torch.log(torch.full((2, 1, 2), 0.5))
        label = torch.tensor([[T_], [R_]])
        ce, cons, align = moe.mixing_losses(log_w, torch.tensor([[0.0], [1.0]]), label, torch.zeros(2, 1, 3), torch.ones(2, 1, 3),
                                     torch.ones(2, 1))
        self.assertAlmostEqual(float(ce), float(torch.log(torch.tensor(2.0))), places=6)
        self.assertEqual((float(cons), float(align)), (0.0, 0.0))

    def test_saturated_mixing_network(self):
        # a mixing network saturated on its target gives a cross-entropy near 0 and finite gradients
        logits = torch.tensor([[[-30.0, 30.0]]], requires_grad=True)
        log_w = torch.log_softmax(logits, -1)
        ce, _, _ = moe.mixing_losses(log_w, torch.ones(1, 1), torch.tensor([[R_]]), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2),
                              torch.ones(1, 1))
        ce.backward()
        self.assertLess(float(ce), 1e-12)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))


class TestStats(unittest.TestCase):
    def test_log_summands(self):
        """train.py's log columns divide the per-regime sums by the counts, the losses by the updates and the flagged
        recovery controls by the recovery controls (indices 0-5, 6-8 over 9, 10 over 3)."""
        w_r = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.5, 0.6]])
        regimes = torch.tensor([[T_, R_], [T_, R_], [B_, R_]])
        out = moe.stats(w_r, regimes, torch.tensor([False, True]).unsqueeze(0), torch.tensor(0.7), torch.tensor(0.02),
                        torch.tensor(0.3))
        self.assertEqual(out.shape, (moe.STATS,))
        want = [0.3, 2, 2.3, 3, 0.5, 1, 0.7, 0.02, 0.3, 1, 3]
        torch.testing.assert_close(out, torch.tensor(want))
        acc = torch.zeros(moe.STATS)
        acc.add_(out)  # as MOE_STATS
        self.assertAlmostEqual(float(acc[2] / acc[3]), 2.3 / 3, places=6)


class TestGradients(unittest.TestCase):
    def rollout(self, pi, H=3, B=4):
        """The imagined rollout's inputs of rollout_mixing_losses: latents, and histories that depend on a stand-in for
        the world model's state (``x``)."""
        o, g = inputs(B * H, seed=5)
        x = torch.ones((), requires_grad=True)
        O = (o * x).view(H, B, HIST, OBS)
        Z = pi.encode(g).view(H, B, -1)
        return x, Z, O

    def grads(self, pi):
        return {name: (sum(float(p.grad.norm() ** 2) for p in m.parameters() if p.grad is not None)) ** 0.5
                for name, m in (("enc", pi.enc), ("expert_m", pi.expert_m), ("expert_r", pi.expert_r), ("mixing_network", pi.mixing_network))}

    def test_the_mixing_cross_entropy_trains_the_mixing_network_only(self):
        pi = with_random_mixing_output(moe_policy())
        x, Z, O = self.rollout(pi)
        label = torch.full((3, 4), R_)
        ce, cons, align, w_r = moe.rollout_mixing_losses(pi, Z, O, label, torch.ones(3, 4), torch.ones(3, 4))
        self.assertFalse(w_r.requires_grad)
        (ce + cons).backward()
        g = self.grads(pi)
        self.assertGreater(g["mixing_network"], 0.0)
        self.assertEqual((g["enc"], g["expert_m"], g["expert_r"]), (0.0, 0.0, 0.0))
        self.assertIsNone(x.grad)  # no gradient to the imagined states

    def test_the_expert_alignment_trains_the_experts_and_the_encoder(self):
        pi = with_random_mixing_output(moe_policy())
        x, Z, O = self.rollout(pi)
        label = torch.full((3, 4), B_)
        _, _, align, _ = moe.rollout_mixing_losses(pi, Z, O, label, torch.full((3, 4), 0.5), torch.ones(3, 4))
        align.backward()
        g = self.grads(pi)
        self.assertEqual(g["mixing_network"], 0.0)
        self.assertTrue(g["expert_m"] > 0 and g["expert_r"] > 0 and g["enc"] > 0)
        self.assertIsNone(x.grad)

    def test_a_saturated_mixing_network_routes_a_recovery_batch_to_the_recovery_expert(self):
        """The mixing network saturated on recovery (w_r = 1 - 2e-9) on a batch of recovery states: the action objective (here a
        squared distance to a target action) and one optimizer step move the recovery expert, the tracking expert
        hardly; with mixing weights of 1/2 both move alike."""
        o, g = inputs(64, seed=7)
        target = torch.randn(64, J)
        moved = {}
        for bias in (10.0, 0.0):
            pi = moe_policy()
            with torch.no_grad():
                pi.mixing_network[-1].bias.copy_(torch.tensor([-bias, bias]))
            before = {k: [p.detach().clone() for p in getattr(pi, k).parameters()] for k in ("expert_m", "expert_r")}
            opt = torch.optim.SGD(pi.parameters(), lr=0.1)
            ((pi(o, g) - target) ** 2).sum(-1).mean().backward()
            opt.step()
            moved[bias] = {k: sum(float((p - q).norm() ** 2) for p, q in zip(getattr(pi, k).parameters(), before[k])) ** 0.5
                           for k in before}  # fmt: skip
        self.assertGreater(moved[10.0]["expert_r"], 1e6 * moved[10.0]["expert_m"])
        self.assertLess(abs(moved[0.0]["expert_r"] / moved[0.0]["expert_m"] - 1), 0.5)


class TestCheckpoints(unittest.TestCase):
    def test_round_trip_ema_and_avg_ckpt(self):
        pi = with_random_mixing_output(moe_policy())
        pi.obs_norm.update(torch.randn(50, OBS))
        pi.goal_norm.update(torch.randn(50, GOAL))
        # the EMA of train.py (AveragedModel with buffers, then ema_update's lerp over parameters and buffers)
        avg = torch.optim.swa_utils.AveragedModel(pi, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995),
                                                  use_buffers=True)
        avg.update_parameters(pi)
        tensors = list(avg.module.parameters()) + list(avg.module.buffers())
        self.assertTrue(all(t.is_floating_point() for t in tensors))
        with torch.no_grad():  # as train.py's ema_update
            torch._foreach_lerp_(tensors, [t.detach() for t in list(pi.parameters()) + list(pi.buffers())], 0.005)
        o, g = inputs()
        args = TRAIN_ARGS | {"moe": True, "robot": "mjlab_g1"}
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "run")
            os.makedirs(run)
            json.dump(args, open(os.path.join(run, "args.json"), "w"))
            for it in (500, 1000):
                torch.save({"pi": {k: v.clone() for k, v in avg.module.state_dict().items()}, "iteration": it, "args": args},
                           os.path.join(run, f"ckpt_{it:06d}.pt"))
            ck = torch.load(os.path.join(run, "ckpt_001000.pt"), map_location="cpu", weights_only=False)
            # built as tools/rollouts.py and tools/evaluate.py build the policy
            a_ = ck["args"]
            widths = lambda key: tuple(int(x) for x in a_[key].split(","))  # noqa: E731
            loaded = Policy(MDP_.obs_dim, MOTION.goal_dim(a_["goal_frames"]), ROBOT.num_joints, hist=a_["hist"],
                            enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"), **policy_kwargs(a_, ROBOT))
            loaded.load_state_dict(ck["pi"])
            self.assertTrue(torch.equal(loaded(o, g), avg.module(o, g)))
            with self.assertRaises(RuntimeError):  # the same widths without the option cannot load it
                Policy(OBS, GOAL, J, hist=HIST, **SMALL).load_state_dict(ck["pi"])
            out = os.path.join(d, "avg")
            r = subprocess.run([sys.executable, os.path.join(TOOLS, "avg_ckpt.py"), "--run_dir", run, "--ckpts", "500,1000",
                                "--out_dir", out], capture_output=True, text=True)  # fmt: skip
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            ck2 = torch.load(os.path.join(out, "ckpt_001000.pt"), map_location="cpu", weights_only=False)
            loaded.load_state_dict(ck2["pi"])
            self.assertTrue(torch.equal(loaded(o, g), avg.module(o, g)))

    def test_rollout_tool_drives_the_policy(self):
        pi = with_random_mixing_output(Policy(MDP_.obs_dim, test_rollouts.MOTION.goal_dim(4), J, hist=3, enc_hidden=(16,), dec_hidden=(16,),
                              **policy_kwargs({"pi_ref_skip": False, "moe": True}, ROBOT)))
        scene = [test_rollouts.trial(1), test_rollouts.trial(2, test_rollouts.LYING + 1, test_rollouts.uniforms(3))]
        rec, _ = test_rollouts.run(scene, 4, policy=pi.eval())
        self.assertEqual(rec["action"].shape, (4, 2, J))
        self.assertTrue(bool(torch.isfinite(torch.from_numpy(rec["action"])).all()))
        self.assertIs(rollouts.Policy, Policy)
        self.assertIs(rollouts.policy_kwargs, policy_kwargs)


if __name__ == "__main__":
    unittest.main()
