"""tools/mujoco_rollouts.py on the CPU: the MuJoCo C simulator object on the robot's model as mjlab compiles it (the
scene of make_scene_cfg with make_sim's options) and on small generated models, driven by rollouts.roll_out on the
synthetic library of synthetic.py; the model built from the training simulator's fields; the checks with a second
MuJoCo C object in the place of the training simulator; the reports; the description and the command line."""

import ast
import copy
import enum
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
sys.path[:0] = [HERE, TOOLS]
import mujoco_rollouts as mr  # noqa: E402
import rollouts  # noqa: E402
from synthetic import ROBOT, library  # noqa: E402
from test_rollouts import StandIn  # noqa: E402

from mjlab.scene import Scene  # noqa: E402

from blinkmimic2.env import DECIMATION, make_scene_cfg, make_sim, placed_state  # noqa: E402
from blinkmimic2.mdp import flatten_state  # noqa: E402
from blinkmimic2.motion import CONTROL_DT  # noqa: E402
from blinkmimic2.networks import Policy  # noqa: E402

KIN, MOTION, MDP_ = library()
J, NAMES = ROBOT.num_joints, list(ROBOT.joint_names)
WALK, LOW, TILTED, LYING = 0, 6, 10, 14  # first rows of the synthetic clips
MODEL = Scene(make_scene_cfg(ROBOT, 1), device="cpu").compile()
make_sim(ROBOT).mujoco.apply(MODEL)
DEFAULT_Q, ALPHA = torch.tensor(ROBOT.default_joint_pos_list()), torch.tensor(ROBOT.action_scale_list())
POLICY = Policy(MDP_.obs_dim, MOTION.goal_dim(4), J, hist=3, enc_hidden=(16,), dec_hidden=(16,)).eval()
with torch.no_grad():
    POLICY.dec[-1].weight.normal_(0.0, 0.05)  # actions of order 0.1, so that the joints move


def xi(T: int, offsets=None):
    """The context of the evaluation (friction 1, joint offsets 0), or the given joint offsets."""
    x = torch.zeros(T, MDP_.xi_dim)
    x[:, 0] = 1.0
    if offsets is not None:
        x[:, 1 : 1 + J] = offsets
    return x


def make(T: int, threads: int = 2, context=None, model=MODEL):
    return mr.MujocoC(model, T, NAMES, MOTION, MDP_, ROBOT.dr, DEFAULT_Q, ALPHA, xi(T) if context is None else context,
                      torch.zeros(T, 3), threads, "cpu")  # fmt: skip


def trial(command_row, body_row=None, uniforms=None, push=None) -> dict:
    t = {"command_row": command_row}
    for k, v in (("body_row", body_row), ("reset_uniforms", uniforms), ("push", push)):
        if v is not None:
            t[k] = v
    return t


def uniforms(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {k: (2 * torch.rand(n, generator=g, dtype=torch.float64) - 1).tolist()
            for k, n in (("position", 3), ("rotation", 3), ("velocity", 6), ("joints", J))}  # fmt: skip


SCENE = [trial(1, uniforms=uniforms(1)), trial(2, LYING + 1, uniforms(2)), trial(3, LOW, push={"control": 1, "velocity_xy": [0.5, -0.2]})]


def read(scene, controls):
    return rollouts.read_scene({"trials": scene}, MOTION.num_frames, J, controls)


def place_all(env, tr):
    T = len(tr["command_row"])
    env.place(torch.arange(T), tr["command_row"], tr["body_row"], {k: tr[k] for k in rollouts.UNIFORMS})


def ctrl_by_name(model, d) -> np.ndarray:
    """ctrl of each joint of the robot, found through the actuator's transmission (actuator_trnid)."""
    out = np.full(J, np.nan)
    for a in range(model.nu):
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0]))
        out[NAMES.index(joint.removeprefix(mr.PREFIX))] = d.ctrl[a]
    return out


SMALL = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="robot/pelvis" pos="0 0 1">
      <freejoint name="robot/root"/>
      <geom type="sphere" size="0.1"/>
      <body name="robot/upper">
        <joint name="robot/a_joint" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03"/>
        <body name="robot/lower" pos="0 0 -0.3">
          <joint name="robot/b_joint" type="hinge" axis="1 0 0" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>ACTUATORS</actuator>
</mujoco>"""
POSITION = '<position name="b" joint="robot/b_joint" kp="10" kv="1"/><position name="a" joint="robot/a_joint" kp="20" kv="2"/>'


def small(actuators: str = POSITION):
    return mujoco.MjModel.from_xml_string(SMALL.replace("ACTUATORS", actuators))


class TestJointMaps(unittest.TestCase):
    def test_robot(self):
        m = mr.joint_maps(MODEL, NAMES)
        for j, name in enumerate(NAMES):
            jid = MODEL.joint(mr.PREFIX + name).id
            self.assertEqual((m["joint_q"][j], m["joint_v"][j]), (MODEL.jnt_qposadr[jid], MODEL.jnt_dofadr[jid]))
            a = m["actuator"][j]
            self.assertEqual(mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_JOINT, int(MODEL.actuator_trnid[a, 0])), mr.PREFIX + name)
        self.assertEqual(sorted([*m["free_q"], *m["joint_q"]]), list(range(MODEL.nq)))
        # the actuators are not in the joints' order, so a map by position would be wrong
        self.assertNotEqual(m["actuator"].tolist(), list(range(J)))

    def test_small_model_by_name(self):
        m = mr.joint_maps(small(), ["a_joint", "b_joint"])
        self.assertEqual(m["actuator"].tolist(), [1, 0])
        self.assertEqual((m["free_q"].tolist(), m["joint_q"].tolist(), m["joint_v"].tolist()), (list(range(7)), [7, 8], [6, 7]))

    def test_refusals(self):
        names = ["a_joint", "b_joint"]
        cases = {
            "missing joint": (small(), ["a_joint", "c_joint"]),
            "motor": (small('<motor joint="robot/a_joint"/><position joint="robot/b_joint" kp="10"/>'), names),
            "two actuators on one joint": (small('<position joint="robot/a_joint" kp="10"/><position joint="robot/a_joint" kp="5"/>'), names),
            "limited ctrl": (small('<position joint="robot/a_joint" kp="10" ctrlrange="-1 1" ctrllimited="true"/>'
                                   '<position joint="robot/b_joint" kp="10"/>'), names),
            "one actuator for two joints": (small('<position joint="robot/a_joint" kp="10"/>'), names),
            "joint names of another order of the robot": (small(), ["b_joint", "a_joint", "a_joint"]),
            "no free joint": (mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><body name="robot/x"><joint name="robot/a_joint" type="hinge"/><geom size="0.1"/>'
                '</body></worldbody><actuator><position joint="robot/a_joint" kp="1"/></actuator></mujoco>'), ["a_joint"]),
        }  # fmt: skip
        for case, (model, joint_names) in cases.items():
            with self.subTest(case), self.assertRaises(ValueError):
                mr.joint_maps(model, joint_names)


class TestTrainerModel(unittest.TestCase):
    def test_written_entries(self):
        m = small()
        worlds = torch.tensor(m.geom_friction, dtype=torch.float32).repeat(3, 1, 1)
        worlds[:, 1, 0] = 0.25  # written after compilation, in every world
        model, written = mr.trainer_model(m, {"geom_friction": worlds})
        self.assertEqual(written, {"geom_friction": {"entries": 1, "values": [0.25]}})
        self.assertEqual(model.geom_friction[1, 0], 0.25)
        # the entries not written keep the compiled float64 value, not its float32 rounding
        np.testing.assert_array_equal(np.delete(model.geom_friction.ravel(), 3), np.delete(m.geom_friction.ravel(), 3))
        self.assertNotEqual(float(np.float32(m.geom_friction[0, 1])), m.geom_friction[0, 1])
        self.assertEqual(m.geom_friction[1, 0], 1.0)  # the given model is not changed
        self.assertEqual(mr.model_differences(model, m), {"geom_friction": 1})

    def test_refusals_of_fields(self):
        m = small()
        f = torch.tensor(m.geom_friction, dtype=torch.float32).repeat(2, 1, 1)
        differ = f.clone()
        differ[1, 0, 0] = 0.5
        for case, fields in {"worlds differ": {"geom_friction": differ}, "shape": {"geom_friction": f[:, :2]},
                             "no worlds": {"geom_friction": f[:0]}, "unknown field": {"geom_frictions": f}}.items():  # fmt: skip
            with self.subTest(case), self.assertRaises(ValueError):
                mr.trainer_model(m, fields)


def warp_like(model, worlds: int = 1) -> dict:
    """The options of ``model`` as MuJoCo Warp's put_model holds them: floats and vectors as float32 [worlds, ...],
    the main tolerance raised to at least 1e-6, impratio as impratio_invsqrt, integers as ints."""
    out = {}
    for k in mr.options(model):
        if k in ("impratio", "noslip_tolerance", "noslip_iterations", "o_margin", "o_solref", "o_solimp", "o_friction",
                 "jacobian"):  # not MuJoCo Warp's options
            continue
        x = getattr(model.opt, k)
        if isinstance(x, (float, np.ndarray)):
            v = max(x, 1e-6) if k == "tolerance" else x
            out[k] = np.repeat(np.asarray(v, np.float32)[None], worlds, 0)
        else:
            out[k] = np.asarray(int(x))
    out["impratio_invsqrt"] = np.repeat(np.float32([1.0 / np.sqrt(max(model.opt.impratio, mujoco.mjMINVAL))]), worlds)
    return out


class TestTrainerOptions(unittest.TestCase):
    def test_tolerance_of_the_training_simulator(self):
        self.assertEqual(MODEL.opt.tolerance, 1e-8)  # the robot's solver setting, below MuJoCo Warp's floor
        model = copy.deepcopy(MODEL)
        written, other = mr.trainer_options(model, warp_like(MODEL, worlds=3))
        self.assertEqual((list(written), other), (["tolerance"], {}))
        mr.check_options(other)
        self.assertEqual(written["tolerance"]["compiled"], 1e-8)
        self.assertEqual(model.opt.tolerance, float(np.float32(1e-6)))
        self.assertEqual(written["tolerance"]["written"], model.opt.tolerance)
        self.assertEqual(MODEL.opt.tolerance, 1e-8)  # the given model is not changed
        self.assertTrue(all(r["equal"] for r in mr.option_comparison(model, warp_like(MODEL)).values()))
        self.assertEqual(mr.model_differences(model, MODEL), {"opt.tolerance": [model.opt.tolerance, 1e-8]})
        # a model whose tolerance MuJoCo Warp keeps gets nothing written
        m = small()
        m.opt.tolerance = 1e-6
        self.assertEqual(mr.trainer_options(m, warp_like(m)), ({}, {}))
        self.assertEqual(m.opt.tolerance, 1e-6)

    def test_comparison(self):
        rows = mr.option_comparison(MODEL, warp_like(MODEL))
        self.assertEqual(rows["tolerance"], {"mujoco_c": 1e-8, "training_simulator": float(np.float32(1e-6)), "equal": False})
        self.assertEqual(rows["integrator"]["mujoco_c"], int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST))
        self.assertTrue(rows["impratio_invsqrt"]["equal"] and rows["gravity"]["equal"] and rows["timestep"]["equal"])
        self.assertEqual(rows["gravity"]["training_simulator"], np.float32(MODEL.opt.gravity).tolist())
        json.dumps(rows, allow_nan=False)

    def test_refusals(self):
        def changed(**values):
            w = warp_like(MODEL, worlds=2)
            w.update(values)
            return w

        # options that differ but are not of WRITTEN_OPTIONS: left as compiled, named, refused by check_options
        for case, values in {"an integer": changed(iterations=np.asarray(MODEL.opt.iterations + 1)),
                             "a float": changed(timestep=np.float32([0.004, 0.004]))}.items():  # fmt: skip
            with self.subTest(case):
                model = copy.deepcopy(MODEL)
                written, other = mr.trainer_options(model, values)
                self.assertEqual((list(written), len(other)), (["tolerance"], 1))
                self.assertEqual((model.opt.iterations, model.opt.timestep), (MODEL.opt.iterations, MODEL.opt.timestep))
                with self.assertRaisesRegex(ValueError, list(other)[0]):
                    mr.check_options(other)
        worlds = np.float32([[0.005], [0.004]])[:, 0]
        cases = {"an option's worlds differ": changed(timestep=worlds),
                 "shape": changed(gravity=np.float32([[0, 0, -9.81, 0]])),
                 "not an option of the MjModel": changed(broadphase_filters=np.asarray(1))}  # fmt: skip
        for case, values in cases.items():
            with self.subTest(case), self.assertRaises(ValueError):
                mr.trainer_options(copy.deepcopy(MODEL), values)


class TestMujocoC(unittest.TestCase):
    def test_placement_is_the_evaluation_reset(self):
        tr = read(SCENE, 4)
        c = make(3)
        place_all(c, tr)
        ref = tuple(x.clone() for x in MOTION.ref_state(tr["body_row"]))
        p, R, v, w, q, qd = placed_state(ref, ROBOT.dr, False, {k: tr[k] for k in rollouts.UNIFORMS})
        S = c.state()
        torch.testing.assert_close(S[0], p, rtol=0, atol=0)
        torch.testing.assert_close(S[1], R)  # through the quaternion of qpos
        torch.testing.assert_close(S[2], v, rtol=0, atol=0)
        torch.testing.assert_close(S[3], w)  # through the body-frame angular velocity of qvel
        torch.testing.assert_close(S[4], torch.clamp(q, KIN.q_lo, KIN.q_hi), rtol=0, atol=0)
        torch.testing.assert_close(S[5], qd, rtol=0, atol=0)
        c.close()

    def test_placing_again_clears_what_the_steps_left(self):
        """A trial placed again after controls, one of them a bad state, holds what a trial placed once holds:
        mj_resetData clears the warm start, ctrl, time and the warning counts, and the next control is the same."""
        scene = [trial(1, push={"control": 0, "velocity_xy": [1e12, 0.0]}), trial(LYING + 1), trial(3, LOW)]
        tr = read(scene, 4)
        c, once = make(3), make(3)
        place_all(c, tr)
        c.push(torch.tensor([0]), tr["push_velocity_xy"][:1])
        g = torch.Generator().manual_seed(1)
        for _ in range(3):
            c.step(0.2 * torch.randn(3, J, generator=g), tr["command_row"])
        self.assertTrue(c.bad[0])
        self.assertTrue(all(d.qacc_warmstart.any() and d.ctrl.any() and d.time > 0 for d in c.data[1:]))
        # warning counts as the steps leave them (an arena-full count, which this model does not reach, written here)
        c.data[1].warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number = 2
        self.assertTrue(any(d.warning[w].number for d in c.data for w in mr.BAD_STATE + mr.FULL))
        place_all(c, tr)
        place_all(once, tr)
        self.assertFalse(c.bad.any() or c.a_prev.any())
        for d in c.data:
            self.assertFalse(d.qacc_warmstart.any() or d.ctrl.any() or d.time)
            self.assertFalse(any(d.warning[w].number for w in mr.BAD_STATE + mr.FULL))
        a = 0.2 * torch.randn(3, J, generator=g)
        for env in (c, once):
            env.step(a, tr["command_row"])
        for d, e in zip(c.data, once.data):
            for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "act", "time"):
                np.testing.assert_array_equal(getattr(d, k), getattr(e, k), err_msg=k)
        c.close(), once.close()

    def test_previous_action_and_context_are_owned(self):
        tr = read(SCENE, 4)
        offsets = 0.01 * torch.arange(J, dtype=torch.float32)
        context = xi(3, offsets)
        c = make(3, context=context)
        context.zero_()  # the object holds its own copy
        place_all(c, tr)
        a_prev = slice(2 * J + 6, 3 * J + 6)  # mdp.obs: q - q0 - offset, qd, w, gravity, previous action, orientation
        self.assertFalse(c.observe(tr["command_row"])[:, a_prev].any())
        a = 0.1 * torch.randn(3, J)
        S, _ = c.step(a, tr["command_row"])
        o = c.observe(tr["command_row"])
        torch.testing.assert_close(o[:, a_prev], a, rtol=0, atol=0)
        torch.testing.assert_close(o[:, :J], S[4] - DEFAULT_Q - offsets, rtol=0, atol=0)
        # ctrl, read through each actuator's joint: the default pose plus the joint offsets plus the scaled action
        target = (DEFAULT_Q + offsets + ALPHA * a).double().numpy()
        for i, d in enumerate(c.data):
            np.testing.assert_array_equal(ctrl_by_name(MODEL, d), target[i])
        place_all(c, tr)
        self.assertFalse(c.a_prev.any())
        c.close()

    def test_one_state_chain(self):
        """After a few controls the object's data equal, bitwise, an mj_step loop on separate MjData started from the
        placed data with the same ctrl: nothing is rounded or written back between controls."""
        tr = read(SCENE, 4)
        c = make(3)
        place_all(c, tr)
        ref = [copy.copy(d) for d in c.data]
        ids = [id(d) for d in c.data]
        g = torch.Generator().manual_seed(0)
        for _ in range(3):
            a = 0.2 * torch.randn(3, J, generator=g)
            c.step(a, tr["command_row"])
            target = (DEFAULT_Q + ALPHA * a).double().numpy()
            for i, d in enumerate(ref):
                d.ctrl[c.maps["actuator"]] = target[i]
                for _ in range(DECIMATION):
                    mujoco.mj_step(MODEL, d)
        self.assertEqual([id(d) for d in c.data], ids)
        for d, e in zip(c.data, ref):
            for k in ("qpos", "qvel", "qacc_warmstart", "act", "time"):
                np.testing.assert_array_equal(getattr(d, k), getattr(e, k), err_msg=k)
        self.assertTrue(any(d.qacc_warmstart.any() for d in c.data))
        c.close()

    def test_report(self):
        tr = read([trial(1), trial(LYING + 1)], 2)
        c = make(2)
        place_all(c, tr)
        _, rep = c.step(torch.zeros(2, J), tr["command_row"])
        self.assertEqual(rep["contacts"].tolist(), [d.ncon for d in c.data])
        self.assertEqual(rep["constraint_rows"].tolist(), [d.nefc for d in c.data])
        self.assertGreater(int(rep["contacts"][1]), 0)  # the lying body touches the floor
        self.assertFalse(rep["contacts_beyond_capacity"].any() or rep["rows_beyond_capacity"].any())
        # the largest counts over the control's substeps are at least those after its last substep
        peaks = c.substep_peaks
        self.assertTrue((peaks[:, 0] >= rep["contacts"].numpy()).all() and (peaks[:, 1] >= rep["constraint_rows"].numpy()).all())
        self.assertTrue((peaks[:, 2] <= peaks[:, 1]).all())
        self.assertGreater(int(peaks[1, 0]), 0)
        c.close()

    def test_push_and_lift(self):
        tr = read([trial(1), trial(2)], 2)
        c = make(2)
        place_all(c, tr)
        q0, v0 = c.raw()
        c.push(torch.tensor([1]), torch.tensor([[0.3, -0.1]]))
        q1, v1 = c.raw()
        np.testing.assert_array_equal(q1, q0)
        dv = np.zeros_like(v0)
        dv[1, c.maps["free_v"][:2]] = np.float32([0.3, -0.1])
        np.testing.assert_array_equal(v1, v0 + dv)
        c.lift(2.0)
        q2, _ = c.raw()
        z = c.maps["free_q"][2]
        np.testing.assert_array_equal(q2[:, z], (q0[:, z].astype(np.float32) + np.float32(2.0)).astype(np.float64))
        np.testing.assert_array_equal(np.delete(q2, z, 1), np.delete(q0, z, 1))
        c.close()

    def test_record_fields_are_those_of_the_stand_in(self):
        tr = read(SCENE, 3)
        rec = rollouts.roll_out(make(3), POLICY, 4, MDP_, tr, 3, False)
        stand_in = rollouts.roll_out(StandIn(3), POLICY, 4, MDP_, tr, 3, False)
        self.assertEqual(set(rec), set(stand_in))
        for k in rec:
            self.assertEqual((rec[k].shape, rec[k].dtype), (stand_in[k].shape, stand_in[k].dtype), k)
        self.assertTrue(rec["available"].all())
        self.assertEqual(rec["push_applied"][:, 2].tolist(), [False, True, False])

    def test_threads_do_not_change_the_record(self):
        tr = read(SCENE, 4)
        one = rollouts.roll_out(make(3, threads=1), POLICY, 4, MDP_, tr, 4, False)
        three = rollouts.roll_out(make(3, threads=3), POLICY, 4, MDP_, tr, 4, False)
        self.assertTrue(mr.same_records(one, three)["identical"])
        self.assertFalse(np.array_equal(one["state"][0], one["state"][-1]))  # the bodies moved

    def test_a_bad_state_ends_the_trial_as_nonfinite(self):
        """MuJoCo C resets data whose velocity exceeds 1e10; the record shows the control as not finite."""
        tr = read([trial(1, push={"control": 1, "velocity_xy": [1e12, 0.0]}), trial(2)], 4)
        c = make(2)
        rec = rollouts.roll_out(c, POLICY, 4, MDP_, tr, 4, False)
        self.assertEqual(rec["nonfinite"][:, 0].tolist(), [False, True, False, False])
        self.assertEqual(rec["done"][:, 0].tolist(), [False, True, False, False])
        self.assertEqual(rec["available"][:, 0].tolist(), [True, False, False, False])
        self.assertTrue(rec["available"][:, 1].all())
        self.assertFalse(c.bad.any())  # the ended trial is placed again before every later control
        c.close()

    def test_a_nonfinite_action_gives_a_nonfinite_state(self):
        tr = read([trial(1), trial(2)], 2)
        c = make(2)
        place_all(c, tr)
        a = torch.zeros(2, J)
        a[0, 3] = float("nan")
        S, _ = c.step(a, tr["command_row"])
        x = flatten_state(S)
        self.assertEqual(torch.isfinite(x).all(-1).tolist(), [False, True])
        place_all(c, tr)
        self.assertTrue(torch.isfinite(flatten_state(c.state())).all())
        c.close()

    def test_refusals(self):
        for case, kwargs in {"threads": {"threads": 0}, "context": {"context": torch.zeros(2, MDP_.xi_dim)}}.items():
            with self.subTest(case), self.assertRaises(ValueError):
                make(3, **kwargs)


class TestChecks(unittest.TestCase):
    def run_checks(self, training=None, bodies=None, rows=None):
        tr = read(SCENE, 3)
        rec = rollouts.roll_out(make(3), POLICY, 4, MDP_, tr, 1, False)
        if rows is not None:
            rec["body_row"] = rec["body_row"] + rows
        names = list(ROBOT.tracked_bodies)
        return mr.checks(lambda: make(3), training or make(3), POLICY, 4, MDP_, tr, rec, bodies or names, names, 3, False)

    def test_two_mujoco_c_objects(self):
        out = self.run_checks()
        json.dumps(out, allow_nan=False)
        a = out["a_placement"]
        for k in ("start_body_pos_w_against_record", "start_body_quat_w_against_record"):
            self.assertEqual(a[k]["largest"], 0.0, k)
        for g in mr.STATE_GROUPS:
            self.assertEqual(a["start_state_against_record"][g]["largest"], 0.0, g)
        # MuJoCo C's own kinematics of the tracked bodies against the release's kinematics of the same state
        self.assertLess(a["mujoco_kinematics_body_pos_against_record"]["largest"], 1e-5)
        self.assertLess(a["mujoco_kinematics_body_quat_against_record"]["largest"], 1e-5)
        for name in ("lifted", "placed"):
            b = out["b_one_control"][name]
            for g in mr.STATE_GROUPS:
                self.assertEqual(b["state_after_all_trials"][g]["largest"], 0.0, (name, g))
            self.assertEqual(b["contacts"]["mujoco_c"], b["contacts"]["training_simulator"])
        self.assertEqual(out["b_one_control"]["placed"]["mujoco_c_twice"]["joint_positions"]["largest"], 0.0)
        lifted = out["b_one_control"]["lifted"]
        # no floor contact 2 m up and no self contact, in any substep
        self.assertTrue(all(n == 0 for n in lifted["contacts"]["mujoco_c_largest_over_substeps"]))
        self.assertEqual(lifted["state_after_without_contact"]["trials"], 3)
        rows = lifted["constraint_rows"]
        without_rows = sum(r == 0 and s == 0 for r, s in zip(rows["mujoco_c_largest_over_substeps"], rows["training_simulator"]))
        self.assertEqual(lifted["state_after_without_constraint_rows"]["trials"], without_rows)
        self.assertTrue(out["d_repeat"]["identical"])
        self.assertEqual(out["d_repeat"]["controls"], 3)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "body order"):
            self.run_checks(bodies=list(reversed(ROBOT.tracked_bodies)))
        with self.assertRaisesRegex(ValueError, "body_row"):
            self.run_checks(rows=1)
        # a training side whose ctrl differs after the same action: a joint-to-actuator mismatch
        with self.assertRaisesRegex(ValueError, "joint order"):
            self.run_checks(training=make(3, context=xi(3, torch.full((J,), 0.01))))


class TestReports(unittest.TestCase):
    def test_describe_model(self):
        d = mr.describe_model(MODEL)
        json.dumps(d, allow_nan=False)
        self.assertEqual((d["options"]["integrator"], d["options"]["cone"], d["options"]["solver"]),
                         ("mjINT_IMPLICITFAST", "mjCONE_PYRAMIDAL", "mjSOL_NEWTON"))  # fmt: skip
        self.assertEqual((d["options"]["timestep"], d["options"]["iterations"], d["options"]["ls_iterations"]),
                         (0.005, ROBOT.solver[0], ROBOT.solver[1]))  # fmt: skip
        self.assertEqual((len(d["geoms"]), len(d["actuators"]), len(d["dofs"])), (MODEL.ngeom, MODEL.nu, MODEL.nv))
        self.assertEqual(len(d["excluded_body_pairs"]), MODEL.nexclude)
        self.assertEqual(mr.model_differences(MODEL, copy.deepcopy(MODEL)), {})

    def test_warp_option_values(self):
        import warp as wp

        cone = enum.IntEnum("Cone", "PYRAMIDAL ELLIPTIC", start=0)
        opt = SimpleNamespace(timestep=wp.array(np.float32([0.005]), dtype=float, device="cpu"), cone=cone.ELLIPTIC,
                              iterations=10, gravity=wp.array(np.float32([[0, 0, -9.81]]), dtype=wp.vec3, device="cpu"),
                              impratio_invsqrt=wp.array(np.float32([1.0]), dtype=float, device="cpu"))  # fmt: skip
        out = mr.warp_option_values(opt, ["timestep", "cone", "iterations", "gravity", "noslip_iterations"])
        self.assertEqual(sorted(out), ["cone", "gravity", "impratio_invsqrt", "iterations", "timestep"])
        self.assertEqual((out["timestep"].shape, out["gravity"].shape, out["cone"].tolist(), out["iterations"].tolist()),
                         ((1,), (1, 3), 1, 10))  # fmt: skip
        self.assertEqual(out["timestep"].dtype, np.float32)
        with self.assertRaises(ValueError):  # neither a warp array nor an integer
            mr.warp_option_values(SimpleNamespace(timestep=0.005), ["timestep"])

    def test_model_field_comparison(self):
        arrays = {"geom_friction": np.float32(MODEL.geom_friction)[None].repeat(2, 0),  # per world
                  "body_mass": np.float32(MODEL.body_mass), "jnt_type": MODEL.jnt_type.copy(),  # the MjModel's shape
                  "geom_pos": np.zeros((2, 1, 3), np.float32), "warp_only": np.zeros(3)}  # fmt: skip
        out = mr.model_field_comparison(MODEL, arrays)
        self.assertEqual((out["fields_compared"], out["differing_entries"]), (3, {}))
        self.assertEqual((len(out["other_layout"]), out["not_in_the_mjmodel"]), (1, ["warp_only"]))
        arrays["geom_friction"][1, 2, 0] = 0.25  # one world differs at one entry
        arrays["jnt_type"][0] = 3
        out = mr.model_field_comparison(MODEL, arrays)
        self.assertEqual(out["differing_entries"], {"geom_friction": 1, "jnt_type": 1})
        json.dumps(out, allow_nan=False)

    def test_largest(self):
        a = np.array([[0.0, 1.0], [2.0, np.nan], [1.0, 1.0]])
        b = np.array([[0.5, 1.0], [2.0, 0.0], [1.0, 3.0]])
        self.assertEqual(mr.largest(a, b), {"largest": 2.0, "trial": 2, "median_of_trials": 1.25, "nonfinite_trials": 1})
        q = np.array([[[0.5, 0.5, -0.5, 0.5]]])
        self.assertEqual(mr.largest(q, -q, quaternion=True)["largest"], 0.0)
        flipped = q * np.array([1.0, -1.0, 1.0, 1.0])  # another rotation: one component's sign
        self.assertEqual(mr.largest(q, flipped, quaternion=True)["largest"], 1.0)

    def test_same_records(self):
        r = {"x": np.array([1.0, np.nan], np.float32), "n": np.array([1, 2], np.int32), "f": np.array([True, False])}
        self.assertTrue(mr.same_records(r, copy.deepcopy(r))["identical"])
        s = copy.deepcopy(r)
        s["x"][0], s["f"][1] = 1.5, True
        out = mr.same_records(r, s)
        self.assertEqual(out["differing_fields"], {"x": {"entries": 1, "largest": 0.5}, "f": {"entries": 1, "largest": 1.0}})
        with self.assertRaises(ValueError):
            mr.same_records(r, {"x": r["x"]})


def rollouts_description_keys():
    """The keys of tools/rollouts.py's description.json, its settings and its versions, read from main's source."""
    with open(os.path.join(TOOLS, "rollouts.py")) as f:
        tree = ast.parse(f.read())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node = next(n.value for n in ast.walk(main) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "description")
    keys = lambda d: [k.value for k in d.keys]  # noqa: E731
    settings = node.values[keys(node).index("settings")]
    versions = settings.values[keys(settings).index("versions")]
    return keys(node), keys(settings), [e.value for e in versions.generators[0].iter.elts]


class TestDescription(unittest.TestCase):
    def test_keys_are_those_of_rollouts(self):
        top, settings, versions = rollouts_description_keys()
        args = SimpleNamespace(run_dir="run", scene="s.json", termination="nonfinite", controls=5, threads=2)
        env = SimpleNamespace(eval_mode=True, randomize=False, reset_noise=False)
        ck = {"iteration": 7, "args": {"goal_frames": 4, "hist": 3, "amp": "bf16"}}
        library = {"motion_dir": "d", "bones_list": "", "bones_lib": None, "motion_store": "fp32"}
        model, written = mr.trainer_model(MODEL, {})
        options_written, _ = mr.trainer_options(model, warp_like(MODEL))
        simulator = mr.simulator_section(model, written, options_written, ROBOT, 2)
        self.assertEqual(list(simulator["options_written"]), ["tolerance"])
        self.assertEqual(simulator["options"]["tolerance"], float(np.float32(1e-6)))
        self.assertEqual(simulator["solver"], list(ROBOT.solver))  # the robot's setting, as tools/rollouts.py records it
        d = mr.describe(args, {"robot": "mjlab_g1"}, "run/ckpt_000007.pt", ck, library, {"trials": []}, MOTION, ROBOT,
                        MDP_, env, simulator, "card")  # fmt: skip
        json.dumps(d, allow_nan=False)
        self.assertEqual(list(d), top)
        self.assertEqual(list(d["settings"]), settings)
        self.assertEqual(list(d["settings"]["versions"]), versions + ["mujoco"])
        self.assertEqual(d["simulator"]["engine"], "MuJoCo C")
        self.assertEqual((d["simulator"]["nconmax"], d["simulator"]["njmax"], d["simulator"]["naconmax_pooled"]), (None, None, None))
        self.assertEqual((d["simulator"]["physics_dt"], d["simulator"]["decimation"]), (0.005, DECIMATION))
        self.assertEqual(d["simulator"]["differences"], list(mr.DIFFERENCES))
        self.assertTrue(any("push" in x for x in mr.DIFFERENCES) and any("CONTACTFULL" in x for x in mr.DIFFERENCES)
                        and any("non-finite" in x for x in mr.DIFFERENCES))  # fmt: skip
        # fields a reader of both records compares, written as tools/rollouts.py writes them
        self.assertEqual((d["termination"], d["controls"], d["settings"]["control_dt"], d["run_args"]["robot"]),
                         ("nonfinite", 5, CONTROL_DT, "mjlab_g1"))  # fmt: skip
        self.assertEqual(d["library"], library | {"clips": len(MOTION.names), "rows": MOTION.num_frames})


class TestCommandLine(unittest.TestCase):
    def call(self, *extra):
        cmd = [sys.executable, os.path.join(TOOLS, "mujoco_rollouts.py"), "--run_dir", "run", "--ckpt", "1", "--scene",
               "s.json", "--termination", "nonfinite", *extra]  # fmt: skip
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as out:
            cases = {"exists": ("--controls", "5", "--threads", "2", "--out", out),
                     "--controls": ("--controls", "0", "--threads", "2", "--out", os.path.join(out, "x")),
                     "--threads": ("--controls", "5", "--threads", "0", "--out", os.path.join(out, "x")),
                     "holds no description.json": ("--controls", "5", "--threads", "2", "--checks", out, "--out",
                                                   os.path.join(out, "x"))}  # fmt: skip
            for message, extra in cases.items():
                with self.subTest(message):
                    r = self.call(*extra)
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertIn(message, r.stderr)
            self.assertFalse(os.path.exists(os.path.join(out, "x")))
        r = self.call("--controls", "5", "--out", "never_made")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--threads", r.stderr)

    def test_help(self):
        r = subprocess.run([sys.executable, os.path.join(TOOLS, "mujoco_rollouts.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--checks", r.stdout)


if __name__ == "__main__":
    unittest.main()
