"""The scene entities and sensors added beside the robot (env.make_scene_cfg, TrackingSim) and the function called
after every physics substep (TrackingSim.step_core). With the defaults, the scene configuration and the compiled model
equal the release's and the step makes the release's calls in the release's order; an added mocap box and contact
sensor leave the robot's part of the compiled model unchanged. MuJoCo's compiler on the CPU, without the simulator."""

import dataclasses
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ROBOT, library  # noqa: E402

from blinkmimic2 import env as env_module  # noqa: E402
from blinkmimic2.env import DECIMATION, PHYSICS_DT, TrackingSim, make_scene_cfg  # noqa: E402

KIN, MOTION, MDP_ = library()


def release_scene_cfg(num_envs: int) -> SceneCfg:
    """The release's make_scene_cfg."""
    return SceneCfg(num_envs=num_envs, env_spacing=0.0, terrain=TerrainEntityCfg(terrain_type="plane"),
                    entities={"robot": ROBOT.robot_cfg()})  # fmt: skip


def box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="body", mocap=True)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.4, 0.25])
    return spec


def box() -> dict:
    return {"box": EntityCfg(spec_fn=box_spec)}


CONTACT = ContactSensorCfg(name="box_contact", primary=ContactMatch(mode="body", pattern=("pelvis", "torso_link"), entity="robot"),
                           secondary=ContactMatch(mode="body", pattern="body", entity="box"), fields=("found", "force"),
                           reduce="netforce")  # fmt: skip


def compiled(cfg: SceneCfg) -> mujoco.MjModel:
    return Scene(cfg, device="cpu").compile()


def model_bytes(m: mujoco.MjModel) -> np.ndarray:
    buf = np.zeros(mujoco.mj_sizeModel(m), np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return buf


class Captured(Exception):
    def __init__(self, cfg):
        self.cfg = cfg


def stop_at_scene(cfg, device):
    raise Captured(cfg)


class TestSceneCfg(unittest.TestCase):
    def assert_release_cfg(self, got: SceneCfg, num_envs: int):
        want = release_scene_cfg(num_envs)
        for f in dataclasses.fields(SceneCfg):
            g, w = getattr(got, f.name), getattr(want, f.name)
            if f.name == "terrain":  # every TerrainEntityCfg gets its own spec_fn lambda, which TerrainEntity never calls
                g, w = dataclasses.replace(g, spec_fn=None), dataclasses.replace(w, spec_fn=None)
            self.assertEqual(g, w, f.name)

    def test_defaults_give_the_release_cfg(self):
        self.assert_release_cfg(make_scene_cfg(ROBOT, 3), 3)
        self.assert_release_cfg(make_scene_cfg(ROBOT, 3, None, ()), 3)

    def test_trackingsim_passes_its_arguments(self):
        with mock.patch.object(env_module, "Scene", side_effect=stop_at_scene), self.assertRaises(Captured) as c:
            TrackingSim(None, 2, ROBOT, MOTION, MDP_, "cpu")
        self.assert_release_cfg(c.exception.cfg, 2)
        with mock.patch.object(env_module, "Scene", side_effect=stop_at_scene), self.assertRaises(Captured) as c:
            TrackingSim(None, 2, ROBOT, MOTION, MDP_, "cpu", scene_entities=box(), scene_sensors=(CONTACT,))
        self.assertEqual(list(c.exception.cfg.entities), ["robot", "box"])
        self.assertEqual(c.exception.cfg.sensors, (CONTACT,))

    def test_defaults_compile_to_the_release_model(self):
        np.testing.assert_array_equal(model_bytes(compiled(make_scene_cfg(ROBOT, 2))), model_bytes(compiled(release_scene_cfg(2))))

    def test_added_box_and_sensor_leave_the_robot_unchanged(self):
        m0 = compiled(make_scene_cfg(ROBOT, 2))
        m1 = compiled(make_scene_cfg(ROBOT, 2, box(), (CONTACT,)))
        self.assertEqual((m1.nq, m1.nv, m1.nu, m1.njnt, m1.nkey), (m0.nq, m0.nv, m0.nu, m0.njnt, m0.nkey))
        self.assertEqual((m1.nbody, m1.ngeom, m1.nmocap), (m0.nbody + 1, m0.ngeom + 1, m0.nmocap + 1))
        for k in ("jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_range", "qpos0", "key_qpos", "actuator_trnid",
                  "actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_armature"):  # fmt: skip
            np.testing.assert_array_equal(getattr(m1, k), getattr(m0, k), k)
        for k, n in (("body", m0.nbody), ("geom", m0.ngeom), ("sensor", m0.nsensor)):
            self.assertEqual([getattr(m1, k)(i).name for i in range(n)], [getattr(m0, k)(i).name for i in range(n)], k)
        for k in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_mocapid"):
            np.testing.assert_array_equal(getattr(m1, k)[: m0.nbody], getattr(m0, k), k)
        for k in ("geom_type", "geom_size", "geom_bodyid", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                  "geom_condim", "geom_priority", "geom_friction", "geom_solref", "geom_solimp", "geom_margin"):  # fmt: skip
            np.testing.assert_array_equal(getattr(m1, k)[: m0.ngeom], getattr(m0, k), k)
        np.testing.assert_array_equal(m1.sensor_adr[: m0.nsensor], m0.sensor_adr)
        body = m1.body("box/body").id
        self.assertGreaterEqual(int(m1.body_mocapid[body]), 0)
        g = m0.ngeom
        self.assertEqual((int(m1.geom_bodyid[g]), int(m1.geom_type[g])), (body, int(mujoco.mjtGeom.mjGEOM_BOX)))
        self.assertEqual(m1.nsensor, m0.nsensor + 4)  # 2 primaries x 2 fields
        for prim in ("pelvis", "torso_link"):
            for field in ("found", "force"):
                s = m1.sensor(f"box_contact_{prim}_{field}")
                self.assertEqual(int(s.type[0]), int(mujoco.mjtSensor.mjSENS_CONTACT))
                self.assertEqual((int(s.objid[0]), int(s.refid[0])), (m1.body(f"robot/{prim}").id, body))

    def test_refusals(self):
        for name in ("robot", "terrain"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                make_scene_cfg(ROBOT, 2, {name: EntityCfg(spec_fn=box_spec)})
        mask = torch.zeros(2, dtype=torch.bool)
        for cfg in (make_scene_cfg(ROBOT, 2, box()), make_scene_cfg(ROBOT, 2, None, (CONTACT,))):
            with self.assertRaises(AssertionError):
                TrackingSim.reset_masked(SimpleNamespace(scene_cfg=cfg), mask)
        # the default scene passes the new check and stops at the next attribute the stand-in lacks
        with self.assertRaisesRegex(AttributeError, "eval_mode"):
            TrackingSim.reset_masked(SimpleNamespace(scene_cfg=make_scene_cfg(ROBOT, 2)), mask)
        scene = SimpleNamespace(compile=lambda: None, initialize=lambda **kw: None, sensor_context=object())
        simulation = SimpleNamespace(mj_model=None, model=None, data=None)
        with mock.patch.object(env_module, "Scene", return_value=scene), mock.patch.object(env_module, "Simulation", return_value=simulation):
            with self.assertRaisesRegex(ValueError, "sense pipeline"):
                TrackingSim(None, 2, ROBOT, MOTION, MDP_, "cpu")


class StandIn:
    """The attributes TrackingSim.step_core reads and writes, without the simulator: each physics step moves the roots
    by 1 cm along x, and every call to the scene, the simulator and the robot is logged."""

    def __init__(self, after_substep=None, stateful=False):
        rows = torch.tensor([0, 7, 12])
        N, J = len(rows), ROBOT.num_joints
        self.log, self.steps, self.S0 = [], 0, tuple(x.clone() for x in MOTION.ref_state(rows))
        self.N, self.J, self.f, self.ep_step = N, J, rows.clone(), torch.zeros(N, dtype=torch.long)
        self.xi, self.a_prev = torch.zeros(N, MDP_.xi_dim), torch.zeros(N, J)
        self.default_q, self.alpha = torch.tensor(ROBOT.default_joint_pos_list()), torch.tensor(ROBOT.action_scale_list())
        self.motion, self.mdp, self.gamma, self.max_episode_steps = MOTION, MDP_, 0.99, 500
        self.pipe, self._obs, self._obs_state, self.push_enabled = SimpleNamespace(stateful=False), None, None, False
        if stateful:  # a sensing filter whose update is logged, with an observation already made
            self.pipe = SimpleNamespace(stateful=True, filter_update=self._filter, pack=lambda s: s)
            self._obs, self._obs_state, self.randomize = torch.zeros(N, 1), torch.zeros(N, 1), False
            self.spec = SimpleNamespace(sensing=SimpleNamespace(imu=None))
        self.ep_disc_ret, self.ep_ret = torch.zeros(N), torch.zeros(N)
        self.after_substep = after_substep
        self.scene = SimpleNamespace(write_data_to_sim=lambda: self.log.append(("write",)),
                                     update=lambda dt: self.log.append(("update", dt)))  # fmt: skip
        self.sim = SimpleNamespace(step=self._step)
        self.robot = SimpleNamespace(set_joint_position_target=lambda t: self.log.append(("target", t.clone())),
                                     data=SimpleNamespace(joint_vel=torch.zeros(N, J)))  # fmt: skip

    def _filter(self, state, joint_vel, gyro, accel, xi, dt, noise):
        self.log.append(("filter",))
        return state

    def _step(self):
        self.log.append(("step",))
        self.steps += 1

    def state(self):
        p, R, v, w, q, qd = self.S0
        return p + torch.tensor([0.01 * self.steps, 0.0, 0.0]), R.clone(), v.clone(), w.clone(), q.clone(), qd.clone()


def action():
    return 0.1 * torch.arange(ROBOT.num_joints, dtype=torch.float32).expand(3, -1).contiguous()


def assert_same(test, x, y):
    if isinstance(x, dict):
        test.assertEqual(set(x), set(y))
        for k in x:
            assert_same(test, x[k], y[k])
    else:
        test.assertTrue(torch.equal(x, y))


class TestAfterSubstep(unittest.TestCase):
    def test_without_a_function_the_release_calls(self):
        env, a = StandIn(), action()
        TrackingSim.step_core(env, a)
        self.assertEqual([c[0] for c in env.log], ["target"] + ["write", "step", "update"] * DECIMATION)
        self.assertTrue(torch.equal(env.log[0][1], env.default_q + env.alpha * a))
        self.assertTrue(all(c[1] == PHYSICS_DT for c in env.log if c[0] == "update"))

    def test_function_runs_after_each_update_and_changes_nothing_else(self):
        seen = []

        def after(sim):
            seen.append((sim, sim.steps))
            sim.log.append(("after",))

        env1, env0, a = StandIn(after), StandIn(), action()
        out1, out0 = TrackingSim.step_core(env1, a), TrackingSim.step_core(env0, a)
        self.assertEqual([c[0] for c in env1.log], ["target"] + ["write", "step", "update", "after"] * DECIMATION)
        self.assertEqual(seen, [(env1, i + 1) for i in range(DECIMATION)])
        assert_same(self, out1, out0)
        for k in ("f", "ep_step", "a_prev", "ep_disc_ret", "ep_ret"):
            self.assertTrue(torch.equal(getattr(env1, k), getattr(env0, k)), k)

    def test_function_runs_before_the_sensing_filter_update(self):
        env = StandIn(lambda sim: sim.log.append(("after",)), stateful=True)
        TrackingSim.step_core(env, action())
        self.assertEqual([c[0] for c in env.log], ["target"] + ["write", "step", "update", "after", "filter"] * DECIMATION)


if __name__ == "__main__":
    unittest.main()
