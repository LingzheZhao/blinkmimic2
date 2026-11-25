"""Sensors added to the tracking scene (env.make_scene_cfg, TrackingSim) and the contact sensors' match limit
(env.make_sim). With the defaults the scene compiles to the former model and the simulation settings equal the former
ones; a contact sensor between the feet and the floor compiles to MuJoCo contact sensors on those bodies and leaves the
rest of the model unchanged. MuJoCo's compiler and mjlab's simulation set-up on the CPU, without stepping."""

import os
import sys
import unittest
from unittest import mock

import mujoco
import numpy as np
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ROBOT, library  # noqa: E402

from blinkmimic2 import env as env_module  # noqa: E402
from blinkmimic2.env import PHYSICS_DT, TrackingSim, make_scene_cfg, make_sim  # noqa: E402

KIN, MOTION, MDP_ = library()
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
FLOOR = ContactSensorCfg(name="feet_floor", primary=ContactMatch(mode="subtree", pattern=FEET, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"), fields=("force",), reduce="netforce",
                         history_length=8)  # fmt: skip


def former_scene_cfg(num_envs: int) -> SceneCfg:
    """make_scene_cfg before the sensors argument."""
    return SceneCfg(num_envs=num_envs, env_spacing=0.0, terrain=TerrainEntityCfg(terrain_type="plane"),
                    entities={"robot": ROBOT.robot_cfg()})  # fmt: skip


def former_sim_cfg() -> SimulationCfg:
    """make_sim before the contact_sensor_maxmatch argument."""
    return SimulationCfg(njmax=ROBOT.njmax, nconmax=ROBOT.nconmax,
                         mujoco=MujocoCfg(timestep=PHYSICS_DT, integrator="implicitfast", cone="pyramidal", solver="newton",
                                          iterations=ROBOT.solver[0], ls_iterations=ROBOT.solver[1],
                                          tolerance=ROBOT.solver[2], ccd_iterations=ROBOT.solver[3]))  # fmt: skip


def model_bytes(m: mujoco.MjModel) -> np.ndarray:
    buf = np.zeros(mujoco.mj_sizeModel(m), np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return buf


class Captured(Exception):
    def __init__(self, cfg):
        self.cfg = cfg


def stop_at_scene(cfg, device):
    raise Captured(cfg)


class TestSceneSensors(unittest.TestCase):
    def test_defaults_compile_to_the_former_model(self):
        self.assertEqual(make_scene_cfg(ROBOT, 2).sensors, ())
        got = Scene(make_scene_cfg(ROBOT, 2), device="cpu").compile()
        want = Scene(former_scene_cfg(2), device="cpu").compile()
        np.testing.assert_array_equal(model_bytes(got), model_bytes(want))

    def test_trackingsim_passes_its_sensors(self):
        for sensors in ((), (FLOOR,)):
            with self.subTest(sensors=sensors), mock.patch.object(env_module, "Scene", side_effect=stop_at_scene), \
                    self.assertRaises(Captured) as c:  # fmt: skip
                TrackingSim(None, 2, ROBOT, MOTION, MDP_, "cpu", sensors=sensors)
            self.assertEqual(c.exception.cfg.sensors, sensors)
        with mock.patch.object(env_module, "Scene", side_effect=stop_at_scene), self.assertRaises(Captured) as c:
            TrackingSim(None, 2, ROBOT, MOTION, MDP_, "cpu")
        self.assertEqual(c.exception.cfg.sensors, ())

    def test_floor_contact_sensor_compiles_on_the_feet_and_the_floor(self):
        m0 = Scene(make_scene_cfg(ROBOT, 2), device="cpu").compile()
        m1 = Scene(make_scene_cfg(ROBOT, 2, (FLOOR,)), device="cpu").compile()
        self.assertEqual((m1.nq, m1.nv, m1.nu, m1.nbody, m1.ngeom), (m0.nq, m0.nv, m0.nu, m0.nbody, m0.ngeom))
        for k in ("body_mass", "geom_type", "geom_size", "geom_friction", "geom_condim", "geom_priority"):
            np.testing.assert_array_equal(getattr(m1, k), getattr(m0, k), k)
        self.assertEqual(m1.nsensor, m0.nsensor + len(FEET))  # one sensor per primary and field
        self.assertEqual([m1.sensor(i).name for i in range(m0.nsensor)], [m0.sensor(i).name for i in range(m0.nsensor)])
        floor = m1.body("terrain").id
        self.assertEqual(int(m1.geom_type[m1.body_geomadr[floor]]), int(mujoco.mjtGeom.mjGEOM_PLANE))
        for foot in FEET:
            s = m1.sensor(f"feet_floor_{foot}_force")
            self.assertEqual(int(s.type[0]), int(mujoco.mjtSensor.mjSENS_CONTACT))
            foot_body = m1.body(f"robot/{foot}").id
            self.assertEqual((int(s.objtype[0]), int(s.objid[0])), (int(mujoco.mjtObj.mjOBJ_XBODY), foot_body))
            self.assertEqual((int(s.reftype[0]), int(s.refid[0])), (int(mujoco.mjtObj.mjOBJ_BODY), floor))


class TestMatchLimit(unittest.TestCase):
    def test_default_is_the_former_simulation(self):
        self.assertEqual(make_sim(ROBOT), former_sim_cfg())
        self.assertEqual(make_sim(ROBOT).contact_sensor_maxmatch, SimulationCfg().contact_sensor_maxmatch)

    def test_given_limit_reaches_the_simulation(self):
        cfg = make_sim(ROBOT, contact_sensor_maxmatch=500)
        self.assertEqual(cfg.contact_sensor_maxmatch, 500)
        scene = Scene(make_scene_cfg(ROBOT, 2, (FLOOR,)), device="cpu")
        sim = Simulation(num_envs=2, cfg=cfg, model=scene.compile(), device="cpu")
        scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
        self.assertEqual(int(sim.wp_model.opt.contact_sensor_maxmatch), 500)
        self.assertIsInstance(scene["feet_floor"], ContactSensor)


if __name__ == "__main__":
    unittest.main()
