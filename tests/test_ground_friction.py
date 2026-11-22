"""Contact friction of the floor (DomainRandomizer.startup) on mjlab's simulator of the mjlab G1 on the CPU: the ground
geoms are the floor's plane, which mjlab puts on a body of its own, and they get the lowest friction of the range in
training and 1.0 in evaluation, so a robot geom without contact priority meets the floor with its own drawn friction."""

import os
import sys
import unittest
from types import SimpleNamespace

import mujoco
import torch
from mjlab.scene import Scene
from mjlab.sim import Simulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import ROBOT  # noqa: E402

from blinkmimic2.dr import DomainRandomizer, XiLayout  # noqa: E402
from blinkmimic2.env import make_scene_cfg, make_sim  # noqa: E402

N = 4


def randomized(eval_mode: bool):
    """(scene, sim, randomizer, xi) after startup of N worlds."""
    torch.manual_seed(0)
    scene = Scene(make_scene_cfg(ROBOT, N), device="cpu")
    sim = Simulation(num_envs=N, cfg=make_sim(ROBOT), model=scene.compile(), device="cpu")
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    env = SimpleNamespace(scene=scene, sim=sim, robot=scene["robot"], num_envs=N, device="cpu")
    layout = XiLayout(ROBOT, "cpu")
    xi = torch.zeros(N, layout.dim)
    r = DomainRandomizer(ROBOT, layout, env, eval_mode=eval_mode)
    r.startup(xi)
    return scene, sim, r, xi[:, layout.slices["friction"]].squeeze(-1)


def plane(m: mujoco.MjModel) -> int:
    """The floor: the model's one plane geom."""
    (g,) = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    return g


def floor_contact_friction(scene: Scene, friction: torch.Tensor, geom: str) -> list[float]:
    """Sliding friction of MuJoCo's contacts between ``geom`` and the floor (the plane), with the robot at the model's
    initial pose lowered until the geom's centre is at the floor's height and the geoms' sliding friction set to
    ``friction``."""
    m = scene.compile()
    m.geom_friction[:, 0] = friction.numpy()
    d = mujoco.MjData(m)
    g, floor = m.geom(geom).id, plane(m)
    mujoco.mj_kinematics(m, d)
    d.qpos[2] -= d.geom_xpos[g, 2]
    mujoco.mj_forward(m, d)
    contacts = [d.contact[i] for i in range(d.ncon)]
    return [float(c.friction[0]) for c in contacts if {int(c.geom1), int(c.geom2)} == {g, floor}]


class TestGroundFriction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = randomized(eval_mode=False)
        cls.eval = randomized(eval_mode=True)

    def test_ground_geoms_are_the_floor_plane(self):
        _, sim, r, _ = self.train
        floor = plane(sim.mj_model)
        self.assertEqual(r.ground_geoms.tolist(), [floor])
        self.assertNotIn(floor, r.friction_geoms.tolist())

    def test_training_puts_the_floor_at_the_bottom_of_the_range(self):
        _, sim, r, drawn = self.train
        lo, hi = ROBOT.dr.friction_range
        fr = sim.model.geom_friction[..., 0]
        self.assertTrue(torch.all(fr[:, plane(sim.mj_model)] == lo))
        self.assertTrue(torch.all((drawn >= lo) & (drawn <= hi)))
        torch.testing.assert_close(fr[:, r.friction_geoms], drawn[:, None].expand(N, len(r.friction_geoms)))

    def test_evaluation_puts_floor_and_robot_at_one(self):
        _, sim, r, drawn = self.eval
        fr = sim.model.geom_friction[..., 0]
        self.assertTrue(torch.all(fr[:, plane(sim.mj_model)] == 1.0))
        self.assertTrue(torch.all(fr[:, r.friction_geoms] == 1.0))
        self.assertTrue(torch.all(drawn == 1.0))

    def test_a_hand_and_a_foot_meet_the_floor_with_the_drawn_friction(self):
        scene, sim, _, drawn = self.train
        self.assertTrue(torch.any(drawn < 1.0))  # worlds where the floor's model friction (1.0) would win the maximum
        for w in range(N):
            for geom in ("robot/left_hand_collision", "robot/left_foot1_collision"):  # priority 0 and 1
                got = floor_contact_friction(scene, sim.model.geom_friction[w, :, 0], geom)
                self.assertGreater(len(got), 0, geom)
                for mu in got:
                    self.assertAlmostEqual(mu, drawn[w].item(), places=6, msg=(w, geom))


if __name__ == "__main__":
    unittest.main()
