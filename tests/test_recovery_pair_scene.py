"""The scene program of the recovery pair (studies/recovery_pair/scene.py) on generated libraries of the mjlab G1 on the
CPU.

``library()`` (clips in this order, 50 Hz rows): getup (200 rows: rows 0-149 lying at pelvis height 0.2 m, rows 150-199
upright at 0.78 m), walk (1,300 upright rows at 0.78 m) and sway (100 upright rows at 0.78 m). getup is the hidden
clip and getup and sway are command-excluded, so the start rows are rows 0-149, and the rows of walk (rows 200-1499)
with 1,000 upright rows ahead before its last row are rows 200-498 (with 500 ahead, rows 200-998)."""

import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(REPO, "studies", "recovery_pair"), os.path.join(REPO, "tools")]
import rollouts  # noqa: E402
import scene as SCENE  # noqa: E402
from declaration import read_declaration  # noqa: E402

from blinkmimic2 import recovery  # noqa: E402
from blinkmimic2.kinematics import Kinematics  # noqa: E402
from blinkmimic2.motion import MotionLib  # noqa: E402
from blinkmimic2.robot import get_robot  # noqa: E402

ROBOT = get_robot("mjlab_g1")
KIN = Kinematics(ROBOT, "cpu")
PAIR = os.path.join(REPO, "studies", "recovery_pair", "pair.json")
PLACED = read_declaration(PAIR)["placed_starts"]
CLIPS = (("getup", 200), ("walk", 1300), ("sway", 100))
# The draws at seed 7, written out: the uniforms and the permutation depend only on the seed, the order of the draws
# and their shapes (29 joints), not on the library.
POSITION_0 = [0.06984508037567139, -0.602393627166748, 0.31842339038848877]
ROTATION_0 = [0.8734664916992188, 0.10339879989624023, -0.34344375133514404]
VELOCITY_50 = [0.9219750165939331, 0.7520275115966797, -0.1744670867919922, -0.35515522956848145,
               -0.5345755815505981, 0.8204742670059204]
JOINTS_99_LAST_TWO = [-0.8979747295379639, -0.5212045907974243]
PERMUTATION = [5, 27, 83, 17, 66, 80, 61, 49, 50, 6, 39, 20, 30, 41, 99, 33, 21, 0, 90, 40, 11, 87, 15, 93, 62, 14, 16,
               43, 75, 31, 72, 8, 42, 53, 4, 22, 3, 51, 84, 1, 18, 97, 45, 44, 60, 13, 58, 19, 7, 64, 23, 81, 67, 71,
               26, 70, 12, 9, 59, 69, 78, 86, 10, 98, 54, 88, 77, 52, 48, 68, 95, 63, 89, 92, 25, 29, 32, 73, 91, 46,
               57, 38, 55, 36, 24, 37, 96, 2, 85, 65, 34, 47, 56, 94, 74, 76, 35, 79, 28, 82]


def library():
    rows = sum(n for _, n in CLIPS)
    z = torch.full((rows,), 0.78)
    z[:150] = 0.2
    p = torch.stack((torch.zeros(rows), torch.zeros(rows), z), -1)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(rows, 1)
    j = torch.tensor(ROBOT.default_joint_pos_list()).repeat(rows, 1)
    cut = np.cumsum([0] + [n for _, n in CLIPS]).tolist()
    parts = lambda x: [x[a:b] for a, b in zip(cut[:-1], cut[1:])]  # noqa: E731
    return MotionLib.from_clips([name for name, _ in CLIPS], parts(p), parts(q), parts(j), KIN, "cpu")


def settings():
    return recovery.Settings(share=0.5, hidden_clips=("getup",), command_excluded_clips=("getup", "sway"),
                             low_pelvis_height=0.5, low_pelvis_up_z=0.5, upright_run_rows=50, longest_span_rows=375,
                             command_upright_rows=500, reward_scale=2.5, stop_controls=100, blend_controls=75)


class SceneTest(unittest.TestCase):
    def test_the_files_hold_the_declared_trials_twins_and_every_clip_s_first_row(self):
        motion = library()
        files, record = SCENE.scene(motion, settings(), PLACED, ROBOT.num_joints)
        self.assertEqual(sorted(files), ["first_rows.json", "placed.json", "twins.json"])
        placed, twins = files["placed.json"]["trials"], files["twins.json"]["trials"]
        self.assertEqual((len(placed), len(twins)), (100, 100))
        self.assertEqual([t["command_row"] for t in twins], [t["command_row"] for t in placed])
        self.assertTrue(all(set(t) == {"command_row"} for t in twins))
        self.assertEqual(files["first_rows.json"]["trials"], [{"command_row": 0}, {"command_row": 200},
                                                              {"command_row": 1500}])
        self.assertEqual(record["first_row_controls"], 1299)
        self.assertEqual([clip["controls"] for clip in record["clips"]], [199, 1299, 99])
        self.assertEqual(record["rows"], dict(low_rows_of_hidden_clips=150, without_a_get_up_end_in_the_clip=0,
                                              beyond_the_longest_span=0, start_rows=150, get_ups=1, command_rows=799))
        self.assertEqual(record["command_rows_with_upright_rows_ahead"], 299)
        self.assertEqual(record["settings"]["hidden_clips"], ("getup",))
        for t in placed:
            self.assertEqual({k: len(v) for k, v in t["reset_uniforms"].items()},
                             dict(position=3, rotation=3, velocity=6, joints=29))
        # the scene format of tools/rollouts.py
        for name, content in files.items():
            trials = rollouts.read_scene(json.loads(json.dumps(content, allow_nan=False)), motion.num_frames, 29, 1000)
            self.assertEqual(trials["command_row"].tolist(), [t["command_row"] for t in content["trials"]], name)
        self.assertEqual(rollouts.read_scene(files["placed.json"], motion.num_frames, 29, 1000)["body_row"].tolist(),
                         [t["body_row"] for t in placed])
        json.dumps(record, allow_nan=False)

    def test_body_and_command_rows_are_evenly_spaced_and_paired_by_the_drawn_permutation(self):
        files, record = SCENE.scene(library(), settings(), PLACED, ROBOT.num_joints)
        body = [t["body_row"] for t in files["placed.json"]["trials"]]
        self.assertEqual(body, np.round(np.linspace(0, 149, 100)).astype(int).tolist())
        self.assertTrue(set(np.diff(body)) <= {1, 2})
        commands = [t["command_row"] for t in files["placed.json"]["trials"]]
        self.assertEqual(commands, np.round(np.linspace(200, 498, 100)).astype(int)[PERMUTATION].tolist())
        self.assertEqual([t["permutation"] for t in record["trials"]], PERMUTATION)
        # every command row has 1,000 upright rows of walk ahead, before walk's last row (1499)
        self.assertTrue(all(200 <= row and row + 1000 < 1499 for row in commands))
        trial = record["trials"][0]
        self.assertEqual((trial["body_clip"], trial["body_row_in_clip"]), ("getup", 0))
        self.assertEqual((trial["command_clip"], trial["command_row_in_clip"]), ("walk", trial["command_row"] - 200))
        uniforms, _ = SCENE.draws(7, 100, 29)
        self.assertEqual([t["reset_uniforms"]["joints"] for t in files["placed.json"]["trials"]],
                         uniforms["joints"].tolist())

    def test_the_draws_at_seed_7_are_the_written_ones_and_the_scene_is_determined_by_its_seed(self):
        uniforms, permutation = SCENE.draws(7, 100, 29)
        self.assertEqual(uniforms["position"][0].tolist(), POSITION_0)
        self.assertEqual(uniforms["rotation"][0].tolist(), ROTATION_0)
        self.assertEqual(uniforms["velocity"][50].tolist(), VELOCITY_50)
        self.assertEqual(uniforms["joints"][99, -2:].tolist(), JOINTS_99_LAST_TWO)
        self.assertEqual(permutation.tolist(), PERMUTATION)
        self.assertTrue(all(bool(((v >= -1) & (v < 1)).all()) for v in uniforms.values()))
        motion = library()
        first, second = (SCENE.scene(motion, settings(), PLACED, 29) for _ in range(2))
        self.assertEqual(first, second)
        other, _ = SCENE.scene(motion, settings(), {**PLACED, "uniforms_seed": 8}, 29)
        self.assertNotEqual(other["placed.json"], first[0]["placed.json"])
        self.assertEqual([t["body_row"] for t in other["placed.json"]["trials"]],
                         [t["body_row"] for t in first[0]["placed.json"]["trials"]])


# The clips of the recovery preset and their 30 fps CSV rows: a get-up recording lies for 45 rows and stands for 45.
HIDDEN = ["fallAndGetUp1_subject1", "fallAndGetUp1_subject5", "fallAndGetUp2_subject2", "fallAndGetUp2_subject3",
          "fallAndGetUp3_subject1"]
RECOVERY_ARGS = dict(recovery_share=0.5, recovery_hidden_clips=HIDDEN,
                     recovery_command_excluded_clips=HIDDEN + ["fallAndGetUp1_subject4"],
                     recovery_low_pelvis=[0.5, 0.5], recovery_rows=[50, 375, 500], recovery_reward_scale=2.5,
                     recovery_seconds=[2.0, 1.5])


def write_clip(path, rows, lying=0):
    """A LAFAN1 G1 retarget CSV (30 fps: root position, root quaternion xyzw, 29 joint angles): upright at 0.78 m, its
    first ``lying`` rows at 0.2 m pitched 90 degrees."""
    root = np.tile([0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0], (rows, 1))
    root[:lying] = [0.0, 0.0, 0.2, 0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]
    joints = np.tile(ROBOT.default_joint_pos_list(), (rows, 1))
    np.savetxt(path, np.concatenate((root, joints), 1), delimiter=",")


class MainTest(unittest.TestCase):
    def write_run(self, d):
        """A recovery run directory (args.json and recovery.json) on a folder of 40 clips; the library it gives."""
        motion_dir = os.path.join(d, "clips")
        os.makedirs(motion_dir)
        for name in RECOVERY_ARGS["recovery_command_excluded_clips"]:
            write_clip(os.path.join(motion_dir, name + ".csv"), 90, lying=45)
        write_clip(os.path.join(motion_dir, "walk1_subject1.csv"), 800)
        for i in range(33):
            write_clip(os.path.join(motion_dir, f"short{i:02d}.csv"), 16)
        run = os.path.join(d, "recovery_s0")
        os.makedirs(run)
        args = dict(robot="mjlab_g1", motion_dir=motion_dir, bones_list="", bones_lib="lib_all.npz",
                    motion_store="fp32", seed=0, **RECOVERY_ARGS)
        json.dump(args, open(os.path.join(run, "args.json"), "w"))
        motion = MotionLib.load(ROBOT, KIN, "cpu", motion_dir, "", "lib_all.npz", "fp32")
        counts = recovery.build(motion, recovery.settings_from_args(argparse.Namespace(**args))).counts
        json.dump({"counts": counts}, open(os.path.join(run, "recovery.json"), "w"))
        return run, motion

    def test_the_program_writes_the_scene_of_the_run_s_library_and_refuses_other_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            run, motion = self.write_run(d)
            out = os.path.join(d, "scene")
            with contextlib.redirect_stdout(io.StringIO()):
                SCENE.main(["--pair", PAIR, "--run_dir", run, "--out", out])
            written = {name: json.load(open(os.path.join(out, name))) for name in sorted(os.listdir(out))}
            self.assertEqual(sorted(written), ["first_rows.json", "placed.json", "scene.json", "twins.json"])
            args = json.load(open(os.path.join(run, "args.json")))
            files, record = SCENE.scene(motion, recovery.settings_from_args(argparse.Namespace(**args)), PLACED, 29)
            for name, content in files.items():
                self.assertEqual(written[name], content, name)
            self.assertEqual(written["scene.json"]["library"],
                             dict(motion_dir=os.path.join(d, "clips"), bones_list="", bones_lib="lib_all.npz",
                                  motion_store="fp32", clips=40, rows=motion.num_frames))
            self.assertEqual(written["scene.json"]["trials"], record["trials"])
            self.assertEqual(written["scene.json"]["settings"]["hidden_clips"], HIDDEN)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                SCENE.main(["--pair", PAIR, "--run_dir", run, "--out", out])
            # the row counts of another library, and recovery options that are not the preset's, are refused
            json.dump({"counts": {**record["rows"], "start_rows": 1}}, open(os.path.join(run, "recovery.json"), "w"))
            with self.assertRaisesRegex(ValueError, "row counts"):
                SCENE.main(["--pair", PAIR, "--run_dir", run, "--out", os.path.join(d, "other")])
            json.dump({**args, "recovery_reward_scale": 2.0}, open(os.path.join(run, "args.json"), "w"))
            with self.assertRaisesRegex(ValueError, "recovery options"):
                SCENE.main(["--pair", PAIR, "--run_dir", run, "--out", os.path.join(d, "other")])
            self.assertFalse(os.path.exists(os.path.join(d, "other")))


if __name__ == "__main__":
    unittest.main()
