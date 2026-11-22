"""Training on a folder of motion files: train.py's --motion_dir and the library tools/evaluate.py reads back.

train.py builds the simulator and trains once its options are parsed, so only its refusal runs here; that it passes
--motion_dir to MotionLib.load needs a GPU. The library is built from the values such a run writes to args.json."""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from blinkmimic2.kinematics import Kinematics
from blinkmimic2.motion import CONTROL_DT, MotionLib
from blinkmimic2.robot import get_robot

ROBOT = get_robot("mjlab_g1")


def write_clip(path: str, rows: int):
    """A LAFAN1 G1 retarget CSV (30 fps: root position, root quaternion xyzw, 29 joint angles)."""
    t = np.arange(rows)[:, None] / 30.0
    root = np.concatenate((0.5 * t, 0 * t, 0 * t + 0.76, 0 * t, 0 * t, 0 * t, 1 + 0 * t), 1)
    joints = np.array(ROBOT.default_joint_pos_list())[None] + 0.1 * np.sin(t)
    np.savetxt(path, np.concatenate((root, joints), 1), delimiter=",")


class TestMotionDir(unittest.TestCase):
    def test_refused_with_a_bones_list(self):
        cmd = [sys.executable, "-m", "blinkmimic2.train", "--motion_dir", "clips", "--bones_list", "keep.txt"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--motion_dir is read only with an empty --bones_list", r.stderr)

    def test_the_library_evaluate_reads_from_a_run_with_a_motion_dir(self):
        kin = Kinematics(ROBOT, "cpu")
        with tempfile.TemporaryDirectory() as d:
            write_clip(os.path.join(d, "b_run.csv"), 31)
            write_clip(os.path.join(d, "a_walk.csv"), 16)
            for store in ("full", "fp32"):
                # the keys train.py writes to args.json with --bones_list= --motion_dir D, passed as tools/evaluate.py
                # passes them without its own --bones_list and --bones_lib
                run = {"motion_dir": d, "bones_list": "", "bones_lib": "lib_all.npz", "motion_store": store}
                motion = MotionLib.load(ROBOT, kin, "cpu", run["motion_dir"], run["bones_list"], run["bones_lib"], run["motion_store"],
                                         control_dt=CONTROL_DT)  # fmt: skip
                self.assertEqual(motion.names, ["a_walk", "b_run"])
                # 50 Hz rows over (rows - 1) / 30 s: 0.5 s -> 26 rows, 1.0 s -> 51 rows
                self.assertEqual(motion.clip_len.tolist(), [26, 51])

    def test_a_folder_without_motion_files_is_refused(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(FileNotFoundError, "no \\*.csv files"):
            MotionLib.load(ROBOT, Kinematics(ROBOT, "cpu"), "cpu", d, "", control_dt=CONTROL_DT)


if __name__ == "__main__":
    unittest.main()
