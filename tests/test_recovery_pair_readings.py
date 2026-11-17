"""The reading entry of the recovery pair (studies/recovery_pair/readings.py) on hand-written tools/rollouts.py records,
with a stand-in for the NablaMimic study's readings_core.py written into a temporary directory: the checks of the
records against the scene and the declaration, and the arguments the entry hands to the core."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "studies", "recovery_pair"))
import readings as ENTRY  # noqa: E402

PAIR = os.path.join(REPO, "studies", "recovery_pair", "pair.json")
DECLARATION = json.load(open(PAIR))
# The stand-in core: the field list of readings_core.py and a save_readings that keeps its arguments.
CORE = '''
FIELDS = dict(body_pos_w=2, body_quat_w=2, command_body_pos_w=2, command_body_quat_w=2, command_row_before=0,
              command_row_after=0, clip_last_row=0, available=0, done=0, nonfinite=0, contacts=0, constraint_rows=0,
              contacts_beyond_capacity=0, rows_beyond_capacity=0)
CALLS = []


def save_readings(placed, twins, first_rows, settings, dt, pelvis, anchor, clip_names, clip_first_rows, hidden_clips):
    CALLS.append(dict(placed=placed, twins=twins, first_rows=first_rows, settings=settings, dt=dt, pelvis=pelvis,
                      anchor=anchor, clip_names=clip_names, clip_first_rows=clip_first_rows, hidden_clips=hidden_clips))
    trials = range(placed["available"].shape[1])
    quantities = dict(R=dict(numerator=2), J=dict(numerator=1), J_over_R=dict(value=0.5), F=dict(numerator=0))
    return dict(quantities=quantities,
                readings={t: dict(status="rejoined" if t else "no_fall") for t in trials},
                twin_readings={t: dict(status="no_fall") for t in trials}, engine=dict(placed={}),
                first_rows=dict(C=dict(completed=len(clip_names)), E_m=dict(value=0.03),
                                C_getup={name: True for name in hidden_clips}))
'''
BODIES = ["pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link",
          "right_knee_link", "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link",
          "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]
LIBRARY = dict(motion_dir="/data/clips", bones_list="", bones_lib="lib_all.npz", motion_store="fp32", clips=2,
               rows=1300)
CLIPS = [dict(clip="walk", first_row=0, last_row=1199, controls=1199),
         dict(clip="getup", first_row=1200, last_row=1299, controls=99)]
COMMANDS, BODY_ROWS = [400, 10, 700], [1210, 1220, 1230]


def record(trials, controls, seed):
    """A rollouts.npz of ``trials`` [(command_row, body_row)]: random values in the record's fields and shapes."""
    rng = np.random.default_rng(seed)
    n, b = len(trials), len(BODIES)
    out = {name: rng.normal(size=(controls, n, b, 4 if name.endswith("quat_w") else 3)).astype(np.float32)
           for name in ("body_pos_w", "body_quat_w", "command_body_pos_w", "command_body_quat_w")}
    out |= {name: rng.integers(0, 1300, (controls, n)).astype(np.int32)
            for name in ("command_row_before", "command_row_after", "clip_last_row", "contacts", "constraint_rows")}
    out |= {name: rng.random((controls, n)) < 0.5 for name in ("available", "done", "nonfinite", "termination_height",
                                                               "termination_tilt", "contacts_beyond_capacity",
                                                               "rows_beyond_capacity", "push_applied")}
    out |= dict(action=rng.normal(size=(controls, n, 29)).astype(np.float32),
                command_row=np.array([c for c, _ in trials], np.int32),
                body_row=np.array([r for _, r in trials], np.int32))
    return out


class EntryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        d = self.directory.name
        with open(os.path.join(d, "readings_core.py"), "w") as f:
            f.write(CORE)
        self.core = ENTRY.load_core(d)
        self.scene = os.path.join(d, "scene")
        os.makedirs(self.scene)
        uniforms = dict(position=[0.1] * 3, rotation=[0.0] * 3, velocity=[0.0] * 6, joints=[0.0] * 29)
        self.files = {"placed.json": {"trials": [dict(command_row=c, body_row=r, reset_uniforms=uniforms)
                                                 for c, r in zip(COMMANDS, BODY_ROWS)]},
                      "twins.json": {"trials": [{"command_row": c} for c in COMMANDS]},
                      "first_rows.json": {"trials": [{"command_row": 0}, {"command_row": 1200}]}}
        trials = [dict(trial=i, body_row=r, body_clip="getup", command_row=c, command_clip="walk")
                  for i, (c, r) in enumerate(zip(COMMANDS, BODY_ROWS))]
        scene = dict(settings=dict(hidden_clips=["getup"]), first_row_controls=1199, clips=CLIPS, trials=trials,
                     robot="mjlab_g1", library=LIBRARY)
        for name, content in {**self.files, "scene.json": scene}.items():
            json.dump(content, open(os.path.join(self.scene, name), "w"))
        self.rollouts, self.records = {}, {}
        for seed, (name, file, termination, controls, rows) in enumerate((
                ("placed", "placed.json", "nonfinite", 1000, list(zip(COMMANDS, BODY_ROWS))),
                ("twins", "twins.json", "nonfinite", 1000, list(zip(COMMANDS, COMMANDS))),
                ("first_rows", "first_rows.json", "recipe", 1199, [(0, 0), (1200, 1200)]))):
            path = os.path.join(d, name)
            os.makedirs(path)
            self.records[name] = record(rows, 4, seed)
            np.savez(os.path.join(path, "rollouts.npz"), **self.records[name])
            self.describe(path, scene=self.files[file], termination=termination, controls=controls)
            self.rollouts[name] = path

    def tearDown(self):
        self.directory.cleanup()

    def describe(self, path, **changes):
        """description.json as tools/rollouts.py writes it (the keys the entry reads), with ``changes``."""
        description = dict(run_dir="/runs/recovery_s0", checkpoint="/runs/recovery_s0/ckpt_007000.pt", iteration=7000,
                           run_args=dict(robot="mjlab_g1", seed=0), source_revision=dict(run=None, rollouts="abc"),
                           library=LIBRARY, card="card",
                           bodies=dict(names=BODIES, pelvis=0, torso_link=7), settings=dict(control_dt=0.02))
        path = os.path.join(path, "description.json")
        if os.path.exists(path):
            description = json.load(open(path))
        json.dump({**description, **changes}, open(path, "w"))

    def read(self):
        return ENTRY.read_save(DECLARATION, self.scene, self.rollouts, self.core)

    def test_the_core_is_given_the_record_s_arrays_the_declaration_and_the_scene_s_clips(self):
        result, lines = self.read()
        (call,) = self.core.CALLS
        for name in ENTRY.ROLLOUTS:
            self.assertEqual(sorted(call[name]), sorted(self.core.FIELDS), name)
            for field, value in call[name].items():
                np.testing.assert_array_equal(value, self.records[name][field], err_msg=f"{name} {field}")
        self.assertEqual(call["settings"], DECLARATION["fall_recovery"])
        self.assertEqual((call["dt"], call["pelvis"], call["anchor"]), (0.02, 0, 7))
        self.assertEqual((call["clip_names"], call["clip_first_rows"], call["hidden_clips"]),
                         (["walk", "getup"], [0, 1200], ["getup"]))
        self.assertEqual(result["save"], dict(run="/runs/recovery_s0", checkpoint="/runs/recovery_s0/ckpt_007000.pt",
                                              iteration=7000, run_source_revision=None))
        self.assertEqual((result["trials"], result["R"], result["first_rows"]["C"]), (3, dict(numerator=2),
                                                                                    dict(completed=2)))
        self.assertEqual([(line["trial"], line["reading"]["status"], line["twin_status"]) for line in lines],
                         [(0, "no_fall", "no_fall"), (1, "rejoined", "no_fall"), (2, "rejoined", "no_fall")])
        self.assertEqual(sorted(result["rollouts"]), sorted(ENTRY.ROLLOUTS))
        core = os.path.realpath(os.path.join(self.directory.name, "readings_core.py"))
        self.assertEqual(result["readings_core"], core)

    def test_the_pelvis_and_torso_indices_are_read_from_the_record(self):
        names = BODIES[1:8] + ["pelvis"] + BODIES[8:]
        for name in ENTRY.ROLLOUTS:
            self.describe(self.rollouts[name], bodies=dict(names=names, pelvis=7, torso_link=6))
        self.read()
        self.assertEqual((self.core.CALLS[0]["pelvis"], self.core.CALLS[0]["anchor"]), (7, 6))

    def test_records_of_another_scene_termination_control_count_run_or_library_are_refused(self):
        cases = (("twins", dict(termination="recipe")), ("first_rows", dict(controls=1000)),
                 ("placed", dict(scene=self.files["twins.json"])), ("first_rows", dict(library={**LIBRARY, "rows": 1})),
                 ("placed", dict(settings=dict(control_dt=0.04))), ("twins", dict(run_args=dict(robot="g1", seed=0))),
                 ("twins", dict(iteration=3500)), ("first_rows", dict(checkpoint="/runs/recovery_s0/ckpt_003500.pt")),
                 ("first_rows", dict(run_args=dict(robot="mjlab_g1", seed=1))),
                 ("twins", dict(bodies=dict(names=BODIES, pelvis=0, torso_link=6))))
        for name, changes in cases:
            path = os.path.join(self.rollouts[name], "description.json")
            original = open(path).read()
            self.describe(self.rollouts[name], **changes)
            with self.assertRaises(ValueError, msg=f"{name} {changes}"):
                self.read()
            open(path, "w").write(original)
        self.read()

    def test_records_whose_rows_are_not_the_scene_s_or_without_rollouts_are_refused(self):
        for name, field, trial in (("placed", "command_row", 2), ("placed", "body_row", 0), ("twins", "command_row", 1),
                                   ("first_rows", "command_row", 1)):
            data = {k: v.copy() for k, v in self.records[name].items()}
            data[field][trial] += 1
            np.savez(os.path.join(self.rollouts[name], "rollouts.npz"), **data)
            with self.assertRaisesRegex(ValueError, "command and body rows", msg=f"{name} {field}"):
                self.read()
            np.savez(os.path.join(self.rollouts[name], "rollouts.npz"), **self.records[name])
        os.remove(os.path.join(self.rollouts["twins"], "rollouts.npz"))
        with self.assertRaisesRegex(ValueError, "did not complete"):
            self.read()

    def test_a_directory_without_a_core_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            ENTRY.load_core(self.scene)

    def test_a_scene_whose_twins_are_not_at_the_placed_starts_command_rows_is_refused(self):
        json.dump({"trials": [{"command_row": c + 1} for c in COMMANDS]},
                  open(os.path.join(self.scene, "twins.json"), "w"))
        with self.assertRaisesRegex(ValueError, "twins' command rows"):
            self.read()

    def test_the_program_writes_the_save_s_readings_and_refuses_an_existing_output(self):
        out = os.path.join(self.directory.name, "readings")
        argv = ["--pair", PAIR, "--readings-core", self.directory.name, "--scene", self.scene,
                "--placed", self.rollouts["placed"], "--twins", self.rollouts["twins"],
                "--first_rows", self.rollouts["first_rows"], "--out", out]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            ENTRY.main(argv)
        self.assertEqual(json.loads(printed.getvalue()), dict(save=json.load(open(os.path.join(
            out, "save_readings.json")))["save"], R=2, J=1, J_over_R=0.5, F=0, C=2, E_m=0.03))
        result = json.load(open(os.path.join(out, "save_readings.json")))
        self.assertEqual(sorted(result), ["F", "J", "J_over_R", "R", "engine", "first_rows", "readings_core",
                                          "rollouts", "save", "scene", "trials"])
        lines = [json.loads(line) for line in open(os.path.join(out, "trials.jsonl"))]
        self.assertEqual([line["trial"] for line in lines], [0, 1, 2])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ENTRY.main(argv)


if __name__ == "__main__":
    unittest.main()
