"""The outcome program of the recovery pair (studies/recovery_pair/outcome.py) on hand-built readings, hand-written run
directories (args.json, train_log.csv, the last checkpoint) and logs."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "studies", "recovery_pair"))
import declaration as DECLARATION_PROGRAM  # noqa: E402
import outcome as OUTCOME  # noqa: E402

PAIR = os.path.join(REPO, "studies", "recovery_pair", "pair.json")
DECLARATION = json.load(open(PAIR))
HIDDEN = ["fallAndGetUp1_subject1", "fallAndGetUp1_subject5", "fallAndGetUp2_subject2", "fallAndGetUp2_subject3",
          "fallAndGetUp3_subject1"]
RECOVERY_ARGS = dict(recovery_share=0.5, recovery_hidden_clips=HIDDEN,
                     recovery_command_excluded_clips=HIDDEN + ["fallAndGetUp1_subject4"],
                     recovery_low_pelvis=[0.5, 0.5], recovery_rows=[50, 375, 500], recovery_reward_scale=2.5,
                     recovery_seconds=[2.0, 1.5])
# Example readings of the eight saves: R (placed starts that get up), J over R (rejoined over the trials that got up
# with a readable rejoin), F (twins with a counted fall), C (first-row clips completed) and E (aligned body-position
# error, m).
SAVES = {("control", 0, 3500): (24, 22 / 24, 0, 40, 0.0345), ("control", 0, 7000): (35, 34 / 35, 0, 40, 0.0286),
         ("control", 1, 3500): (26, 1.0, 1, 40, 0.0343), ("control", 1, 7000): (58, 55 / 58, 0, 40, 0.0285),
         ("recovery", 0, 3500): (99, 1.0, 2, 36, 0.0542), ("recovery", 0, 7000): (99, 1.0, 1, 39, 0.0390),
         ("recovery", 1, 3500): (99, 98 / 99, 0, 38, 0.0533), ("recovery", 1, 7000): (99, 1.0, 0, 39, 0.0385)}
KEYS = [(condition, seed) for condition in ("control", "recovery") for seed in (0, 1)]


def values(changes=None, getup=True):
    out = {key: dict(R=r, J_over_R=j, F=f, C=c, E_m=e, C_getup={clip: getup for clip in HIDDEN})
           for key, (r, j, f, c, e) in SAVES.items()}
    for key, change in (changes or {}).items():
        out[key] = {**out[key], **change}
    return out


def at(**saves):
    """Changes {(condition, seed, iteration): {name: value}} from keywords such as recovery_0_7000=dict(R=3)."""
    return {(name.split("_")[0], int(name.split("_")[1]), int(name.split("_")[2])): change
            for name, change in saves.items()}


class OutcomeOrderTest(unittest.TestCase):
    def outcome(self, changes=None, getup=True):
        return OUTCOME.outcome(DECLARATION, values(changes, getup))

    def test_the_example_saves_give_tracking_cost_at_exposure(self):
        result = self.outcome()
        self.assertEqual(result["outcome"], "tracking_cost_at_exposure")
        self.assertEqual(result["classes"], {0: "gets_up", 1: "gets_up"})
        self.assertEqual(result["larger_control_R"], 58)
        self.assertEqual((result["margins"]["tracking_E_m_3500"], result["margins"]["tracking_C_7000"]), (0.005, 2))
        self.assertEqual(result["tracking"]["seed 0 against control 7000"]["terms"], dict(C=True, E=False, F=True))
        self.assertTrue(result["tracking"]["seed 1 against control 3500"]["within"])
        self.assertEqual([row["name"] for row in result["rows_in_order"]],
                         [item["name"] for item in DECLARATION["outcomes"]])

    def test_hand_built_readings_reach_each_outcome_in_the_declared_order(self):
        # run_stopped, the first row, cannot hold for this trainer
        cases = {
            "split_seeds": dict(changes=at(recovery_1_7000=dict(R=3))),
            "control_within": dict(changes=at(control_0_7000=dict(R=95), control_1_7000=dict(R=92))),
            "precondition_unmet": dict(changes=at(recovery_0_7000=dict(R=30), recovery_1_7000=dict(R=30)),
                                       getup=False),
            "still_rising": dict(changes=at(recovery_0_3500=dict(R=10), recovery_0_7000=dict(R=30),
                                            recovery_1_3500=dict(R=10), recovery_1_7000=dict(R=30))),
            "no_get_up": dict(changes=at(recovery_0_3500=dict(R=1), recovery_0_7000=dict(R=2),
                                         recovery_1_3500=dict(R=2), recovery_1_7000=dict(R=3))),
            "some_get_up": dict(changes=at(recovery_0_3500=dict(R=28), recovery_0_7000=dict(R=30),
                                           recovery_1_3500=dict(R=2), recovery_1_7000=dict(R=3))),
            "gets_up_without_rejoining": dict(changes=at(recovery_0_7000=dict(J_over_R=0.3))),
            "gets_up_and_tracks": dict(changes=at(recovery_0_7000=dict(E_m=0.030), recovery_1_7000=dict(E_m=0.030))),
            "tracking_cost_at_exposure": dict(),
            "tracking_worse": dict(changes=at(recovery_0_7000=dict(E_m=0.05), recovery_1_7000=dict(E_m=0.05))),
            "no_call": dict(changes=at(recovery_0_7000=dict(E_m=0.030)))}
        self.assertEqual(["run_stopped", *cases], [item["name"] for item in DECLARATION["outcomes"]])
        for name, case in cases.items():
            result = self.outcome(**case)
            self.assertEqual(result["outcome"], name, name)
            self.assertFalse(result["rows_in_order"][0]["holds"])

    def test_an_unavailable_j_over_r_holds_neither_below_nor_at_its_margin(self):
        result = self.outcome(at(recovery_0_7000=dict(J_over_R=None)))
        self.assertEqual((result["outcome"], result["J_over_R_unavailable_at_seeds"]), ("no_call", [0]))

    def test_the_classes_and_margins_at_their_bounds(self):
        # near_zero at R <= 5; gets_up at R >= 50 and R >= the larger control R at 7,000 + 20 (58 + 20)
        for r, expected in ((5, "near_zero"), (6, "some"), (77, "some"), (78, "gets_up")):
            found = OUTCOME.classes(DECLARATION, values(at(recovery_0_7000=dict(R=r), recovery_1_7000=dict(R=r))))
            self.assertEqual(found, {0: expected, 1: expected}, r)
        control = values()["control", 0, 7000]
        for change, within in ((dict(C=38), True), (dict(C=37), False), (dict(E_m=control["E_m"] + 0.005), True),
                               (dict(E_m=control["E_m"] + 0.005 + 1e-9), False), (dict(F=2), True),
                               (dict(F=3), False)):
            found = values(at(recovery_0_7000=dict(dict(C=40, E_m=control["E_m"], F=0), **change)))
            bounds = OUTCOME.margins(DECLARATION, found)
            self.assertEqual(OUTCOME.tracking_within(found, bounds, 0, 7000, 7000)["within"], within, change)
        bounds = OUTCOME.margins(DECLARATION, values(at(control_1_7000=dict(C=35, E_m=0.0386),
                                                        recovery_1_7000=dict(R=80))))
        self.assertEqual((bounds["tracking_C_7000"], bounds["still_rising"]), (5, 19))
        self.assertAlmostEqual(bounds["tracking_E_m_7000"], 0.01)


def config(condition, run_seed, **changes):
    """An args.json of the fields the program reads, with ``changes``."""
    out = dict(seed=run_seed, iterations=7000, ckpt_every=100, log_every=100, log_dir=f"/runs/{condition}_s{run_seed}",
               pi_batch=2048, horizon=32, lr=0.0004, robot="mjlab_g1", motion_dir="/data/clips", bones_list="")
    return out | (RECOVERY_ARGS if condition == "recovery" else {}) | changes


def write_log(path, recovery, last=7000, header_again=False):
    """A train_log.csv with a row every 100 iterations up to ``last``; a recovery log's running total of imagined
    tracking controls grows by 30,000 per iteration."""
    columns = "iteration,wall_h" + (",recovery_imagined_tracking_total" if recovery else "") + ",nonfinite"
    rows = [f"{it},{it / 7000:.6g}" + (f",{30_000 * it}" if recovery else "") + ",0"
            for it in range(100, last + 1, 100)]
    if header_again:
        rows.insert(10, columns)
    with open(path, "w") as f:
        f.write("\n".join([columns, *rows]) + "\n")


def write_run(root, condition, seed, **changes):
    os.makedirs(root)
    json.dump(config(condition, seed, log_dir=root, **changes), open(os.path.join(root, "args.json"), "w"))
    write_log(os.path.join(root, "train_log.csv"), condition == "recovery")
    open(os.path.join(root, "ckpt_007000.pt"), "w").close()


def reading(root, condition, seed, iteration):
    r, j, f, c, e = SAVES[condition, seed, iteration]
    save = dict(run=root, checkpoint=os.path.join(root, f"ckpt_{iteration:06d}.pt"), iteration=iteration)
    return dict(save=save, readings_core="/nablamimic/studies/recovery/readings_core.py", trials=100,
                R=dict(numerator=r), J_over_R=dict(value=j), F=dict(numerator=f),
                first_rows=dict(C=dict(completed=c), E_m=dict(value=e), C_getup={clip: True for clip in HIDDEN}))


class TrainingTest(unittest.TestCase):
    def test_completion_is_the_last_checkpoint_and_the_log_s_last_row_at_the_last_iteration(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "run")
            write_run(root, "control", 0)
            end = OUTCOME.training_end(root, OUTCOME.log_rows(root), 7000)
            self.assertEqual(end, dict(completed=True, last_checkpoint=True, last_log_iteration=7000))
            write_log(os.path.join(root, "train_log.csv"), False, last=6900)
            self.assertFalse(OUTCOME.training_end(root, OUTCOME.log_rows(root), 7000)["completed"])
            write_log(os.path.join(root, "train_log.csv"), False, last=0)
            self.assertEqual(OUTCOME.training_end(root, OUTCOME.log_rows(root), 7000)["last_log_iteration"], None)
            write_log(os.path.join(root, "train_log.csv"), False)
            os.remove(os.path.join(root, "ckpt_007000.pt"))
            self.assertEqual(OUTCOME.training_end(root, OUTCOME.log_rows(root), 7000),
                             dict(completed=False, last_checkpoint=False, last_log_iteration=7000))
            write_log(os.path.join(root, "train_log.csv"), False, header_again=True)
            with self.assertRaisesRegex(ValueError, "second header"):
                OUTCOME.log_rows(root)

    def test_the_identity_rules_on_the_trainings_args_json(self):
        configs = {key: config(*key) for key in KEYS}
        self.assertEqual(OUTCOME.training_problems(DECLARATION, configs), [])
        cases = dict(other_seed={("control", 0): config("control", 0, seed=1)},
                     other_iterations={("recovery", 1): config("recovery", 1, iterations=3500)},
                     other_interval={("control", 1): config("control", 1, ckpt_every=500)},
                     other_log_interval={("recovery", 0): config("recovery", 0, log_every=50)},
                     options_of_the_control={("recovery", 0): config("control", 0)},
                     options_in_the_control={("control", 0): config("recovery", 0)},
                     other_option_value={("recovery", 0): config("recovery", 0, recovery_reward_scale=2.0)},
                     other_learning_rate={("recovery", 1): config("recovery", 1, lr=0.0003)},
                     one_side_only={("control", 1): config("control", 1, amp="bf16")})
        for name, change in cases.items():
            self.assertTrue(OUTCOME.training_problems(DECLARATION, configs | change), name)
        # the recovery options and the log directory are the only args.json keys a seed's two trainings differ in
        self.assertEqual(sorted(set(config("recovery", 0)) - set(config("control", 0))), sorted(RECOVERY_ARGS))
        other_log_dir = configs | {("control", 0): config("control", 0, log_dir="/elsewhere")}
        self.assertEqual(OUTCOME.training_problems(DECLARATION, other_log_dir), [])

    def test_the_declaration_and_the_trainings_given_are_refused_outside_the_declared_ones(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pair.json")
            json.dump(DECLARATION | {"wall_time_s": 1}, open(path, "w"))
            with self.assertRaisesRegex(ValueError, "keys"):
                DECLARATION_PROGRAM.read_declaration(path)
        self.assertEqual(DECLARATION_PROGRAM.training_roots(DECLARATION, ["recovery:1=/runs/r"]),
                         {("recovery", 1): "/runs/r"})
        for items in (["stopped:0=/runs/a"], ["control:2=/runs/a"], ["control:0=/runs/a", "control:0=/runs/b"]):
            with self.assertRaises(ValueError, msg=items):
                DECLARATION_PROGRAM.training_roots(DECLARATION, items)

    def test_the_presets_recovery_options_are_those_args_json_records_for_each_condition(self):
        self.assertEqual(DECLARATION_PROGRAM.preset_options("PRESET[recovery_control]"), {})
        self.assertEqual(DECLARATION_PROGRAM.preset_options("PRESET[recovery]"),
                         DECLARATION_PROGRAM.run_options(config("recovery", 0)))
        self.assertEqual(DECLARATION_PROGRAM.run_options(config("control", 0)), {})

    def test_the_tracking_exposure_is_read_from_the_recovery_log_against_the_control_s_controls_per_iteration(self):
        with tempfile.TemporaryDirectory() as d:
            logs = {}
            for condition in ("control", "recovery"):
                os.makedirs(os.path.join(d, condition))
                write_log(os.path.join(d, condition, "train_log.csv"), condition == "recovery")
                logs[condition, 0] = OUTCOME.log_rows(os.path.join(d, condition))
            configs = {key: config(*key) for key in logs}
            result = OUTCOME.exposure(DECLARATION, configs, logs)
            self.assertEqual(result["control seed 0"]["imagined_tracking_controls"],
                             {"3500": 3500 * 65_536, "7000": 7000 * 65_536})
            recovery = result["recovery seed 0"]
            self.assertEqual(recovery["imagined_tracking_controls"], {"3500": 105_000_000, "7000": 210_000_000})
            # the first control iteration with at least 210,000,000 imagined tracking controls at 65,536 per iteration
            self.assertEqual(recovery["matched_control_iteration"], {"3500": 1603, "7000": 3205})
            self.assertEqual((3204 * 65_536 < 210_000_000, 3205 * 65_536 >= 210_000_000), (True, True))
            logs["recovery", 0] = [row for row in logs["recovery", 0] if row["iteration"] != "3500"]
            with self.assertRaisesRegex(ValueError, "no row at iterations"):
                OUTCOME.exposure(DECLARATION, configs, logs)


class ProgramTest(unittest.TestCase):
    """outcome.py on run directories and save readings written into a temporary directory."""

    def write(self, d):
        arguments, saves = ["--pair", PAIR], []
        for condition, seed in KEYS:
            root = os.path.join(d, "runs", f"{condition}_s{seed}")
            write_run(root, condition, seed)
            arguments += ["--training", f"{condition}:{seed}={root}"]
            for iteration in (3500, 7000):
                save = os.path.join(d, "readings", f"{condition}_s{seed}-{iteration}")
                os.makedirs(save)
                json.dump(reading(root, condition, seed, iteration),
                          open(os.path.join(save, "save_readings.json"), "w"))
                saves += ["--save", save]
        return arguments, saves

    def run_program(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            OUTCOME.main(argv)
        return json.load(open(os.path.join(argv[argv.index("--out") + 1], "outcome.json")))

    def test_the_program_writes_the_outcome_with_the_exposure_and_refuses_an_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            arguments, saves = self.write(d)
            out = os.path.join(d, "outcome")
            result = self.run_program(arguments + saves + ["--out", out])
            self.assertEqual((result["outcome"]["outcome"], result["unreadable"]), ("tracking_cost_at_exposure", []))
            self.assertEqual(result["exposure"]["recovery seed 1"]["matched_control_iteration"]["7000"], 3205)
            self.assertTrue(all(item["end"]["completed"] for item in result["trainings"].values()))
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                OUTCOME.main(arguments + saves + ["--out", out])

    def test_each_unreadable_rule_gives_no_outcome_and_its_reason(self):
        with tempfile.TemporaryDirectory() as d:
            arguments, saves = self.write(d)
            # a save without a reading
            result = self.run_program(arguments + saves[:-2] + ["--out", os.path.join(d, "a")])
            self.assertEqual((result["outcome"]["outcome"], len(result["unreadable"])), (None, 1))
            self.assertIn("no reading", result["unreadable"][0])
            # a training that did not complete its iterations
            root = os.path.join(d, "runs", "control_s1")
            write_log(os.path.join(root, "train_log.csv"), False, last=6900)
            result = self.run_program(arguments + saves + ["--out", os.path.join(d, "b")])
            self.assertEqual((result["outcome"]["outcome"], result["exposure"]), (None, None))
            self.assertIn("did not complete", result["unreadable"][0])
            write_log(os.path.join(root, "train_log.csv"), False)
            # args.json of a seed's two trainings that differ outside the recovery options and the log directory
            json.dump(config("recovery", 0, log_dir=os.path.join(d, "runs", "recovery_s0"), lr=0.0003),
                      open(os.path.join(d, "runs", "recovery_s0", "args.json"), "w"))
            result = self.run_program(arguments + saves + ["--out", os.path.join(d, "c")])
            self.assertEqual(result["unreadable"], ["seed 0: the two args.json files differ in ['lr']"])
            json.dump(config("recovery", 0, log_dir=os.path.join(d, "runs", "recovery_s0")),
                      open(os.path.join(d, "runs", "recovery_s0", "args.json"), "w"))
            # a training not given
            result = self.run_program(arguments[:-2] + saves[:-4] + ["--out", os.path.join(d, "e")])
            self.assertEqual(result["unreadable"][-1], "recovery seed 1: training not given")
            self.assertEqual((result["outcome"]["outcome"], len(result["unreadable"])), (None, 3))
            # a save whose E is unavailable, a save of another trial count, and saves read by two readings cores
            path = os.path.join(saves[1], "save_readings.json")
            kept = json.load(open(path))
            for name, change, reason in (
                    ("f", dict(first_rows={**kept["first_rows"], "E_m": dict(value=None)}), "E_m unavailable"),
                    ("g", dict(trials=99), "99 placed starts"),
                    ("h", dict(readings_core="/elsewhere/readings_core.py"), "more than one readings_core.py")):
                json.dump(kept | change, open(path, "w"))
                result = self.run_program(arguments + saves + ["--out", os.path.join(d, name)])
                self.assertEqual((result["outcome"]["outcome"], len(result["unreadable"])), (None, 1), name)
                self.assertIn(reason, result["unreadable"][0])
            json.dump(kept, open(path, "w"))
            # a reading of a run not given, or two readings of one save
            stranger = os.path.join(d, "readings", "stranger")
            os.makedirs(stranger)
            json.dump(reading(os.path.join(d, "elsewhere"), "control", 0, 3500),
                      open(os.path.join(stranger, "save_readings.json"), "w"))
            for extra in (["--save", stranger], saves[:2]):
                with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                    OUTCOME.main(arguments + saves + extra + ["--out", os.path.join(d, "d")])


if __name__ == "__main__":
    unittest.main()
