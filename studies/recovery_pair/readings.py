"""Readings of one save of the recovery pair (studies/recovery_pair/README.md) from its three tools/rollouts.py
records, by the reading rules of the NablaMimic study; no model, library or simulator is loaded.

    python studies/recovery_pair/readings.py --pair studies/recovery_pair/pair.json --readings-core CORE \\
        --scene SCENE --placed PLACED --twins TWINS --first_rows FIRST_ROWS --out OUT

CORE is the directory of the NablaMimic study (studies/recovery of github.com/LingzheZhao/NablaMimic at the revision
README.md names) that holds readings_core.py, the one owner of the reading rules. SCENE is the output of scene.py;
PLACED, TWINS and FIRST_ROWS are the rollouts of one checkpoint on its placed.json, twins.json and first_rows.json.
A rollout without rollouts.npz, a record of another scene, termination, control count, control period, run,
checkpoint, robot or library, and a record whose command or body rows are not its scene file's are refused. OUT must
not exist; the program writes save_readings.json, which names the readings_core.py that read it, and trials.jsonl (one
line per placed start), and prints the save with R (placed starts that get up), J (placed starts that rejoin), J over
R, F (twins with a counted fall), C (first-row clips completed) and E_m (the aligned body-position error E, m).
"""

import argparse
import importlib.util
import json
import os

import numpy as np

from declaration import read_declaration

# Each rollout's scene file and the section of pair.json that declares its termination.
ROLLOUTS = {"placed": ("placed.json", "placed_starts"), "twins": ("twins.json", "placed_starts"),
            "first_rows": ("first_rows.json", "first_rows")}
# What the three rollouts of one save share: the run, its checkpoint and the tracked bodies.
IDENTITY = ("run_dir", "checkpoint", "iteration", "run_args", "bodies")


def load_core(directory):
    """readings_core.py of the directory ``directory``."""
    spec = importlib.util.spec_from_file_location("readings_core", os.path.join(directory, "readings_core.py"))
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def read_save(declaration, scene_dir, rollouts, core):
    """The readings of one save from ``rollouts`` {name: directory} (the keys of ROLLOUTS) by ``core`` (the module
    readings_core)."""
    record = json.load(open(os.path.join(scene_dir, "scene.json")))
    scenes = {name: json.load(open(os.path.join(scene_dir, file))) for name, (file, _) in ROLLOUTS.items()}
    if [t["command_row"] for t in scenes["placed"]["trials"]] != [t["command_row"] for t in scenes["twins"]["trials"]]:
        raise ValueError(f"{scene_dir}: the twins' command rows are not the placed starts'")
    controls = dict(placed=declaration["placed_starts"]["controls"], twins=declaration["placed_starts"]["controls"],
                    first_rows=record["first_row_controls"])
    descriptions, arrays = {}, {}
    for name, (file, section) in ROLLOUTS.items():
        directory = rollouts[name]
        description = json.load(open(os.path.join(directory, "description.json")))
        if not os.path.isfile(os.path.join(directory, "rollouts.npz")):
            raise ValueError(f"{directory}: the rollouts did not complete")
        expected = dict(scene=scenes[name], termination=declaration[section]["termination"], controls=controls[name],
                        library=record["library"])
        differing = [key for key, value in expected.items() if description[key] != value]
        if description["settings"]["control_dt"] != declaration["control_dt_s"]:
            differing.append("settings.control_dt")
        if description["run_args"]["robot"] != record["robot"]:
            differing.append("run_args.robot")
        if differing:
            raise ValueError(f"{directory}: {differing} are not those of the {file} rollouts of this study")
        with np.load(os.path.join(directory, "rollouts.npz")) as stored:
            rows = [(t["command_row"], t.get("body_row", t["command_row"])) for t in scenes[name]["trials"]]
            if list(zip(stored["command_row"].tolist(), stored["body_row"].tolist())) != rows:
                raise ValueError(f"{directory}: the record's command and body rows are not those of {file}")
            arrays[name] = {key: stored[key] for key in core.FIELDS}
        descriptions[name] = description
    for name, description in descriptions.items():
        if any(description[key] != descriptions["placed"][key] for key in IDENTITY):
            raise ValueError(f"the {name} rollouts are of another run, checkpoint or body list than the placed "
                             "starts'")
    placed = descriptions["placed"]
    clips = record["clips"]
    out = core.save_readings(arrays["placed"], arrays["twins"], arrays["first_rows"], declaration["fall_recovery"],
                             declaration["control_dt_s"], placed["bodies"]["pelvis"], placed["bodies"]["torso_link"],
                             [clip["clip"] for clip in clips], [clip["first_row"] for clip in clips],
                             record["settings"]["hidden_clips"])
    lines = [dict(trial=i, body_clip=t["body_clip"], command_clip=t["command_clip"],
                  twin_status=out["twin_readings"][i]["status"], reading=out["readings"][i])
             for i, t in enumerate(record["trials"])]
    result = dict(save=dict(run=placed["run_dir"], checkpoint=placed["checkpoint"], iteration=placed["iteration"],
                            run_source_revision=placed["source_revision"]["run"]),
                  readings_core=os.path.realpath(core.__file__),
                  rollouts={name: dict(directory=os.path.abspath(rollouts[name]),
                                       source_revision=d["source_revision"]["rollouts"], card=d["card"])
                            for name, d in descriptions.items()},
                  scene=os.path.abspath(scene_dir), trials=len(record["trials"]), **out["quantities"],
                  engine=out["engine"], first_rows=out["first_rows"])
    return result, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read one save of the recovery pair from its three rollouts.")
    parser.add_argument("--pair", required=True, help="the declaration, studies/recovery_pair/pair.json")
    parser.add_argument("--readings-core", required=True, help="the directory that holds readings_core.py")
    parser.add_argument("--scene", required=True, help="output directory of scene.py")
    for name, (file, _) in ROLLOUTS.items():
        parser.add_argument(f"--{name}", required=True, help=f"tools/rollouts.py output on {file}")
    parser.add_argument("--out", required=True, help="output directory; it must not exist")
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error(f"the output directory {args.out} exists")
    result, lines = read_save(read_declaration(args.pair), args.scene, {name: getattr(args, name) for name in ROLLOUTS},
                              load_core(args.readings_core))
    os.makedirs(args.out)
    with open(os.path.join(args.out, "save_readings.json"), "x") as f:
        f.write(json.dumps(result, indent=1, allow_nan=False) + "\n")
    with open(os.path.join(args.out, "trials.jsonl"), "x") as f:
        f.writelines(json.dumps(line, allow_nan=False) + "\n" for line in lines)
    print(json.dumps(dict(save=result["save"], R=result["R"]["numerator"], J=result["J"]["numerator"],
                          J_over_R=result["J_over_R"]["value"], F=result["F"]["numerator"],
                          C=result["first_rows"]["C"]["completed"], E_m=result["first_rows"]["E_m"]["value"])))


if __name__ == "__main__":
    main()
