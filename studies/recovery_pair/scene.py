"""The scene files of the recovery pair (studies/recovery_pair/README.md) for tools/rollouts.py: the placed lying
starts, their twins and the first row of every clip, on the 40-clip library of the trainings.

    python studies/recovery_pair/scene.py --pair studies/recovery_pair/pair.json --run_dir RUNS/recovery_s0 --out SCENE

RUN_DIR is a recovery training of the pair. Its args.json gives the library, built from the motion folder as
tools/rollouts.py builds it (here on the CPU), and the recovery settings, which must be those of PRESET[recovery]; the
row counts of the build must equal those of its recovery.json. SCENE must not exist; the program writes into it
placed.json, twins.json and first_rows.json (scene files of tools/rollouts.py) and scene.json (the settings, the six
row counts, the command rows with 1,000 upright rows ahead, the controls of the first-row rollouts, every clip and
trial, and the library inputs).

The placed starts: body rows evenly spaced over the start rows, command rows evenly spaced over the command rows with
1,000 upright rows ahead, both by rounding a float64 linspace over the row indices; unit uniforms from one torch CPU
generator in the order position, rotation, velocity, joints; then a permutation from the same generator pairs the
command rows with the body rows.
"""

import argparse
import dataclasses
import json
import os

import torch

from declaration import preset_options, read_declaration, run_options

from blinkmimic2 import recovery
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.motion import MotionLib
from blinkmimic2.robot import get_robot

# The keys of args.json that tools/rollouts.py builds the library from.
LIBRARY = ("motion_dir", "bones_list", "bones_lib", "motion_store")


def evenly(values, count):
    """``count`` entries of ``values`` at the rounded points of a float64 linspace over its indices."""
    return values[torch.linspace(0, values.shape[0] - 1, count, dtype=torch.float64).round().long()]


def draws(seed, count, joints):
    """The unit uniforms {name: [count, n] float32} of a scene's reset_uniforms, drawn in the order position,
    rotation, velocity, joints, and then the pairing permutation [count], from one torch CPU generator seeded with
    ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    widths = {"position": 3, "rotation": 3, "velocity": 6, "joints": joints}
    uniforms = {name: 2.0 * torch.rand((count, width), generator=generator, dtype=torch.float32) - 1.0
                for name, width in widths.items()}
    return uniforms, torch.randperm(count, generator=generator)


def scene(motion, settings, placed, joints):
    """The scene files and their record on ``motion`` (a MotionLib) under the recovery ``settings`` and the
    declaration's ``placed_starts``, for a robot of ``joints`` joints."""
    rows = recovery.build(motion, settings)
    ahead = recovery.build(motion, dataclasses.replace(settings, command_upright_rows=placed["command_upright_rows"]))
    count = placed["trials"]
    body = evenly(rows.start_rows.cpu(), count).tolist()
    uniforms, permutation = draws(placed["uniforms_seed"], count, joints)
    command = evenly(ahead.command_rows.cpu(), count)[permutation].tolist()
    starts, ends = motion.clip_start.tolist(), motion.clip_end.tolist()

    def where(row):
        clip = int(motion.clip_id[row])
        return motion.names[clip], row - starts[clip]

    trials = []
    for i in range(count):
        (body_clip, body_in_clip), (command_clip, command_in_clip) = where(body[i]), where(command[i])
        trials.append(dict(trial=i, body_row=body[i], body_clip=body_clip, body_row_in_clip=body_in_clip,
                           command_row=command[i], command_clip=command_clip, command_row_in_clip=command_in_clip,
                           permutation=int(permutation[i])))
    clips = [dict(clip=name, first_row=first, last_row=last, controls=last - first)
             for name, first, last in zip(motion.names, starts, ends)]
    record = dict(settings=dataclasses.asdict(settings), rows=rows.counts,
                  command_rows_with_upright_rows_ahead=ahead.counts["command_rows"], placed_starts=placed,
                  first_row_controls=max(clip["controls"] for clip in clips), clips=clips, trials=trials)
    files = {"placed.json": {"trials": [dict(command_row=t["command_row"], body_row=t["body_row"],
                                             reset_uniforms={k: v[i].tolist() for k, v in uniforms.items()})
                                        for i, t in enumerate(trials)]},
             "twins.json": {"trials": [{"command_row": t["command_row"]} for t in trials]},
             "first_rows.json": {"trials": [{"command_row": clip["first_row"]} for clip in clips]}}
    return files, record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the scene files of the recovery pair.")
    parser.add_argument("--pair", required=True, help="the declaration, studies/recovery_pair/pair.json")
    parser.add_argument("--run_dir", required=True, help="a recovery training of the pair (args.json, recovery.json)")
    parser.add_argument("--out", required=True, help="output directory; it must not exist")
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error(f"the output directory {args.out} exists")
    declaration = read_declaration(args.pair)
    run = json.load(open(os.path.join(args.run_dir, "args.json")))
    preset = declaration["trainings"]["conditions"]["recovery"]
    if run_options(run) != preset_options(preset):
        raise ValueError(f"{args.run_dir}: the recovery options of args.json are not those of {preset}")
    robot = get_robot(run["robot"])
    library = {key: run[key] for key in LIBRARY}
    motion = MotionLib.load(robot, Kinematics(robot, "cpu"), "cpu", *library.values())
    if len(motion.names) != 40:
        raise ValueError(f"the library holds {len(motion.names)} clips, not 40")
    files, record = scene(motion, recovery.settings_from_args(argparse.Namespace(**run)), declaration["placed_starts"],
                          robot.num_joints)
    counts = json.load(open(os.path.join(args.run_dir, "recovery.json")))["counts"]
    if record["rows"] != counts:
        raise ValueError(f"the row counts {record['rows']} are not those of the training, {counts}")
    record.update(run_dir=os.path.abspath(args.run_dir), robot=run["robot"],
                  library=library | {"clips": len(motion.names), "rows": motion.num_frames})
    os.makedirs(args.out)
    for name, content in {**files, "scene.json": record}.items():
        with open(os.path.join(args.out, name), "x") as f:
            f.write(json.dumps(content, indent=1, allow_nan=False) + "\n")
    print(json.dumps(dict(rows=record["rows"], first_row_controls=record["first_row_controls"],
                          command_rows_with_upright_rows_ahead=record["command_rows_with_upright_rows_ahead"])))


if __name__ == "__main__":
    main()
