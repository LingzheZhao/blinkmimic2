"""The declaration of the recovery pair (pair.json), the trainings a program of the study is given, and the recovery
options of a condition's preset and of a run's args.json, read alike by scene.py, readings.py and outcome.py."""

import json
import os
import subprocess

from blinkmimic2.recovery import OPTIONS

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The top-level keys of the NablaMimic study's pair.json, whose outcome rules this declaration follows.
KEYS = {"question", "trainings", "saves", "actor_key", "control_dt_s", "placed_starts", "first_rows", "fall_recovery",
        "quantities", "margins", "classes", "outcomes", "unreadable_when"}


def read_declaration(path):
    """pair.json; a declaration with another set of top-level keys is refused."""
    declaration = json.load(open(path))
    if set(declaration) != KEYS:
        raise ValueError(f"{path}: keys {sorted(declaration)}, expected {sorted(KEYS)}")
    return declaration


def training_roots(declaration, items):
    """{(condition, seed): run directory} from --training CONDITION:SEED=RUN arguments; a condition or seed the
    declaration does not name, and a training given twice, are refused."""
    roots = {}
    for item in items:
        name, root = item.split("=", 1)
        condition, seed = name.split(":")
        key = condition, int(seed)
        if condition not in declaration["trainings"]["conditions"] or key[1] not in declaration["trainings"]["seeds"]:
            raise ValueError(f"--training {item}: {condition} seed {seed} is not a training of the declaration")
        if key in roots:
            raise ValueError(f"--training {item}: {condition} seed {seed} is given twice")
        roots[key] = root
    return roots


def preset_options(condition):
    """The recovery options of a condition's preset ("PRESET[name]" in pair.json) in scripts/presets.sh, {option:
    the words that follow it}; {} for a preset without them."""
    name = condition.removeprefix("PRESET[").removesuffix("]")
    command = ["bash", "-c", f'source "{REPO}/scripts/presets.sh" && echo "${{PRESET[{name}]}}"']
    out, option = {}, None
    for word in subprocess.run(command, capture_output=True, text=True, check=True).stdout.split():
        if word.startswith("--"):
            option = word[2:] if word[2:] in OPTIONS else None
            if option:
                out[option] = []
        elif option:
            out[option].append(word)
    return out


def run_options(run_args):
    """The recovery options of a run's args.json as preset_options gives them, the values written as words."""
    return {name: [str(v) for v in (run_args[name] if isinstance(run_args[name], list) else [run_args[name]])]
            for name in OPTIONS if name in run_args}
