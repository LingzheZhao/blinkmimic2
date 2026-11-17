"""The outcome of the recovery pair (studies/recovery_pair/README.md): the declared outcome order of pair.json applied
to the readings of the eight saves, with the trainings' tracking exposure beside it, printed and written with every
reading it used.

    python studies/recovery_pair/outcome.py --pair studies/recovery_pair/pair.json \\
        --training control:0=RUN --training recovery:0=RUN --training control:1=RUN --training recovery:1=RUN \\
        --save READINGS [--save READINGS ...] --out OUT

RUN is a blinkmimic2.train run directory (args.json, train_log.csv, ckpt_NNNNNN.pt); READINGS an output directory of
readings.py, whose save names its run and checkpoint iteration. OUT must not exist; the program writes outcome.json.
Each seed's class and the outcome rows are those of pair.json, read from each save's R (placed starts that get up),
J over R (rejoined over the trials that got up with a readable rejoin), F (twins with a counted fall), C (first-row
clips completed), C_getup (completion of each hidden get-up clip) and E (aligned body-position error). Nothing is
pooled across saves or seeds.

save_values, margins, classes, tracking_within, outcome_rules and outcome are the rule functions of the NablaMimic
study's outcome.py (studies/recovery/outcome.py of github.com/LingzheZhao/NablaMimic at the revision README.md names),
copied here because that file imports the NablaMimic package at its top; at that revision they are the same but for the
docstring of outcome_rules, and the copy does not follow later revisions. Reading a training's args.json, its
completion and its tracking exposure are this trainer's: this trainer writes no record of a run's end, so a training
completed when its checkpoint at the last iteration exists and the last row of its train_log.csv is at that
iteration.
"""

import argparse
import csv
import json
import math
import os

from declaration import preset_options, read_declaration, run_options, training_roots

from blinkmimic2.recovery import OPTIONS

# The args.json keys in which the two trainings of a seed may differ: the seven recovery options and the output.
PAIR_DIFFERENCES = (*OPTIONS, "log_dir")
# The log column of a recovery run's imagined tracking-regime controls since the run started (RECOVERY.md, Records).
TRACKING_TOTAL = "recovery_imagined_tracking_total"


def log_rows(root):
    """The rows of a run's train_log.csv. train.py writes the header at every start and has no resumption, so a log
    that holds a second header, of a second start in the same directory, is refused."""
    with open(os.path.join(root, "train_log.csv")) as f:
        rows = list(csv.DictReader(f))
    if any(row["iteration"] == "iteration" for row in rows):
        raise ValueError(f"{root}: train_log.csv holds a second header, of a second start")
    return rows


def training_end(root, rows, iterations):
    """Whether a training completed its ``iterations``: its checkpoint ckpt_NNNNNN.pt at the last iteration exists
    and the last of its log ``rows`` is at that iteration."""
    checkpoint = os.path.isfile(os.path.join(root, f"ckpt_{iterations:06d}.pt"))
    last = int(rows[-1]["iteration"]) if rows else None
    return dict(completed=checkpoint and last == iterations, last_checkpoint=checkpoint, last_log_iteration=last)


def training_problems(declaration, configs):
    """The declaration's unreadable_when on the trainings' args.json files {(condition, seed): args}."""
    trainings = declaration["trainings"]
    declared = dict(iterations=trainings["iterations"], ckpt_every=trainings["checkpoint_every"],
                    log_every=trainings["log_every"])
    options = {condition: preset_options(preset) for condition, preset in trainings["conditions"].items()}
    problems = []
    for (condition, seed), config in configs.items():
        problems += [f"{condition} seed {seed}: args.json {name} {config[name]!r}, declared {value!r}"
                     for name, value in dict(seed=seed, **declared).items() if config[name] != value]
        if run_options(config) != options[condition]:
            problems.append(f"{condition} seed {seed}: its recovery options are not those of "
                            f"{trainings['conditions'][condition]}")
    for seed in trainings["seeds"]:
        if ("control", seed) in configs and ("recovery", seed) in configs:
            a, b = configs["control", seed], configs["recovery", seed]
            differing = sorted(key for key in set(a) | set(b)
                               if key not in PAIR_DIFFERENCES and (key not in a or key not in b or a[key] != b[key]))
            if differing:
                problems.append(f"seed {seed}: the two args.json files differ in {differing}")
    return problems


def exposure(declaration, configs, logs):
    """At each save, the imagined tracking-regime controls each training has trained on (RECOVERY.md, Records): a
    recovery training's recovery_imagined_tracking_total at that iteration's log row, a control training's iterations
    times pi_batch times horizon; and for each recovery save, the first control iteration whose total reaches it."""
    out = {}
    for (condition, seed), config in configs.items():
        per_iteration = config["pi_batch"] * config["horizon"]
        if condition == "recovery":
            by_iteration = {int(row["iteration"]): row for row in logs[condition, seed]}
            missing = [save for save in declaration["saves"] if save not in by_iteration]
            if missing:
                raise ValueError(f"the recovery seed {seed} log holds no row at iterations {missing}")
            totals = {str(save): int(by_iteration[save][TRACKING_TOTAL]) for save in declaration["saves"]}
        else:
            totals = {str(save): save * per_iteration for save in declaration["saves"]}
        out[f"{condition} seed {seed}"] = dict(imagined_controls_per_iteration=per_iteration,
                                               imagined_tracking_controls=totals)
    for seed in declaration["trainings"]["seeds"]:
        recovery, control = out.get(f"recovery seed {seed}"), out.get(f"control seed {seed}")
        if recovery and control:
            recovery["matched_control_iteration"] = {
                save: math.ceil(total / control["imagined_controls_per_iteration"])
                for save, total in recovery["imagined_tracking_controls"].items()}
    return out


# ---- the rule functions of the NablaMimic study's outcome.py ----

def save_values(reading):
    """The numbers the outcome reads from one save's readings; None where unavailable."""
    first_rows = reading["first_rows"]
    return dict(R=reading["R"]["numerator"], J_over_R=reading["J_over_R"]["value"], F=reading["F"]["numerator"],
                C=first_rows["C"]["completed"], E_m=first_rows["E_m"]["value"], C_getup=first_rows["C_getup"])


def margins(declaration, values):
    declared = declaration["margins"]
    seeds, saves = declaration["trainings"]["seeds"], declaration["saves"]
    out = {name: declared[name] for name in ("near_zero", "gets_up_R", "gets_up_lead", "rejoins_J_over_R",
                                             "tracking_F", "control_within")}
    for updates in saves:
        control = [values["control", seed, updates] for seed in seeds]
        out[f"tracking_C_{updates}"] = max(declared["tracking_C"]["least"], abs(control[0]["C"] - control[1]["C"]))
        out[f"tracking_E_m_{updates}"] = max(declared["tracking_E_m"]["least"],
                                             abs(control[0]["E_m"] - control[1]["E_m"]))
    recovery = [values["recovery", seed, saves[-1]]["R"] for seed in seeds]
    out["still_rising"] = max(abs(recovery[0] - recovery[1]), declared["still_rising"]["least"])
    return out


def classes(declaration, values):
    """Each seed's class of its recovery save at the last update, against the larger control R there."""
    declared, seeds, last = declaration["margins"], declaration["trainings"]["seeds"], declaration["saves"][-1]
    larger = max(values["control", seed, last]["R"] for seed in seeds)
    out = {}
    for seed in seeds:
        r = values["recovery", seed, last]["R"]
        if r <= declared["near_zero"]:
            out[seed] = "near_zero"
        elif r >= declared["gets_up_R"] and r >= larger + declared["gets_up_lead"]:
            out[seed] = "gets_up"
        else:
            out[seed] = "some"
    return out


def tracking_within(values, bounds, seed, updates, last):
    recovery, control = values["recovery", seed, last], values["control", seed, updates]
    terms = dict(C=recovery["C"] >= control["C"] - bounds[f"tracking_C_{updates}"],
                 E=recovery["E_m"] <= control["E_m"] + bounds[f"tracking_E_m_{updates}"],
                 F=recovery["F"] <= control["F"] + bounds["tracking_F"])
    return dict(within=all(terms.values()), terms=terms)


def outcome_rules(declaration, values):
    """Whether each outcome row of the declaration holds, with the readings the rows read. run_stopped is False here,
    so that the rows are the declaration's whole order: this trainer does not read the simulator's capacity during
    training (read_pair)."""
    seeds, (first, last) = declaration["trainings"]["seeds"], declaration["saves"]
    bounds = margins(declaration, values)
    seed_class = classes(declaration, values)
    gets_up = [seed_class[seed] == "gets_up" for seed in seeds]
    near = [seed_class[seed] == "near_zero" for seed in seeds]
    # J over R enters only the rows that need both seeds gets_up. Where no recovered trial's rejoin was readable it is
    # unavailable: neither below the margin nor at it, so the rows that need its value at that seed do not hold.
    j_over_r = [values["recovery", seed, last]["J_over_R"] for seed in seeds]
    below = [value is not None and value < bounds["rejoins_J_over_R"] for value in j_over_r]
    rejoins = [value is not None and value >= bounds["rejoins_J_over_R"] for value in j_over_r]
    tracking = {(seed, updates): tracking_within(values, bounds, seed, updates, last)
                for seed in seeds for updates in (first, last)}
    rise = {seed: values["recovery", seed, last]["R"] - values["recovery", seed, first]["R"] for seed in seeds}
    getup_completions = [done for seed in seeds for updates in (first, last)
                         for done in values["control", seed, updates]["C_getup"].values()]
    up_and_rejoin = all(gets_up) and all(rejoins)
    rules = dict(
        run_stopped=False,
        split_seeds=(gets_up[0] and near[1]) or (gets_up[1] and near[0]),
        control_within=all(values["control", seed, last]["R"] > bounds["near_zero"]
                           and abs(values["control", seed, last]["R"] - values["recovery", seed, last]["R"])
                           <= bounds["control_within"] for seed in seeds),
        precondition_unmet=not any(gets_up) and not any(getup_completions),
        still_rising=any(not gets_up[index] and rise[seed] > bounds["still_rising"]
                         for index, seed in enumerate(seeds)),
        no_get_up=all(near),
        some_get_up=not any(gets_up),
        gets_up_without_rejoining=all(gets_up) and any(below),
        gets_up_and_tracks=up_and_rejoin and all(tracking[seed, last]["within"] for seed in seeds),
        tracking_cost_at_exposure=up_and_rejoin and all(
            not tracking[seed, last]["within"] and tracking[seed, first]["within"] for seed in seeds),
        tracking_worse=up_and_rejoin and all(
            not tracking[seed, last]["within"] and not tracking[seed, first]["within"] for seed in seeds),
        no_call=True)
    return dict(rules=rules, classes=seed_class, margins=bounds, rise=rise,
                J_over_R_at_least_margin=dict(zip(seeds, rejoins)),
                J_over_R_unavailable_at_seeds=[seed for seed, value in zip(seeds, j_over_r) if value is None],
                tracking={f"seed {seed} against control {updates}": item for (seed, updates), item in tracking.items()},
                larger_control_R=max(values["control", seed, last]["R"] for seed in seeds))


def outcome(declaration, values):
    """The first outcome row of the declaration that holds, with every row's value."""
    found = outcome_rules(declaration, values)
    names = [item["name"] for item in declaration["outcomes"]]
    if names != list(found["rules"]):
        raise ValueError(f"the declaration's outcome rows {names} are not the rows of this program")
    return dict(outcome=next(name for name in names if found["rules"][name]),
                rows_in_order=[dict(name=name, holds=found["rules"][name]) for name in names],
                **{key: value for key, value in found.items() if key != "rules"})


# ---- this trainer's ----

def read_pair(declaration, readings, trainings, configs, logs):
    """The pair's outcome from the save ``readings`` and the trainings {(condition, seed): run directory} with their
    args.json files and log rows. run_stopped cannot be read here: this trainer does not read the simulator's capacity
    during training, and a training that ends early makes the pair unreadable."""
    declared = declaration["trainings"]
    wanted = [(condition, seed, iteration) for condition in declared["conditions"] for seed in declared["seeds"]
              for iteration in declaration["saves"]]
    runs = {os.path.realpath(root): key for key, root in trainings.items()}
    by_save = {}
    for reading in readings:
        save = reading["save"]
        if os.path.realpath(save["run"]) not in runs:
            raise ValueError(f"the save of {save['run']} is not of a training given")
        key = (*runs[os.path.realpath(save["run"])], save["iteration"])
        if key in by_save:
            raise ValueError(f"two readings of the save {key}")
        by_save[key] = reading
    values, unreadable = {}, []
    for key in wanted:
        if key not in by_save:
            unreadable.append(f"{key}: no reading of this save")
            continue
        values[key] = save_values(by_save[key])
        if by_save[key]["trials"] != declaration["placed_starts"]["trials"]:
            unreadable.append(f"{key}: {by_save[key]['trials']} placed starts, declared "
                              f"{declaration['placed_starts']['trials']}")
        unreadable += [f"{key}: {name} unavailable" for name, value in values[key].items()
                       if value is None and name != "J_over_R"]
    cores = sorted({reading["readings_core"] for reading in by_save.values()})
    if len(cores) > 1:
        unreadable.append(f"the saves were read by more than one readings_core.py: {cores}")
    ends = {key: training_end(root, logs[key], declared["iterations"]) for key, root in trainings.items()}
    problems = [f"{condition} seed {seed}: training not given" for condition in declared["conditions"]
                for seed in declared["seeds"] if (condition, seed) not in trainings]
    problems += training_problems(declaration, configs)
    problems += [f"{condition} seed {seed}: the training did not complete its {declared['iterations']} iterations "
                 f"({end})" for (condition, seed), end in ends.items() if not end["completed"]]
    out = dict(saves={f"{c}/seed {s}/{i} iterations": values.get((c, s, i)) for c, s, i in wanted},
               trainings={f"{c} seed {s}": dict(run_dir=os.path.realpath(trainings[c, s]), end=ends[c, s])
                          for c, s in trainings},
               readings_core=cores, unreadable=unreadable + problems,
               exposure=None if problems else exposure(declaration, configs, logs))
    if out["unreadable"]:
        out["outcome"] = dict(outcome=None, note="unreadable by the declaration's unreadable_when; no row is taken")
    else:
        out["outcome"] = outcome(declaration, values)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply the recovery pair's outcome order to its saves' readings.")
    parser.add_argument("--pair", required=True, help="the declaration, studies/recovery_pair/pair.json")
    parser.add_argument("--training", action="append", required=True, metavar="CONDITION:SEED=RUN",
                        help="a blinkmimic2.train run directory of one condition and seed")
    parser.add_argument("--save", action="append", required=True, metavar="READINGS",
                        help="an output directory of readings.py")
    parser.add_argument("--out", required=True, help="output directory; it must not exist")
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error(f"the output directory {args.out} exists")
    declaration = read_declaration(args.pair)
    trainings = training_roots(declaration, args.training)
    configs = {key: json.load(open(os.path.join(root, "args.json"))) for key, root in trainings.items()}
    logs = {key: log_rows(root) for key, root in trainings.items()}
    readings = [json.load(open(os.path.join(directory, "save_readings.json"))) for directory in args.save]
    result = read_pair(declaration, readings, trainings, configs, logs)
    result["save_readings"] = [os.path.abspath(directory) for directory in args.save]
    text = json.dumps(result, indent=1, allow_nan=False)
    os.makedirs(args.out)
    with open(os.path.join(args.out, "outcome.json"), "x") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
