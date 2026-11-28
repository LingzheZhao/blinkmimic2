"""Clip-group readings of saves of the recovery pair and its follow-ups: tracking over the commanded and the excluded
clips, and trial ends per get-up of the reference, from readings.py's save_readings.json and the first-row records of
tools/rollouts.py. This program computes the readings that README.md describes in Results under "Commanded and
excluded clips" and "Ends on get-up clips".

    python studies/recovery_pair/groups.py --scene SCENE --motion_dir DIR \\
        --set READINGS FIRST_ROWS [--set READINGS FIRST_ROWS ...] --out OUT.json

SCENE is the output of scene.py; each --set is one rollout set of one save: the output directory of readings.py
(save_readings.json) and the tools/rollouts.py directory of its first-row rollouts. DIR holds the 40 LAFAN1 G1 CSVs
of the library; the reference's pelvis rows are taken from it (``reference_rows``) and compared with the clips of
scene.json and the command bodies the records hold. OUT must not exist.

Definitions (those of README.md, Results, "Ends on get-up clips"):

- The **commanded** clips are those outside scene.json's ``command_excluded_clips``; the **excluded** clips are
  those clips (the five hidden clips and fallAndGetUp1_subject4). E over a group is the mean of its clips' aligned
  body-position errors (``aligned_body_error_m`` of save_readings.json, each read up to the clip's first
  termination); completion counts the group's clips with ``completed``.
- A reference row is **low** when its pelvis is below 0.50 m or the world z of its pelvis up axis is below 0.5. A
  **low span** is a maximal run of low rows inside a clip of at least 25 rows. A **get-up window** is rows
  [e - 50, e + 50] around a span's last row e; in the controls of a first-row trial (control k targets the clip's
  first row + k + 1) it is controls [max(e - 50 - first - 1, 0), min(e + 50 - first - 1, last - first - 1)].
- A trial **enters** a window unless it ended before the window's first control; it **ends in** the window when its
  end control (the first control with ``done``) lies inside it. Overlapping windows count separately.
- Groups of the ends: the hidden clips (``hidden_clips``), fallAndGetUp1_subject4 (the excluded clips that are not
  hidden) and the commanded clips with at least one window.
"""

import argparse
import json
import os

import numpy as np

LOW_HEIGHT, LOW_UP_Z = 0.50, 0.5
MIN_SPAN_ROWS, HALF_WINDOW_ROWS = 25, 50


def low_rows(height, up_z, low_height=LOW_HEIGHT, low_up_z=LOW_UP_Z):
    """Rows whose pelvis is below ``low_height`` or whose up-axis z is below ``low_up_z``."""
    return (np.asarray(height) < low_height) | (np.asarray(up_z) < low_up_z)


def low_spans(low, min_rows=MIN_SPAN_ROWS):
    """(first, last) indices of the maximal runs of True in ``low`` of at least ``min_rows`` entries."""
    low = np.asarray(low, dtype=bool)
    edges = np.diff(np.concatenate(([0], low.astype(np.int8), [0])))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_rows]


def get_up_windows(low_clip, first, last, half=HALF_WINDOW_ROWS, min_rows=MIN_SPAN_ROWS):
    """The get-up windows of one clip of rows ``first``..``last`` (``low_clip`` its low flags): per low span, its first
    and last row and the window's first and last control of a trial started at the clip's first row."""
    out = []
    for s, e in low_spans(low_clip, min_rows):
        span_first, span_last = first + s, first + e
        c0 = max(span_last - half - first - 1, 0)
        c1 = min(span_last + half - first - 1, last - first - 1)
        out.append(dict(span_first_row=span_first, span_last_row=span_last, window_controls=[c0, c1]))
    return out


def window_ends(end, windows):
    """(windows entered, windows ended in) by a trial whose first ``done`` is control ``end`` (None: no end)."""
    entered = ended = 0
    for w in windows:
        c0, c1 = w["window_controls"]
        if end is None or end >= c0:
            entered += 1
            ended += int(end is not None and c0 <= end <= c1)
    return entered, ended


def first_end(done):
    """The first control with ``done`` of one trial's column, None without one."""
    hits = np.nonzero(np.asarray(done))[0]
    return int(hits[0]) if hits.size else None


def clip_group(clips, names):
    """E (mean of the clips' errors; None if a clip has none) and completion over the clips ``names`` of the
    first-row readings ``clips`` (save_readings.json first_rows.clips)."""
    by = {c["clip"]: c for c in clips}
    errors = [by[n]["aligned_body_error_m"] for n in names]
    e = None if any(x is None for x in errors) else float(np.mean(errors))
    return dict(clips=len(names), clips_completed=int(sum(bool(by[n]["completed"]) for n in names)), tracking_error_m=e)


def read_set(scene, reference, readings, done, command_rows):
    """The readings of one rollout set. ``scene`` is scene.json; ``reference`` (height, up_z) per library row;
    ``readings`` save_readings.json; ``done`` [C, T] and ``command_rows`` [T] of the first-row record."""
    clips = scene["clips"]
    names = [c["clip"] for c in clips]
    excluded = list(scene["settings"]["command_excluded_clips"])
    hidden = list(scene["settings"]["hidden_clips"])
    commanded = [n for n in names if n not in excluded]
    if [int(r) for r in command_rows] != [c["first_row"] for c in clips]:
        raise ValueError("the first-row record's command rows are not the first rows of scene.json's clips")
    rc = readings["first_rows"]["clips"]
    if [c["clip"] for c in rc] != names:
        raise ValueError("save_readings.json's first-row clips are not scene.json's")
    low = low_rows(*reference)
    windows = {c["clip"]: get_up_windows(low[c["first_row"]:c["last_row"] + 1], c["first_row"], c["last_row"])
               for c in clips}
    group_of = {n: ("hidden_clips" if n in hidden else "excluded_clips_not_hidden" if n in excluded
                    else "commanded_clips_with_get_ups" if windows[n] else None) for n in names}
    ends = {g: dict(ended_in_window=0, windows_entered=0) for g in ("hidden_clips", "excluded_clips_not_hidden",
                                                                    "commanded_clips_with_get_ups")}
    trials = []
    for t, c in enumerate(clips):
        end = first_end(done[:, t])
        entered, ended = window_ends(end, windows[c["clip"]])
        g = group_of[c["clip"]]
        if g is not None:
            ends[g]["windows_entered"] += entered
            ends[g]["ended_in_window"] += ended
        trials.append(dict(clip=c["clip"], group=g, end_control=end, windows=len(windows[c["clip"]]),
                           windows_entered=entered, ended_in_window=ended))
    q = readings
    return dict(save=q["save"], R=q["R"]["numerator"], J=q["J"]["numerator"], F=q["F"]["numerator"],
                C=q["first_rows"]["C"]["completed"], E_m=q["first_rows"]["E_m"]["value"],
                commanded_clips=clip_group(rc, commanded), excluded_clips=clip_group(rc, excluded),
                excluded_clips_completed_names=[n for n in excluded if next(x for x in rc if x["clip"] == n)["completed"]],
                ends_per_reference_get_up_window=ends, trials=trials), windows


def pooled(sets):
    """The ends per window summed over the sets (rollouts of one save; not independent)."""
    out = {}
    for s in sets:
        for g, v in s["ends_per_reference_get_up_window"].items():
            o = out.setdefault(g, dict(ended_in_window=0, windows_entered=0))
            o["ended_in_window"] += v["ended_in_window"]
            o["windows_entered"] += v["windows_entered"]
    return out


def reference_rows(motion_dir, robot_name):
    """Pelvis height and up-axis z (root_R[2, 2]) of every library row built from ``motion_dir`` on the CPU as the
    trainings build it (fp32 store), with the clip names, first and last rows."""
    from blinkmimic2.kinematics import Kinematics
    from blinkmimic2.motion import MotionLib
    from blinkmimic2.robot import get_robot

    robot = get_robot(robot_name)
    m = MotionLib.load(robot, Kinematics(robot, "cpu"), "cpu", motion_dir, "", None, "fp32")
    rows = np.arange(m.num_frames)
    import torch

    with torch.no_grad():
        idx = torch.as_tensor(rows)
        h = m.root_p[idx][:, 2].numpy().astype(np.float64)
        g = m.root_gz[idx].numpy().astype(np.float64)
    return (h, g), list(m.names), m.clip_start.tolist(), m.clip_end.tolist()


def compare_reference(reference, record, pelvis):
    """Largest differences between the reference rows and a first-row record's command pelvis (height, up-axis z
    from its WXYZ quaternion) over its available controls, and the controls whose low flag differs."""
    av = record["available"]
    rows = record["command_row_after"][av]
    h = record["command_body_pos_w"][:, :, pelvis, 2][av].astype(np.float64)
    q = record["command_body_quat_w"][:, :, pelvis, :][av].astype(np.float64)
    g = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    rh, rg = reference[0][rows], reference[1][rows]
    return dict(height_max_abs_m=float(np.abs(h - rh).max()), up_z_max_abs=float(np.abs(g - rg).max()),
                low_flag_mismatches=int((low_rows(h, g) != low_rows(rh, rg)).sum()), controls=int(av.sum()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clip-group readings and ends per get-up window of saves.")
    parser.add_argument("--scene", required=True, help="output directory of scene.py")
    parser.add_argument("--motion_dir", required=True, help="the 40 LAFAN1 G1 CSVs of the library")
    parser.add_argument("--set", nargs=2, action="append", required=True, metavar=("READINGS", "FIRST_ROWS"),
                        help="readings.py output and first-row rollouts of one rollout set")
    parser.add_argument("--out", required=True, help="output JSON file; it must not exist")
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error(f"{args.out} exists")
    scene = json.load(open(os.path.join(args.scene, "scene.json")))
    reference, names, firsts, lasts = reference_rows(args.motion_dir, scene["robot"])
    clips = scene["clips"]
    if (names, firsts, lasts) != ([c["clip"] for c in clips], [c["first_row"] for c in clips],
                                  [c["last_row"] for c in clips]):
        raise ValueError(f"the library of {args.motion_dir} is not the one of scene.json")
    sets, comparisons, windows = [], [], None
    keys = ("done", "command_row", "available", "command_row_after", "command_body_pos_w", "command_body_quat_w")
    for readings_dir, first_rows in args.set:
        readings = json.load(open(os.path.join(readings_dir, "save_readings.json")))
        description = json.load(open(os.path.join(first_rows, "description.json")))
        if os.path.realpath(readings["rollouts"]["first_rows"]["directory"]) != os.path.realpath(first_rows):
            raise ValueError(f"{readings_dir} was not read from {first_rows}")
        with np.load(os.path.join(first_rows, "rollouts.npz")) as z:
            record = {k: z[k] for k in keys}
        result, windows = read_set(scene, reference, readings, record["done"], record["command_row"])
        result["first_rows"] = os.path.abspath(first_rows)
        sets.append(result)
        comparisons.append(compare_reference(reference, record, description["bodies"]["pelvis"]))
    out = dict(definitions=__doc__.split("Definitions")[1].strip(), scene=os.path.abspath(args.scene),
               motion_dir=os.path.abspath(args.motion_dir), reference_comparisons=comparisons,
               windows={n: w for n, w in windows.items() if w}, sets=sets,
               ends_per_reference_get_up_window_summed_over_sets=pooled(sets))
    with open(args.out, "x") as f:
        f.write(json.dumps(out, indent=1, allow_nan=False) + "\n")
    for s in sets:
        print(json.dumps(dict(checkpoint=s["save"]["checkpoint"], R=s["R"], C=s["C"],
                              tracking_error_over_commanded_clips_m=s["commanded_clips"]["tracking_error_m"],
                              commanded_clips_completed=s["commanded_clips"]["clips_completed"],
                              tracking_error_over_excluded_clips_m=s["excluded_clips"]["tracking_error_m"],
                              excluded_clips_completed=s["excluded_clips"]["clips_completed"],
                              ends_per_reference_get_up_window=s["ends_per_reference_get_up_window"])))
    print(json.dumps(dict(ends_per_reference_get_up_window_summed_over_sets=pooled(sets), reference_comparisons=comparisons)))


if __name__ == "__main__":
    main()
