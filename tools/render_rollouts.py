"""Replay the records of tools/rollouts.py as videos (ROLLOUTS.md, "Videos of a record").

    python tools/render_rollouts.py rollouts/run_s1_6500 --trials 0 3 --out videos/run_s1_6500
    python tools/render_rollouts.py rollouts/run_s1_6500 rollouts/run_s2_6500 --labels s1 s2 --trials 0 --out videos/s1_s2

A video replays the recorded states: no physics, no policy and no second rollout. Each frame places the robot of the
simulator's model at a recorded state and the command's reference pose, at the command row after that control, as a
translucent second robot at its own world pose (--ghost_at world) or moved in the horizontal plane to the robot's root
(--ghost_at robot). A video ends --tail seconds after the trial ended or its command reached its clip's last row,
whichever comes first. Two record folders of one scene are shown side by side on one time axis. Rendering needs an
OpenGL backend for MuJoCo, chosen by the environment variable MUJOCO_GL (default here: egl).
"""

import argparse
import json
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RECORD_FILES = ("description.json", "rollouts.npz")
TRIAL_FIELDS = ("start_state", "command_row", "body_row")  # [T, ...], read whole
CONTROL_FIELDS = ("state", "command_row_after", "clip_last_row", "done", "available", "command_body_pos_w")  # [C, T, ...]
FIELDS = TRIAL_FIELDS + CONTROL_FIELDS
GHOST_RGB = (0.35, 0.65, 1.0)
GHOST_AT = ("world", "robot")  # where the command pose is drawn: its own world pose, or moved to the robot's root xy
COMMAND_BODY_TOLERANCE = 1e-3  # m, between the library's reference bodies and the record's command bodies
START_STATE_TOLERANCE = 1e-5  # largest difference of two records' start_state shown as one start
SHEET_FRAMES = 8


class Record:
    """One record folder of tools/rollouts.py: its description, the [T, ...] fields of TRIAL_FIELDS for every trial and
    the [C, T, ...] fields of CONTROL_FIELDS for the trials ``trials`` only (all when None), read by ``column``."""

    def __init__(self, folder: str, trials: list | None = None):
        self.folder = os.path.abspath(folder)
        for f in RECORD_FILES:
            if not os.path.isfile(os.path.join(folder, f)):
                raise ValueError(f"{folder} holds no {f}")
        with open(os.path.join(folder, "description.json")) as fh:
            d = json.load(fh)
        need = {"run_args": ("robot",), "library": ("motion_dir", "bones_list", "bones_lib", "motion_store", "rows"),
                "settings": ("control_dt",), "scene": (), "controls": ()}  # fmt: skip
        missing = [k for k in need if k not in d] + [f"{k}.{s}" for k, sub in need.items() if k in d for s in sub
                                                     if not isinstance(d[k], dict) or s not in d[k]]  # fmt: skip
        if missing:
            raise ValueError(f"{folder}/description.json lacks {missing}")
        self.description = d
        with np.load(os.path.join(folder, "rollouts.npz")) as z:
            missing = [k for k in FIELDS if k not in z.files]
            if missing:
                raise ValueError(f"{folder}/rollouts.npz lacks the fields {missing}")
            a = {k: z[k] for k in TRIAL_FIELDS}
            if a["start_state"].ndim != 2 or (a["start_state"].shape[1] - 13) % 2:
                raise ValueError(f"{folder}/rollouts.npz: start_state {a['start_state'].shape} is not [T, 13 + 2J]")
            T, D = a["start_state"].shape
            self.trials, self.joints = T, (D - 13) // 2
            wrong = [k for k in ("command_row", "body_row") if a[k].shape != (T,)]
            if wrong:
                raise ValueError(f"{folder}/rollouts.npz: the fields {wrong} do not match start_state {(T, D)}")
            self.chosen = list(range(T)) if trials is None else choose_trials(trials, T)
            C = None
            for k in CONTROL_FIELDS:  # state first; one field decompressed at a time, only the chosen trials kept
                x = z[k]
                C = x.shape[0] if C is None else C
                bodies = x.shape[2] if x.ndim == 4 else -1
                want = {"state": (C, T, D), "command_body_pos_w": (C, T, bodies, 3)}.get(k, (C, T))
                if x.shape != want:
                    raise ValueError(f"{folder}/rollouts.npz: the field {k} {x.shape} does not match start_state "
                                     f"{(T, D)} and state's {C} controls")  # fmt: skip
                a[k] = x[:, self.chosen]
        self.controls = C
        if C != d["controls"]:
            raise ValueError(f"{folder}: the record holds {C} controls, its description {d['controls']}")
        self.arrays, self.index = a, {t: i for i, t in enumerate(self.chosen)}
        self.robot = d["run_args"]["robot"]
        self.control_dt = float(d["settings"]["control_dt"])

    def __getitem__(self, key):
        """A field of TRIAL_FIELDS for every trial."""
        if key not in TRIAL_FIELDS:
            raise KeyError(f"{key} is read per trial with column")
        return self.arrays[key]

    def column(self, key: str, trial: int) -> np.ndarray:
        """A field of CONTROL_FIELDS for one trial: [C, ...]."""
        if trial not in self.index:
            raise ValueError(f"{self.folder}: trial {trial} is not among the trials read, {self.chosen}")
        return self.arrays[key][:, self.index[trial]]


def check_same_trials(a: Record, b: Record):
    """Refuse two records unless they hold the same trials: the same scene content, trial count, body and command rows,
    start states (within START_STATE_TOLERANCE), controls, robot, library rows and control period. Returns the largest
    difference of the start states."""
    if a.description["scene"] != b.description["scene"]:
        raise ValueError(f"{a.folder} and {b.folder} hold different scenes")
    if a.trials != b.trials:
        raise ValueError(f"{a.folder} holds {a.trials} trials, {b.folder} {b.trials}")
    for k in ("body_row", "command_row"):
        if not np.array_equal(a[k], b[k]):
            raise ValueError(f"{a.folder} and {b.folder} hold different {k}s")
    same = {"controls": (a.controls, b.controls), "robot": (a.robot, b.robot), "control_dt": (a.control_dt, b.control_dt),
            "library rows": (a.description["library"]["rows"], b.description["library"]["rows"])}  # fmt: skip
    for k, (x, y) in same.items():
        if x != y:
            raise ValueError(f"{a.folder} and {b.folder} hold different {k}: {x} and {y}")
    gap = float(np.abs(a["start_state"].astype(np.float64) - b["start_state"]).max())
    if not gap <= START_STATE_TOLERANCE:
        raise ValueError(f"the start states of {a.folder} and {b.folder} differ by up to {gap:.3g} "
                         f"(tolerance {START_STATE_TOLERANCE})")  # fmt: skip
    return gap


def choose_trials(trials: list, count: int) -> list:
    """The trial indices in the order given; an index outside [0, count) or given twice raises ValueError."""
    if not trials:
        raise ValueError("no trial is given")
    for t in trials:
        if not 0 <= t < count:
            raise ValueError(f"trial {t} is not in [0, {count})")
    if len(set(trials)) != len(trials):
        raise ValueError(f"the trials {trials} name a trial twice")
    return list(trials)


def frame_times(controls: int, stride: int) -> list:
    """The frames' times in controls from the start: 0 (the placed start), stride, 2 stride, ... up to ``controls``."""
    if stride < 1:
        raise ValueError(f"the stride {stride} is not at least 1")
    return list(range(0, controls + 1, stride))


def ended_at(rec: Record, trial: int) -> int | None:
    """The record index of the control that ended the trial, None if it ran to the last control."""
    done = np.flatnonzero(rec.column("done", trial))
    return int(done[0]) if len(done) else None


def clip_ended_at(rec: Record, trial: int) -> int | None:
    """The record index of the control at which the command reached its clip's last row and held, None if it never
    did: the first control from which command_row_after equals clip_last_row at every later control the trial runs
    (command_row_after >= 0; the rollout tool writes -1 after a trial's end), so the command row no longer advances.
    A command that starts at its clip's last row gives control 0; the control that ends a trial still counts."""
    after, last = rec.column("command_row_after", trial), rec.column("clip_last_row", trial)
    running = after >= 0
    held = np.flatnonzero(running & (after != last))  # running controls that are not at the clip's last row
    first = 0 if len(held) == 0 else int(held[-1]) + 1
    at_end = np.flatnonzero(running[first:])
    return first + int(at_end[0]) if len(at_end) else None


def end_point(rec: Record, trial: int) -> tuple:
    """(control, reason): the earlier of the control that ended the trial ("trial ended") and clip_ended_at ("clip
    ended"), "trial ended" when they are the same control; (None, "all controls") when there is neither."""
    end, clip = ended_at(rec, trial), clip_ended_at(rec, trial)
    if end is not None and (clip is None or end <= clip):
        return end, "trial ended"
    if clip is not None:
        return clip, "clip ended"
    return None, "all controls"


def tail_controls(tail_s: float | None, control_dt: float) -> int | None:
    """A tail in seconds as a whole number of controls (rounded); None (every recorded control) stays None."""
    return None if tail_s is None else int(round(tail_s / control_dt))


def video_stop(records: list, trial: int, tail: int | None) -> dict:
    """The last control a trial's video shows, for one record or two side by side (one time axis): ``tail`` controls
    after the later of the panels' end_point, at most the last recorded control. ``control`` is that control;
    ``reason`` is the reason of that end point, or "all controls" when ``tail`` is None, a panel has no end point or
    the tail runs past the record; ``end_points`` holds each panel's end_point."""
    last = records[0].controls - 1
    points = [end_point(r, trial) for r in records]
    out = {"end_points": [{"control": c, "reason": why} for c, why in points]}
    if tail is None or any(c is None for c, _ in points):
        return {"control": last, "reason": "all controls"} | out
    c, why = max(points, key=lambda p: p[0])
    if c + tail > last:
        return {"control": last, "reason": "all controls"} | out
    return {"control": c + tail, "reason": why} | out


def trial_frames(rec: Record, trial: int, stride: int, stop: int | None = None) -> list:
    """The frames of one trial at frame_times: per frame ``time`` (controls from the start), ``control`` (the record
    index of the control that ends at that time, -1 at the start), ``source`` (the record index of the state shown,
    -1 for start_state), ``row`` (the command row paired with that state: command_row_after of the source, the trial's
    command_row at the start) and ``ended_at`` (the control that ended the trial, from that control on; else None).

    ``stop`` (video_stop; None: the last recorded control) is the last control to show: the frames run to the first
    frame at or after it, so a stride above 1 shows the stop's control or one up to stride - 1 controls after it; when
    that frame would be past the record, they end at the last frame within it, which can come before the stop.

    The record holds no state after a trial's end, so from its end on the frame holds the last recorded state: the
    ending control's own state when it is available, else (a non-finite state) the state before it."""
    end = ended_at(rec, trial)
    available, after = rec.column("available", trial), rec.column("command_row_after", trial)
    last = np.maximum.accumulate(np.where(available, np.arange(rec.controls), -1))
    if stop is not None and not 0 <= stop < rec.controls:
        raise ValueError(f"the stop control {stop} is not in [0, {rec.controls})")
    times = rec.controls if stop is None else min(-(-(stop + 1) // max(stride, 1)) * stride, rec.controls)
    frames = []
    for k in frame_times(times, stride):
        c = k - 1
        src = -1 if c < 0 else int(last[c])
        ended = end is not None and c >= end
        if not ended and src != c:
            raise ValueError(f"{rec.folder}: trial {trial} has no recorded state after control {c} before its end")
        row = int(rec["command_row"][trial]) if src < 0 else int(after[src])
        if row < 0:
            raise ValueError(f"{rec.folder}: trial {trial} has no command row at control {src}")
        frames.append({"time": k, "control": c, "source": src, "row": row, "ended_at": end if ended else None})
    return frames


def body_qpos(rec: Record, trial: int, source: int) -> np.ndarray:
    """MuJoCo's free-joint and joint positions [7 + J] of a recorded state (state layout of ROLLOUTS.md: root position,
    root WXYZ quaternion, root velocities, joint positions, joint velocities); ``source`` -1 is start_state."""
    x = rec["start_state"][trial] if source < 0 else rec.column("state", trial)[source]
    q = np.concatenate((x[0:3], x[3:7], x[13 : 13 + rec.joints])).astype(np.float64)
    if not np.isfinite(q).all():
        raise ValueError(f"{rec.folder}: trial {trial} has no finite state at control {source}")
    return q


def command_qpos(motion, rows) -> np.ndarray:
    """MuJoCo's free-joint and joint positions [n, 7 + J] of the library's reference state at ``rows``."""
    import torch

    from blinkmimic2.rotations import mat_to_quat

    rows = torch.as_tensor(np.asarray(rows, dtype=np.int64))
    if bool(((rows < 0) | (rows >= motion.num_frames)).any()):
        raise ValueError(f"rows outside the library of {motion.num_frames} rows")
    p, R, _, _, q, _ = motion.ref_state(rows)
    return torch.cat((p, mat_to_quat(R), q), -1).double().numpy()


def library_inputs(rec: Record, motion_dir: str | None) -> dict:
    """The four library inputs of the record (as tools/rollouts.py passes them to MotionLib.load); ``motion_dir``
    replaces the record's folder of a library built from one."""
    lib = {k: rec.description["library"][k] for k in ("motion_dir", "bones_list", "bones_lib", "motion_store")}
    if motion_dir is not None:
        if lib["bones_list"]:
            raise ValueError(f"the record's library is the BONES-SEED list {lib['bones_list']}, not a motion folder")
        lib["motion_dir"] = motion_dir
    return lib


def load_library(robot, library: dict):
    """The motion library on the CPU, as tools/rollouts.py loads it on its device."""
    from blinkmimic2.kinematics import Kinematics
    from blinkmimic2.motion import MotionLib

    return MotionLib.load(robot, Kinematics(robot, "cpu"), "cpu", *library.values())


def check_library(motion, rec: Record, trials: list):
    """Refuse a library that is not the record's: another row count, or reference bodies at the recorded command rows
    farther than COMMAND_BODY_TOLERANCE from the record's command bodies (trials ``trials``)."""
    import torch

    rows = rec.description["library"]["rows"]
    if motion.num_frames != rows:
        raise ValueError(f"the motion library has {motion.num_frames} rows, the library of {rec.folder} {rows}")
    for t in trials:
        ok = rec.column("available", t)
        if not ok.any():
            continue
        after = torch.as_tensor(rec.column("command_row_after", t)[ok].astype(np.int64))
        gap = np.abs(motion.ref_bodies(after)[0].numpy() - rec.column("command_body_pos_w", t)[ok]).max()
        if not gap <= COMMAND_BODY_TOLERANCE:
            raise ValueError(f"the motion library's reference bodies differ from the command bodies of {rec.folder} "
                             f"trial {t} by up to {gap:.4f} m (tolerance {COMMAND_BODY_TOLERANCE} m)")  # fmt: skip


def frame_poses(rec: Record, trial: int, plan: list, motion, ghost_at: str) -> tuple:
    """MuJoCo poses [n, 7 + J] of a panel's frames ``plan`` (trial_frames): the robot at each frame's recorded state
    and the command's reference pose at its row, at its own world pose ("world") or with its root moved in the
    horizontal plane to the robot's root xy ("robot"; its height, orientation and joints unchanged)."""
    if ghost_at not in GHOST_AT:
        raise ValueError(f"the command pose is drawn at one of {GHOST_AT}, not {ghost_at!r}")
    body = np.stack([body_qpos(rec, trial, f["source"]) for f in plan])
    ghost = command_qpos(motion, [f["row"] for f in plan])
    if ghost_at == "robot":
        ghost[:, :2] = body[:, :2]
    return body, ghost


def build_model(robot):
    """The MuJoCo model of the rollout tool's simulator for one world: mjlab's scene of make_scene_cfg (the robot
    entity under "robot/", the plane terrain), compiled on the CPU. The replay uses its kinematics and geometry only."""
    from mjlab.scene import Scene

    from blinkmimic2.env import make_scene_cfg

    return Scene(make_scene_cfg(robot, 1), device="cpu").compile()


class Poser:
    """Places the robot of the model at free-joint and joint positions [7 + J] (joints in ``joint_names`` order) and
    runs MuJoCo's forward kinematics."""

    def __init__(self, model, joint_names):
        import mujoco

        self.mujoco, self.model = mujoco, model
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"the model has {len(free)} free joints, not one")
        self.free = int(model.jnt_qposadr[free[0]])
        self.joints = np.array([model.jnt_qposadr[model.joint(f"robot/{n}").id] for n in joint_names])
        if 7 + len(self.joints) != model.nq:
            raise ValueError(f"the free joint and {len(self.joints)} joints do not make the model's {model.nq} positions")

    def place(self, data, qpos: np.ndarray):
        if not np.isfinite(qpos).all():
            raise ValueError("a pose that is not finite")
        data.qpos[self.free : self.free + 7] = qpos[:7]
        data.qpos[self.joints] = qpos[7:]
        self.mujoco.mj_kinematics(self.model, data)


class GLUnavailable(RuntimeError):
    def __init__(self, what: str, e: Exception):
        gl = os.environ.get("MUJOCO_GL", "(unset)")
        super().__init__(f"MuJoCo could not {what} with MUJOCO_GL={gl} ({type(e).__name__}: {e}); set the environment "
                         "variable MUJOCO_GL to a backend this machine has: egl (a GPU or Mesa without a display), "
                         "osmesa, or glfw (a display)")  # fmt: skip


def import_mujoco():
    """mujoco, which loads the OpenGL backend that MUJOCO_GL selects when it is first imported."""
    try:
        import mujoco

        if not hasattr(mujoco, "Renderer"):  # mujoco leaves Renderer out when its backend does not load
            import mujoco.renderer  # noqa: F401  (raises the backend's error)

            raise ImportError("mujoco has no Renderer")
    except Exception as e:
        raise GLUnavailable("load its OpenGL backend", e) from e
    return mujoco


class PanelRenderer:
    """One panel: the robot of ``body`` solid and, when ghost_alpha > 0, the robot of ``ghost`` (MjData of the same
    model) drawn translucent blue; a camera at a fixed distance, azimuth and elevation looking at the body's root xy
    at the height lookat_z."""

    def __init__(self, model, width, height, distance, azimuth, elevation, lookat_z, ghost_alpha):
        import mujoco

        self.mujoco, self.model, self.alpha, self.lookat_z = mujoco, model, ghost_alpha, lookat_z
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        model.vis.headlight.ambient[:] = 0.45  # as tools/render_gif.py
        model.vis.headlight.diffuse[:] = 0.55
        try:
            self.renderer = mujoco.Renderer(model, height, width)
        except Exception as e:
            raise GLUnavailable("open an OpenGL context", e) from e
        self.option, self.perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = distance, azimuth, elevation

    def render(self, body, ghost, root_xy) -> np.ndarray:
        mj = self.mujoco
        self.camera.lookat[:] = (root_xy[0], root_xy[1], self.lookat_z)
        self.renderer.update_scene(body, camera=self.camera, scene_option=self.option)
        if self.alpha > 0:
            scene = self.renderer.scene
            n0 = scene.ngeom
            mj.mjv_addGeoms(self.model, ghost, self.option, self.perturb, int(mj.mjtCatBit.mjCAT_DYNAMIC), scene)
            if scene.ngeom == n0:
                raise RuntimeError("the command pose added no geoms to the scene")
            for i in range(n0, scene.ngeom):
                g = scene.geoms[i]
                g.rgba[:] = (*GHOST_RGB, self.alpha)
                g.matid = -1
                g.transparent = 1
        return self.renderer.render()


def ffmpeg_with_libx264() -> str:
    """The ffmpeg executable (PATH, else imageio-ffmpeg's), refused unless it has the libx264 encoder."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            raise ValueError("no ffmpeg on PATH and no imageio-ffmpeg") from None
    out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True)
    if "libx264" not in out.stdout:
        raise ValueError(f"{exe} has no libx264 encoder")
    return exe


class Encoder:
    """H.264 (libx264, yuv420p) MP4 from RGB frames, through an ffmpeg pipe (the arguments of tools/render_gif.py)."""

    def __init__(self, exe: str, path: str, width: int, height: int, fps: str):
        cmd = [exe, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", fps,
               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]  # fmt: skip
        self.path, self.p = path, subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        err = self.p.stderr.read().decode(errors="replace")
        if self.p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {self.path}: {err}")


def control_rate_fps(control_dt: float) -> str:
    """The control rate as a frame rate for ffmpeg's -r."""
    rate = 1.0 / control_dt
    return str(round(rate)) if abs(rate - round(rate)) < 1e-9 else f"{rate:g}"


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def panel_text(label: str, trial: int, clips: tuple, frame: dict, control_dt: float, ghost_at: str = "world") -> list:
    """The panel's lines: label and trial; the body's and the command's clips (and, under ghost_at "robot", that the
    command is drawn at the robot's position); the time and the end."""
    when = "start" if frame["control"] < 0 else f"after control {frame['control']}"
    line = f"{when} | {frame['time'] * control_dt:.2f} s"
    if frame["ended_at"] is not None:
        line += f" | ended at control {frame['ended_at']}"
    clip_line = f"body {clips[0]} | command {clips[1]}"
    if ghost_at == "robot":
        clip_line += ", drawn at the robot's position"
    return [f"{label} | trial {trial}", clip_line, line]


def draw_lines(image: Image.Image, lines: list, f, xy=(6, 4)):
    d = ImageDraw.Draw(image, "RGBA")
    h = f.size + 3 if hasattr(f, "size") else 13
    d.rectangle((0, 0, image.width, xy[1] + h * len(lines) + 2), fill=(255, 255, 255, 170))
    for i, s in enumerate(lines):
        d.text((xy[0], xy[1] + i * h), s, fill=(40, 40, 50), font=f)


def contact_sheet(frames: list, columns: int = 4) -> Image.Image:
    """The frames at half size in rows of ``columns``."""
    w, h = frames[0].width // 2, frames[0].height // 2
    sheet = Image.new("RGB", (w * min(columns, len(frames)), h * -(-len(frames) // columns)), "white")
    for i, im in enumerate(frames):
        sheet.paste(im.resize((w, h)), ((i % columns) * w, (i // columns) * h))
    return sheet


def render_trial(out, trial, records, labels, plans, stop, motion, poser, renderer, exe, fps, args, fonts, start_gap):
    """One trial's MP4, contact sheet (PNG) and description (JSON) in ``out``; ``plans``: each panel's trial_frames up to
    ``stop`` (video_stop); ``start_gap``: the largest difference of two records' start states (check_same_trials), None
    for one record."""
    import mujoco

    model = poser.model
    body, ghost = mujoco.MjData(model), mujoco.MjData(model)
    n = len(plans[0])
    clip = lambda row: motion.names[int(motion.clip_id[int(row)])]  # noqa: E731
    clips = (clip(records[0]["body_row"][trial]), clip(records[0]["command_row"][trial]))
    W, H = args.width, args.height
    header_font, text_font = fonts
    top = 2 * ((getattr(header_font, "size", 12) + 9) // 2) if len(records) == 2 else 0  # even, for yuv420p
    name = f"trial_{trial:03d}"
    enc = Encoder(exe, os.path.join(out, name + ".mp4"), W * len(records), H + top, fps)
    picks = set(np.linspace(0, n - 1, SHEET_FRAMES).round().astype(int).tolist())
    sheet = []
    poses = [frame_poses(rec, trial, plan, motion, args.ghost_at) for rec, plan in zip(records, plans)]
    noted = args.ghost_at if args.ghost_alpha > 0 else "world"  # the text names where a drawn command pose is
    try:
        for k in range(n):
            canvas = Image.new("RGB", (W * len(records), H + top), "white")
            for i, (rec, plan) in enumerate(zip(records, plans)):
                f, q = plan[k], poses[i][0][k]
                poser.place(body, q)
                poser.place(ghost, poses[i][1][k])
                im = Image.fromarray(renderer.render(body, ghost, q[:2]))
                draw_lines(im, panel_text(labels[i], trial, clips, f, rec.control_dt, noted), text_font)
                canvas.paste(im, (i * W, top))
            if top:
                ImageDraw.Draw(canvas).text((6, 4), f"two runs from one start: trial {trial}, the same placed state and "
                                            "command rows", fill=(40, 40, 50), font=header_font)  # fmt: skip
            enc.write(np.asarray(canvas))
            if k in picks:
                sheet.append(canvas)
    finally:
        enc.close()
    contact_sheet(sheet).save(os.path.join(out, name + ".png"))
    description = {
        "records": [r.folder for r in records], "labels": labels, "trial": trial,
        "body_row": int(records[0]["body_row"][trial]), "command_row": int(records[0]["command_row"][trial]),
        "clips": {"body": clips[0], "command": clips[1]},
        "frames": n, "stride": args.stride, "fps": float(fps), "control_dt": records[0].control_dt,
        "playback_speed": float(fps) * args.stride * records[0].control_dt,  # 1: trial time
        "ended_at": [ended_at(r, trial) for r in records], "start_state_gap": start_gap,
        "stop": stop | {"last_frame_control": plans[0][-1]["control"]},
        "options": {"width": W, "height": H, "distance": args.distance, "azimuth": args.azimuth,
                    "elevation": args.elevation, "lookat_z": args.lookat_z, "ghost_alpha": args.ghost_alpha,
                    "ghost_at": args.ghost_at, "tail_s": args.tail},
        "video": name + ".mp4", "contact_sheet": name + ".png",
    }  # fmt: skip
    with open(os.path.join(out, name + ".json"), "w") as fh:
        json.dump(description, fh, indent=1)
    return description


def tail_seconds(text: str) -> float | None:
    """--tail: seconds, or None (every recorded control) for "all" or a negative value."""
    if text == "all":
        return None
    try:
        tail = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds or 'all'") from None
    if not math.isfinite(tail):
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number of seconds")
    return None if tail < 0 else tail


def main(argv=None):
    p = argparse.ArgumentParser(description="Replay the recorded states of tools/rollouts.py records as videos.")
    p.add_argument("records", nargs="+", help="one record folder, or two of the same trials to show side by side")
    p.add_argument("--trials", type=int, nargs="+", required=True, help="trial indices, rendered in this order")
    p.add_argument("--out", required=True, help="output folder; it must not exist")
    p.add_argument("--labels", nargs="+", help="one label per record (default: the folder names)")
    p.add_argument("--width", type=int, default=480, help="panel width (even)")
    p.add_argument("--height", type=int, default=360, help="panel height (even)")
    p.add_argument("--stride", type=int, default=1, help="controls per frame")
    p.add_argument("--fps", type=float, help="frames per second (default: the control rate; a stride above 1 then plays "
                                             "faster than trial time by the stride)")  # fmt: skip
    p.add_argument("--motion_dir", help="motion folder of the library, in place of the record's (same rows)")
    p.add_argument("--distance", type=float, default=3.0, help="camera distance (m)")
    p.add_argument("--azimuth", type=float, default=135.0, help="camera azimuth (degrees, world)")
    p.add_argument("--elevation", type=float, default=-15.0, help="camera elevation (degrees)")
    p.add_argument("--lookat_z", type=float, default=0.6, help="height the camera looks at (m)")
    p.add_argument("--ghost_alpha", type=float, default=0.3, help="opacity of the command pose; 0 leaves it out")
    p.add_argument("--ghost_at", choices=GHOST_AT, default="world", help="the command pose at its own world pose, or "
                   "moved in the horizontal plane to the robot's root at every frame")  # fmt: skip
    p.add_argument("--tail", type=tail_seconds, default=2.0, help="seconds shown after the trial ended or its command "
                   "reached its clip's last row, whichever is first; a negative value or 'all' shows every recorded "
                   "control")  # fmt: skip
    a = p.parse_args(argv)
    if os.path.exists(a.out):
        p.error(f"the output folder {a.out} exists")
    if len(a.records) > 2:
        p.error("one or two record folders")
    labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.records]
    if len(labels) != len(a.records):
        p.error(f"{len(labels)} labels for {len(a.records)} records")
    if a.width < 16 or a.height < 16 or a.width % 2 or a.height % 2:
        p.error("--width and --height must be even and at least 16 (H.264 in yuv420p)")
    if not 0 <= a.ghost_alpha <= 1:
        p.error("--ghost_alpha must be in [0, 1]")
    if a.fps is not None and not a.fps > 0:
        p.error("--fps must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")  # before the first import of mujoco
    try:
        import_mujoco()
        records = [Record(r, a.trials) for r in a.records]
        start_gap = check_same_trials(*records) if len(records) == 2 else None
        trials = records[0].chosen
        tail = tail_controls(a.tail, records[0].control_dt)
        stops = {t: video_stop(records, t, tail) for t in trials}
        plans = {t: [trial_frames(r, t, a.stride, stops[t]["control"]) for r in records] for t in trials}
        library = library_inputs(records[0], a.motion_dir)
        from blinkmimic2.robot import get_robot

        robot = get_robot(records[0].robot)
        if robot.num_joints != records[0].joints:
            raise ValueError(f"the robot {robot.name} has {robot.num_joints} joints, the record {records[0].joints}")
        motion = load_library(robot, library)
        for r in records:
            check_library(motion, r, trials)
        exe = ffmpeg_with_libx264()
        model = build_model(robot)
        poser = Poser(model, robot.joint_names)
        renderer = PanelRenderer(model, a.width, a.height, a.distance, a.azimuth, a.elevation, a.lookat_z, a.ghost_alpha)
    except (ValueError, FileNotFoundError, GLUnavailable) as e:
        p.error(str(e))
    fps = f"{a.fps:g}" if a.fps is not None else control_rate_fps(records[0].control_dt)
    fonts = font(max(10, a.height // 26)), font(max(10, a.height // 28))
    os.makedirs(a.out)
    for t in trials:
        d = render_trial(a.out, t, records, labels, plans[t], stops[t], motion, poser, renderer, exe, fps, a, fonts,
                         start_gap)  # fmt: skip
        print(f"trial {t}: {d['frames']} frames to control {d['stop']['control']} ({d['stop']['reason']}), ended at "
              f"{d['ended_at']}; wrote {os.path.join(a.out, d['video'])}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
