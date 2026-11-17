"""Recovery training from placed lying starts against a hidden get-up reference (RECOVERY.md), in a reduced,
single-network form of StableMimic (arXiv 2608.02385v1).

A share of the resets places the body at a low row of a get-up recording while the command, the reference row the
policy reads, starts at a separate upright row. The reward and the termination of such an episode then follow the
recording (the hidden row) until the body is up, and a blend leads the reward back to the command. ``advance`` is the
rule of one control; the simulator step (env.TrackingSim.step_core) and the imagined step (train.policy_update) both
call it. The policy's inputs and its action map read the command row only.

A control's regime is that of the state it starts from: RECOVERY while the hidden row is followed, BLEND while controls
of the blend back to the command remain, TRACKING otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .mdp import MAX_STEP_REWARD
from .motion import CONTROL_DT

TRACKING, RECOVERY, BLEND = 0, 1, 2
OPTIONS = ("recovery_share", "recovery_hidden_clips", "recovery_command_excluded_clips", "recovery_low_pelvis",
           "recovery_rows", "recovery_reward_scale", "recovery_seconds")
# The regime of a state, one entry per world: the row its last control targeted (the hidden row while recovering, the
# command row otherwise; at a reset the body row), whether the hidden row is followed, the consecutive controls of
# target error, the blend controls still to run and the planar offset of the command reference set at the switch.
KEYS = ("hidden_frame", "recovering", "error_controls", "blend_left", "command_offset")


@dataclass(frozen=True)
class Settings:
    """The recovery setting (train.py's --recovery_* options), durations in controls."""
    share: float
    hidden_clips: tuple
    command_excluded_clips: tuple
    low_pelvis_height: float
    low_pelvis_up_z: float
    upright_run_rows: int
    longest_span_rows: int
    command_upright_rows: int
    reward_scale: float
    stop_controls: int
    blend_controls: int

    def __post_init__(self):
        if not 0 < self.share <= 1:
            raise ValueError(f"recovery share {self.share} must lie in (0, 1]")
        if not self.hidden_clips:
            raise ValueError("recovery needs at least one hidden clip")
        outside = sorted(set(self.hidden_clips) - set(self.command_excluded_clips))
        if outside:
            # a reset is a recovery start where its body row differs from its command row, which needs the two apart
            raise ValueError(f"hidden clips {outside} must also be command-excluded clips")
        if not (math.isfinite(self.low_pelvis_height) and math.isfinite(self.low_pelvis_up_z)):
            raise ValueError(f"recovery low-pelvis bounds {self.low_pelvis_height}, {self.low_pelvis_up_z} must be finite")
        for name in ("upright_run_rows", "longest_span_rows", "command_upright_rows", "stop_controls", "blend_controls"):
            if getattr(self, name) <= 0:
                raise ValueError(f"recovery {name} {getattr(self, name)} must be positive")
        if not (math.isfinite(self.reward_scale) and self.reward_scale > 0):
            raise ValueError(f"recovery reward scale {self.reward_scale} must be positive")


def settings_from_args(args) -> Settings | None:
    """The Settings of parsed train.py arguments, or None when no recovery option is given. A partial set of the options
    is refused, and so is a duration that is not a whole number of controls."""
    given = [name for name in OPTIONS if getattr(args, name, None) is not None]
    if not given:
        return None
    missing = [name for name in OPTIONS if name not in given]
    if missing:
        raise ValueError(f"the recovery options go together; missing {['--' + n for n in missing]}")
    controls = []
    for seconds in args.recovery_seconds:
        n = seconds / CONTROL_DT
        if not abs(n - round(n)) <= 1e-9:
            raise ValueError(f"--recovery_seconds {seconds} is not a whole number of {CONTROL_DT} s controls")
        controls.append(round(n))
    height, up_z = args.recovery_low_pelvis
    upright, span, command = args.recovery_rows
    return Settings(share=args.recovery_share, hidden_clips=tuple(args.recovery_hidden_clips),
                    command_excluded_clips=tuple(args.recovery_command_excluded_clips), low_pelvis_height=height,
                    low_pelvis_up_z=up_z, upright_run_rows=upright, longest_span_rows=span, command_upright_rows=command,
                    reward_scale=args.recovery_reward_scale, stop_controls=controls[0], blend_controls=controls[1])


@dataclass(frozen=True)
class Recovery:
    """The settings and the row tables of a motion library: the get-up end of every row (-1 where none), the start
    rows and the command rows (int64, on the library's device), and the six counts behind them."""
    settings: Settings
    get_up_end: torch.Tensor
    start_rows: torch.Tensor
    command_rows: torch.Tensor
    counts: dict


def upright_run_lengths(upright, first):
    """For every row, the number of consecutive upright rows that end at it inside its clip; ``first`` is the first
    row of each row's clip."""
    index = torch.arange(upright.shape[0], device=upright.device)
    breaks = torch.where(upright, torch.full_like(index, -1), index)
    start = torch.maximum(torch.cummax(breaks, 0).values + 1, first)
    return torch.where(upright, index - start + 1, torch.zeros_like(index))


def get_up_last_rows(upright, first, last, run_rows):
    """The last row of the first run of ``run_rows`` upright rows ending at or after each row inside its clip, -1 where
    none."""
    index = torch.arange(upright.shape[0], device=upright.device)
    ends = upright_run_lengths(upright, first) >= run_rows
    candidates = torch.where(ends, index, torch.full_like(index, upright.shape[0]))
    following = torch.flip(torch.cummin(torch.flip(candidates, (0,)), 0).values, (0,))
    return torch.where(following <= last, following, torch.full_like(index, -1))


def upright_ahead(upright, first, last, rows_ahead):
    """Rows that are upright with the next ``rows_ahead`` rows, all before their clip's last row."""
    index = torch.arange(upright.shape[0], device=upright.device)
    end = index + rows_ahead
    lengths = upright_run_lengths(upright, first)
    return (end < last) & (lengths[torch.clamp_max(end, upright.shape[0] - 1)] >= rows_ahead + 1)


def clip_mask(names, clip_of_row, wanted, what):
    """Rows of the clips ``wanted`` (names of ``names``); unknown names are refused."""
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise ValueError(f"{what} {unknown} are not clips of the motion library")
    selected = torch.zeros(len(names), dtype=torch.bool, device=clip_of_row.device)
    selected[[names.index(name) for name in wanted]] = True
    return selected[clip_of_row]


def build(motion, settings: Settings) -> Recovery:
    """The row tables of ``motion`` (MotionLib or CompactMotionLib), as RECOVERY.md (Rows of the motion library)
    defines them. Unknown clips and an empty start or command table are refused."""
    s = settings
    rows = torch.arange(motion.num_frames, device=motion.clip_id.device)
    clip = motion.clip_id
    first, last = motion.clip_start[clip], motion.clip_end[clip]
    # pelvis height, and the world z of the pelvis up axis (R[2, 2])
    low = (motion.root_p[rows][:, 2] < s.low_pelvis_height) | (motion.root_gz[rows] < s.low_pelvis_up_z)
    end = get_up_last_rows(~low, first, last, s.upright_run_rows)
    low_hidden = clip_mask(motion.names, clip, s.hidden_clips, "hidden clips") & low
    ends_in_clip = low_hidden & (end >= 0)
    starting = ends_in_clip & (end - rows <= s.longest_span_rows)
    excluded = clip_mask(motion.names, clip, s.command_excluded_clips, "command-excluded clips")
    commanding = ~excluded & upright_ahead(~low, first, last, s.command_upright_rows)
    start_rows, command_rows = rows[starting], rows[commanding]
    if start_rows.numel() == 0 or command_rows.numel() == 0:
        raise ValueError(f"recovery needs start rows and command rows; the motion library gives {start_rows.numel()} "
                         f"and {command_rows.numel()}")
    counts = dict(low_rows_of_hidden_clips=int(low_hidden.sum()),
                  without_a_get_up_end_in_the_clip=int((low_hidden & (end < 0)).sum()),
                  beyond_the_longest_span=int((ends_in_clip & ~starting).sum()), start_rows=start_rows.numel(),
                  get_ups=torch.unique(end[starting]).numel(), command_rows=command_rows.numel())
    return Recovery(s, end, start_rows, command_rows, counts)


def draw(rec: Recovery, frames, bins, frame_bin):
    """The recovery part of the reset draws of n worlds whose reset rows ``frames`` (bins ``bins``) are drawn. Draws,
    in this order, n uniforms (below the share: a recovery start), n uniforms for a start row and n for a command row,
    each uniform over its table. Returns the command rows, their bins and the body rows: a recovery start gets a
    command row and a start row, any other reset its reset row for both."""
    n, dev = frames.shape[0], frames.device
    pick = lambda table: table[(torch.rand(n, device=dev) * len(table)).long().clamp(max=len(table) - 1)]  # noqa: E731
    recovering = torch.rand(n, device=dev) < rec.settings.share
    start, command = pick(rec.start_rows), pick(rec.command_rows)
    return (torch.where(recovering, command, frames), torch.where(recovering, frame_bin[command], bins),
            torch.where(recovering, start, frames))


def start_state(body, frames) -> dict:
    """The regime of states placed at the rows ``body`` under the command rows ``frames``: recovering where the two
    differ, which happens only for recovery starts (the start and command tables are apart)."""
    zero = torch.zeros_like(frames)
    return {"hidden_frame": body.clone(), "recovering": body != frames, "error_controls": zero,
            "blend_left": zero.clone(), "command_offset": torch.zeros(frames.shape[0], 2, device=frames.device)}


def regime(recovering, blend_left):
    """The regime (TRACKING, RECOVERY or BLEND) of controls that start from states with these flags and blend counts."""
    return torch.where(recovering, RECOVERY, torch.where(blend_left > 0, BLEND, TRACKING))


def advance(mdp, rec: Recovery, S1, f1, state: dict):
    """One control's regime rule. ``state`` is the regime of the state the control starts from, ``S1`` the state it
    reaches and ``f1`` the command row after it. In this order:

    - the target row is min(hidden + 1, get-up end of hidden) for a recovering state (the hidden row advances and holds
      at its get-up end), the command row otherwise;
    - a recovering control is one of target error when the body is outside the termination thresholds of
      mdp.termination at the target row, or, at the get-up end, at the command row; a non-finite pelvis height or up
      axis is outside every row's thresholds; an error adds one to the count of consecutive errors, a control without
      error sets it to 0;
    - the switch: a recovering control at the get-up end with the body within both rows' thresholds stops recovering,
      starts the blend (BLEND controls, lowered by one per later control) and sets the command offset to the pelvis's
      planar position minus the command row's (without gradient; the command's yaw is kept);
    - the stop: a control still recovering after STOP or more consecutive errors;
    - the reward scale, from the regime the control starts in: the recovery reward scale, (BLEND - left + 1) / BLEND
      with ``left`` blend controls still to run, or 1.

    Returns the regime of the reached state, the target row (the row the reward reads), the reward scale and the stop
    and switch flags. Branch-free and without host synchronization (captured in CUDA graphs)."""
    s, end = rec.settings, rec.get_up_end
    recovering, hidden, left = state["recovering"], state["hidden_frame"], state["blend_left"]
    row = torch.where(recovering, torch.minimum(hidden + 1, end[hidden]), f1)
    finite = torch.isfinite(S1[0][:, 2]) & torch.isfinite(S1[1][:, 2, 2])  # mdp.termination counts NaN as within
    row_within, command_within = finite & ~mdp.termination(S1, row), finite & ~mdp.termination(S1, f1)
    at_end = row == end[row]
    error = ~row_within | (at_end & ~command_within)
    switch = recovering & at_end & row_within & command_within
    still = recovering & ~switch
    errors = torch.where(still & error, state["error_controls"] + 1, 0)
    offset = (S1[0][:, :2] - mdp.motion.root_p[f1][:, :2]).detach()
    scale = torch.where(recovering, s.reward_scale, torch.where(left > 0, (s.blend_controls - left + 1) / s.blend_controls, 1.0))
    nxt = {"hidden_frame": row, "recovering": still, "error_controls": errors,
           "blend_left": torch.where(switch, s.blend_controls, torch.clamp(left - 1, min=0)),
           "command_offset": torch.where(switch[:, None], offset, state["command_offset"])}
    return nxt, row, scale, still & (errors >= s.stop_controls), switch


def termination(start_recovering, stop, release):
    """The termination of a control: one that starts recovering ends only by the recovery stop, any other by the
    release's termination against the command row ``release`` (mdp.termination in the simulator, the soft
    mdp.soft_termination in imagination)."""
    return torch.where(start_recovering, stop.to(release.dtype), release)


def value_bound(gamma: float, w_contact: float, reward_scale: float) -> float:
    """train.py's bound on the critic value, the largest step reward over 1 - gamma, with the tracking part of the step
    reward (at most MAX_STEP_REWARD) times the larger of the recovery reward scale and 1 (the largest scale of a
    tracking or blend control); the contact term (at most 2 w_contact dt) is not scaled and the penalties are at most
    0."""
    return (max(reward_scale, 1.0) * MAX_STEP_REWARD + max(w_contact, 0.0) * 2 * CONTROL_DT) / (1 - gamma)
