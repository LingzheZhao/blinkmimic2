# Recovery training from placed lying starts

`python -m blinkmimic2.train` can train a tracking policy that also gets up from lying on the floor and returns to
its command. This is recovery training in a reduced, single-network form of StableMimic
([arXiv 2608.02385](https://arxiv.org/abs/2608.02385)). A share of the episodes starts with the body placed at a low
row of a get-up recording. During such an episode the reward, the termination and the critic follow that recording,
called the hidden reference, while the policy reads only its command, which starts at a separate upright row and
advances as in any episode. Once the body is up, a blend leads the reward back to the command.

The same setting exists for the PyTorch trainer of [NablaMimic](https://github.com/LingzheZhao/NablaMimic) (its
`RECOVERY.md`), so that the two trainers can be compared. This trainer learns a model of the robot's dynamics from
simulated steps and trains the policy and the critic on rollouts imagined in that model, so the rule of the setting
is applied twice: in the simulator step that collects data and in the imagined steps that train the policy. One
function, `recovery.advance` in [`src/blinkmimic2/recovery.py`](src/blinkmimic2/recovery.py), holds the rule for both.

```bash
source scripts/presets.sh
bash scripts/train.sh logs/recovery_s0 --seed 0 --bones_list= --motion_dir DIR ${PRESET[recovery]}
bash scripts/train.sh logs/control_s0 --seed 0 --bones_list= --motion_dir DIR ${PRESET[recovery_control]}
```

`DIR` holds the 40 LAFAN1 G1 retarget CSVs. `PRESET[recovery_control]` is the host configuration (below), the
control condition; `PRESET[recovery]` adds the seven recovery options and nothing else. The options go together: a
partial set is refused. Without them the trainer runs as the release (BlinkMimic 2 v1.0.0) does: the same random
draws in the same order, the same buffer fields, the same captured graphs' inputs and log columns, and checkpoints
with the same weights and entries; `args.json` and the checkpoints' `args` do not hold the seven options.

## Settings

A **control** is one 0.02 s policy step (50 Hz); a **row** is one 0.02 s frame of the motion library. The
**command** is the row the policy reads: its observation's reference orientation and its goal (the next 10 rows).

| Option | Value | Meaning | Source |
| --- | --- | --- | --- |
| `--recovery_share` | 0.5 | probability that a sampled reset is a recovery start | StableMimic, Sections III-B and III-D, Table V |
| `--recovery_hidden_clips` | `fallAndGetUp1_subject1`, `fallAndGetUp1_subject5`, `fallAndGetUp2_subject2`, `fallAndGetUp2_subject3`, `fallAndGetUp3_subject1` | clips that give the start rows and the hidden reference | this repository's choice; StableMimic builds its get-up library from the retargeted LAFAN1 get-up subset (Section IV-A) and names no clips; `fallAndGetUp1_subject4`, a get-up recording of the library, is not a hidden clip |
| `--recovery_command_excluded_clips` | the five hidden clips and `fallAndGetUp1_subject4` | clips that never give a command row | this repository's choice; in StableMimic commands come only from the LAFAN1 dance subset, apart from the get-up library (Section IV-A) |
| `--recovery_low_pelvis HEIGHT UPZ` | 0.50 m, 0.5 | bounds of a low row | this repository's choice; StableMimic starts recovery from perturbed prone, supine and intermediate get-up rows (Section III-B) with no height or tilt rule, and uses 0.50 m and 60 degrees as the onset of a fall in its evaluation (Section IV-A) |
| `--recovery_rows UPRIGHT_RUN LONGEST_SPAN COMMAND_UPRIGHT` | 50 (1 s), 375 (7.5 s), 500 | upright run that ends a get-up; longest span from a start row to its get-up end; upright rows ahead of a command row | this repository's choice; StableMimic starts the transition back to the command on similarity to the end of the get-up and gives no threshold (Section III-B) |
| `--recovery_reward_scale` | 2.5 | factor on the six tracking terms while recovering | StableMimic, Section III-C, Table V |
| `--recovery_seconds STOP BLEND` | 2.0 s (100 controls), 1.5 s (75 controls) | consecutive time of target error that stops a recovery episode; time of the blend | StableMimic, Section III-B, Table V, for the durations; counting consecutive controls is this repository's choice, as the paper does not say how persistence is counted |

The durations are converted to controls; a duration that is not a whole number of controls is refused.
LONGEST_SPAN leaves room for the blend and 1 s of command tracking inside the 10 s (500-control) episode.
COMMAND_UPRIGHT equals the episode limit, so no clip end and no low command row meets a recovery episode.

## Host configuration

`PRESET[recovery_control]` is `PRESET[baseline]` (the release's first-phase settings for the whole run: no exams,
frontier practice, failure penalty, deployment stream, practice starts or phase schedule; 7,000 iterations, a
checkpoint every 100) with `--no-pi_ref_skip`, so that actions are set around the default pose. With the recovery
options `train.py` refuses each option that reads the command row as if the body followed it:

| Refused | Why |
| --- | --- |
| `--exam_frac` other than 0 | exam worlds play whole clips from their first rows and are scored by the danger and terminations against the command row |
| `--front_share` other than 0 | frontier practice restarts before the rows where exams failed |
| `--fail_penalty` other than 0 | the penalty is the soft termination against the command row at every imagined step |
| `--wstream_frac` other than 0 | the deployment stream plays clips from their first rows in the world model, ends them by the danger against the command row, and its starts hold no regime |
| `--start_prac_frac` of 0 or more | it splits imagination starts between exam and practice worlds |
| a non-empty `--phase` | the schedule turns the options above on during the run |
| `--pi_ref_skip` | the action adds the command row's next joint positions, `(q_ref(f+1) - q0) / alpha` |
| `--reset_aprev_ref` | a reset in the middle of a clip sets the previous action to the one that holds the reset pose, judged on the command row |

`--exam_pre_every`, `--exam_mix` and `--share_ramp` act only with exams or the schedule (their own checks need
them) and are not refused. `--fail_reset` is kept: the reset bins see tracking episodes only (below).
`--contact_reward` is kept: the term reads the target row.

## Rows of the motion library

`recovery.build` classifies the rows once, after the library is loaded, from the pelvis (root) pose of each row.

- A row is **low** where its pelvis is below HEIGHT or the world z component of its pelvis up axis is below UPZ (a
  tilt beyond 60 degrees). Every other row is upright.
- The **get-up end** of row `r` is the last row of the first run of UPRIGHT_RUN consecutive upright rows that ends at
  or after `r` inside its clip (none where the clip holds no such run).
- A **start row** is a low row of a hidden clip whose get-up ends inside its clip at most LONGEST_SPAN rows later.
- A **command row** lies outside the command-excluded clips, and it and the next COMMAND_UPRIGHT rows are upright and
  before its clip's last row.

Unknown clip names, an empty start or command table, and a hidden clip that is not command-excluded are refused: a
reset is a recovery start where its body row differs from its command row, which needs the two tables apart.
`recovery.json` records the counts of a run's library. Measured with these rules on the 40 LAFAN1 G1 clips at this
library's 50 Hz, the hidden clips hold 18,280 low rows, of which 16,067 are start rows ending at 91 distinct get-up
ends.

## Starts

Every sampled reset (the release's reset bins: 1 s bins drawn by a mix of uniform, low-return and running-average
weights) draws, in this order: the reset rows; one uniform per world, below the share for a recovery start; a start
row, uniform over the start rows; a command row, uniform over the command rows; then the reset noise and the joint
offsets. The reset of training (`TrackingSim.reset_masked`, captured with the collection step) draws these for every
world and uses them for the worlds being reset. Without the options only the release's draws are made. The initial states, set
when the simulator is built, are sampled resets too; as in the release they start at episode step 0.

A recovery start places the body at the start row as any reset places it at its row: the row's positions and
velocities plus the reset noise (half-widths 0.05, 0.05, 0.01 m; 0.1, 0.1, 0.2 rad; 0.5, 0.5, 0.2 m/s; 0.52, 0.52,
0.78 rad/s; 0.1 rad on the joints, StableMimic, Table V), a zero previous action. The command starts at the command
row and its reset bin is the command row's. Any other reset gets both at its reset row.

## The hidden row and what reads it

Each world's state carries five more values (`recovery.KEYS`), which the simulator keeps per world and the replay
buffer stores with every transition, as the values of the state the transition starts from:

| Value | Meaning |
| --- | --- |
| `hidden_frame` | the row the last control targeted: the hidden row while recovering, the command row otherwise (at a reset: the body row) |
| `recovering` | whether the hidden row is followed |
| `error_controls` | consecutive controls of target error |
| `blend_left` | blend controls still to run |
| `command_offset` | planar offset of the command reference, set at the switch back to the command |

| Reads the hidden row (while recovering) | Reads the command row |
| --- | --- |
| the six tracking terms of the reward, in the simulator and in imagination | the policy's observation and goal |
| the imagined contact term (`--contact_reward`, imagination only) | the action map (the default pose plus the scaled action) |
| the recovery stop, in the simulator and in imagination | the clip end and the episode limit |
| the critic: its features (`MDP.critic_features`) and its goal (`motion.goal`) | the reset bins |
| | the release's termination, outside recovery (hard in the simulator, soft in imagination) |

The policy gets no regime flag and no blend count, so it sees what it sees at deployment. The critic's input keeps
its size and has no regime entry.

## Order inside a control

A control's regime is that of the state it starts from: **recovery** while `recovering`, **blend** while
`blend_left` > 0, **tracking** otherwise. `recovery.advance` takes the regime of the state a control starts from, the
state it reaches and the command row after it, and gives:

1. the **target row**: `min(hidden + 1, get-up end of hidden)` for a recovering state, so the hidden row advances
   and holds at its get-up end; otherwise the command row;
2. **target error** of a recovering control: the body outside the release's termination thresholds (0.25 m pelvis
   height difference, 0.8 difference of the world z of the pelvis up axis; the paper gives no training thresholds)
   at the target row, or, at the get-up end, at the command row. A non-finite pelvis height or up axis is outside
   every row's thresholds. An error adds one to `error_controls`; a control
   without error sets it to 0;
3. the **switch**: a recovering control at the get-up end with the body within both rows' thresholds stops
   recovering. `blend_left` becomes BLEND, and later controls lower it by one to 0. `command_offset` becomes the
   pelvis's planar position minus the command row's, without gradient; the command's yaw is kept (StableMimic,
   Section III-D, re-aligns the command's horizontal translation to the recovered pose and keeps its yaw);
4. the **stop**: a control that is still recovering with STOP or more consecutive errors;
5. the **reward scale** of the control's regime: the recovery reward scale; `(BLEND - left + 1) / BLEND` in the blend
   with `left` blend controls still to run (1/75 at the first blend control, 1 at the last); 1 in tracking.

In the **simulator** (`TrackingSim.step_core`, inside the captured collection step), a control runs: the policy
acts on the command row's inputs; the physics steps; the command row advances by one; `recovery.advance`; the reward
at the target row; the termination; the timeout at the command clip's end or 500 controls; the transition and the
regime of its starting state go to the buffer; the regime is updated in place. Nothing else changes at the switch:
the previous action, the history, the episode step and the command row continue.

In **imagination** (`policy_update` in `train.py`), each rollout of 32 controls starts from a buffer transition and
takes the regime stored with it, the error count included, since the 100-control stop is longer than the horizon. At
each imagined control the policy acts on the command row's inputs, the world model steps, the command row advances
by one and `recovery.advance` runs on the imagined state, without gradient. The rewards and terminations of the
whole rollout are then computed at the target rows, the critic's values at each state's own row.

**Reward.** A recovery control takes the reward scale times each of the six tracking terms at the target row, with
the squared pelvis height difference alone as its root position error (StableMimic, Section III-C, uses root
height; the pelvis is the root and the anchor of this trainer). A blend control takes its blend weight, linear in
the blend control (step 5 above), times the six terms; the paper gives neither the shape of this weight nor a value
for its separate transition coefficient, and that coefficient is not applied here. A tracking control takes the release's terms. Outside recovery the root position error is taken
against the command's root moved by `command_offset`; the body terms are anchored to the robot's planar position and
heading in the release and need no offset. The action-rate and joint-limit terms and the imagined contact term keep
their weights. Recovery rows can exceed the release's largest step reward (`MAX_STEP_REWARD`, 0.02 s times the sum
of the six tracking-term weights), so `sim_rew`, `imag_rew` and `ep_ret_norm` of the log are not comparable between
the two conditions.

**Termination.** A control that starts recovering ends only by the recovery stop (StableMimic, Section III-B:
termination is measured against the get-up target during recovery); the release's height and tilt termination
against the command row ends any other control. In the simulator the stop ends the episode as a termination; in
imagination the stop flag (0 or 1, without gradient) takes the place of the soft
termination in the survival weights, the critic's targets and the bootstrap. The rule is written on the regime the
control starts in. In the simulator it equals a mask on the reached state's flag, because a switch needs the body
within the command row's thresholds, where the hard termination is 0. In imagination it does not: a switch control
gets 0, where a mask on the reached state's flag would give the soft termination against the command row, which is
small but above 0 within the thresholds.

**Value bound.** The release clamps the critic's values, targets and bootstrap at `(MAX_STEP_REWARD + 2 w_c dt) /
(1 - gamma)` = (0.1 + 0.01) / 0.01 = 11, the largest step reward over `1 - gamma` with the contact weight `w_c` =
0.25. With recovery the tracking part is at most 2.5 times as large: `(2.5 x 0.1 + 0.01) / 0.01` = 26
(`recovery.value_bound`, which takes the larger of the reward scale and 1). The penalties are at most 0 and the blend
weight at most 1.

**Reset bins.** Only whole tracking episodes update the reset-bin returns and failure rates; a recovery episode,
its blend and the tracking after it leave the bins of the command rows as they are. The log's `ep_len`,
`ep_ret_norm` and `term_frac` count every ended episode, recovery stops among the terminations.

## Records

`RUN/recovery.json`, beside `args.json`, holds the settings (durations in controls), the control time and six row
counts: the low rows of the hidden clips, those without a get-up end in their clip, those beyond the longest span,
the start rows, the distinct get-up ends and the command rows. `args.json` and the `args` of every checkpoint hold
the seven options as parsed in a run with them; a run without them has none of these keys. The checkpoints hold no regime: the replay buffer is not
saved, as in the release.

`train_log.csv` has these columns with the options only, per log row (over the iterations since the last row):

| Column | Meaning |
| --- | --- |
| `recovery_start_share` | share of the episodes started that are recovery starts |
| `recovery_start_low_heading_share` | share of the recovery starts whose placed pelvis x axis has a horizontal length below 0.1 (below) |
| `recovery_collected_share_tracking`, `_recovery`, `_blend` | share of the collected transitions by the regime of the state they start from |
| `recovery_switches`, `recovery_stops` | switches back to the command and recovery stops in the simulator |
| `recovery_imagined_share_tracking`, `_recovery`, `_blend` | share of the imagined controls by the regime they start in |
| `recovery_imagined_tracking_total` | running total of the imagined tracking controls since the run started |
| `recovery_imagined_survival_recovery_starts` | mean survival weight at the horizon over imagined rollouts that start recovering; empty (nan) without one. Such a rollout ends only by the stop, which needs 100 consecutive errors against a 32-control horizon, or by the soft termination after a switch, so the column stays near 1 when the termination is masked as above and falls towards 0 when it is not. It checks the mask; it does not measure getting up, which `recovery_switches` against the recovery starts and the blend share of the collected rows show |
| `recovery_clamped_share_values`, `_targets`, `_bootstrap` | share of the critic's values, targets and bootstraps clamped at the value bound (the release prints them as `vclamp`) |

**Tracking exposure.** A control run trains on 2,048 x 32 = 65,536 imagined tracking controls per update. The
control update matched in tracking exposure to an update of a recovery run is the first whose running total, 65,536
times its update count, reaches that update's `recovery_imagined_tracking_total`.

## Known limit: the pelvis heading

The release takes a heading frame from the horizontal projection of the pelvis x axis (`rotations.yaw_mat`). Measured
on the start rows of the 40 LAFAN1 G1 clips (above), that projection is shorter than 0.1 for 1,212 of the 16,067
(7.5 %) and shorter than 0.3 for 2,737 (17 %): the pelvis x axis is near vertical on prone and supine bodies. The heading then turns fast with small rotations. It
enters the critic's features (`invariant_features`, the body and root errors), the world model's inputs and frames
(`networks.WorldModel`), and the goal frame of the critic at a hidden row (`motion.goal`). The reward's anchoring
uses the heading of the body relative to its reference row, which stays defined near that row. The frames are
unchanged; `recovery_start_low_heading_share` gives the share of recovery starts in this band.

## Differences from StableMimic

| StableMimic | This repository |
| --- | --- |
| two experts whose actions a mixing network on proprioception blends, trained with extra mixing losses (Sections III-A, III-C) | one policy network |
| trained by PPO (Section III-D) | trained through rollouts imagined in a learned dynamics model, closed by a critic |
| start frames half uniform, half weighted towards failed phases (Section III-B) | start rows uniform |
| a success term in the reward, not defined further (Section III-C) | no success term |
| action-rate, joint-limit, self-collision and power penalties (Section III-C) | the release's action-rate and joint-limit penalties |
| the critic reads the command and the active get-up reference (Section III-A, Table II) | while recovering, the critic reads the hidden row in place of the command row |
| commands from the dance subset only (Section IV-A) | every clip outside the command-excluded clips, get-up recordings among them |
| 20 s training episodes (Table V) | at most 500 controls (10 s) |

For the start, command and get-up end rows, the settings table gives what the paper states.

## Differences from the NablaMimic setting

The settings, the row rules, the regime rule and the reward and termination rules are those of NablaMimic's
`RECOVERY.md`. These differ:

| NablaMimic | This repository |
| --- | --- |
| the policy trains through 64-control windows of NablaMimic's differentiable simulator that start from replay states | the policy trains through 32-control rollouts imagined in a learned dynamics model; the rule runs in the simulator step and again in the imagined steps, with the regime and the error count carried in from the buffer |
| the stop terminates the reached state | in imagination a recovering control's termination is the stop flag, without gradient; other controls keep the release's soft termination against the command row |
| the termination masks the command-row terms on the reached state's flag | written on the regime the control starts in: equal in the simulator; in imagination a switch control gets 0 in place of the soft termination against the command row (see Termination) |
| a non-finite state ends the episode at once | the release's termination has no non-finite test, and none is added; in the regime rule a non-finite pelvis height or up axis is a target error, as in NablaMimic |
| the anchor-position term and the offset use the torso link | they use the pelvis (the paper's root, this trainer's anchor) |
| a termination penalty at the target row | no termination penalty; the release's failure penalty is refused |
| no contact term | the imagined contact term (weight 0.25) reads the target row and is not scaled |
| reset bins updated by transitions that start in tracking | reset bins updated by whole tracking episodes |
| draws: reset rows, reset noise, then the share, start and command uniforms | reset rows, the share, start and command uniforms, then the reset noise and joint offsets (the body rows must be known before the noise is applied) |
| initial states get random episode ages | initial states start at episode step 0, as in the release |
| tracking exposure in window rows (262,144 per update) | in imagined controls (65,536 per update) |
| `--recovery-controls STOP BLEND` in controls | `--recovery_seconds STOP BLEND` in seconds, converted to the same 100 and 75 controls |
| log columns of the switch yaw, clip ends and episode limits of recovery controls, terminations in the blend and the tracking reward | none of these; the start shares, the imagined shares and survival and the clamped shares instead |
| host recipe on NablaMimic's differentiable simulator with absolute actions | `PRESET[baseline]` with `--no-pi_ref_skip` on MuJoCo Warp with mjlab's G1 |
| library: 18,138 low rows, 15,952 start rows, 91 get-ups | 18,280, 16,067 and 91, measured on this library's resampling of the same clips (rows per clip differ); `recovery.json` holds the six counts of a run, the command rows among them |
| `recovery.json` also states which parts read the hidden row and which the command row | `recovery.json` holds the settings, the control time and the counts; this document states the readers |

The critic reading the hidden row in place of the command row is the same choice in both, and a difference from
StableMimic.

## Not implemented

- pushes during training (the paper's pushes belong to its evaluation);
- falls during tracking turned into recovery during training;
- recovery with the options the host configuration refuses (above);
- the eager episode end `TrackingSim.step_tail`, used outside training, keeps the release's bins (recovery episodes
  are not excluded there);
- the world model's prediction error by regime;
- reading MuJoCo Warp's contact and constraint capacity during training. Measured with the release's reset and
  MuJoCo Warp 3.11.0 on the 40 LAFAN1 G1 clips, a body placed at each low row and held for 50 controls used at most
  34 of the 128 contacts and 125 of the 500 constraint rows per world.

## Rollouts and evaluation

`tools/rollouts.py` and `tools/evaluate.py` build the simulator without the recovery options, whatever the run's
`args.json` holds: a rollout reads a policy, and the hidden reference is a device of training.

## Tests

`tests/test_recovery.py` runs on the CPU with a generated get-up library and stand-ins for the simulator. It covers
the row tables on both library classes and their refusals; the regime rule (the hidden row's advance and hold, the
stop at control 100 and not 99 by the height and by the tilt alone, the count reset, the switch only within both
rows' height and tilt thresholds, a non-finite body as a target error, the blend weights, the detached offset, the
scale); the order and tables of the start draws; the termination of a recovering control
with the simulator's and imagination's terminations; the value bound; the reward and critic features without the
options equal to the release's functions, with the scale on the tracking terms only, the height-only root term and
the offset; the buffer's fields and the regime a start carries; the simulator step with and without the options
(the target row, the masked termination, the stored regime of the starting state, the update in place); the reset
bins; the eager and the masked reset (no recovery draw without the options); the option refusals, each refused option
alone, the kept options, the presets and an `args.json` without the options.

A GPU run is needed for the rest: the imagined step in `policy_update` (the carried regime, the reward and
termination, the critic and bootstrap at the hidden row, the raised value bound), the captured collection and update
graphs with the recovery tensors, the log columns, `recovery.json` on the LAFAN1 library, the initial states of a
built simulator, and the training time and memory. The GPU simulator is not bitwise reproducible, so a run without
the options is the release's by the code path, not by comparing runs.
