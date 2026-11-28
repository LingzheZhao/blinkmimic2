# BlinkMimic 2's world model by posture, width and link inputs in recovery training

**Question.** Is BlinkMimic 2's world model less accurate on low-posture states than on upright-posture states, and is
that why trials started from the first row of a get-up clip end early (first-row trials end when the robot falls or the
tracking error exceeds the termination limit)? The pair study names a competing explanation for those ends:
NablaMimic's trainer penalizes termination using the hidden get-up reference during recovery, while BlinkMimic 2's
recovery training has its failure penalty switched off.

**Answer.**

- In first-row trials, each of the pair study's four world models has a higher error on low-posture than on
  upright-posture states: stepped open loop 8 controls ahead, the low-posture error is 8.01 to 11.20 times the
  upright-posture error; after 1 control it is 5.61 to 6.22 times (Table 1).
- The declared rule on world model error by posture gives no outcome: its second condition needs a checkpoint of the
  condition "failure penalty during recovery", and that condition could not be trained to the end. The comparison with
  DynaMimic's world model was kept on the recovery benchmark of the pair study; this choice was made after the readings
  and is not an outcome of the rule.
- First-row trials of the single-network recovery trainings end in get-up windows rarely: in each of the 13 trainings of
  Table 3, 0 to 4 trials end in a get-up window, among 235 to 273 windows reached on the five clips that give the hidden
  references, summed over three sets. The study does not relate those ends to the world model's error at the state where
  a trial ends.
- Each of the four wider world models without link inputs (hidden widths 2048, 2048, 2048) has a lower error after 8
  controls than each of the four 1024-wide ones without them: 0.067 to 0.071 against 0.097 to 0.101 on upright-posture
  states, 0.697 to 0.837 against 0.993 to 1.054 on low-posture states (Table 4).
- A lower tracking error over the 34 commanded clips with the wider world model is not shown. A first pair of trainings
  read 0.0320 and 0.0334 m (seeds 0, 1) and a lower tracking error was stated; the second pair read 0.0352 and 0.0342 m,
  against 0.0342 and 0.0327 m for the 1024-wide trainings made beside them, and the statement is withdrawn.
- For world models given DynaMimic's link inputs (18 links), both seeds' error after 1 control lies below the range of
  the four trainings without link inputs at the same width, on both posture classes and at both widths (Table 8). After
  8 controls the two seeds lie below that range on upright-posture states at 1024 width (0.0967 and 0.0834 against
  0.0972 to 0.1013) and overlap it in the other three readings. The wider world model with link inputs was added after
  results had been read.
- Trainings repeated at one seed with the same options (the pair study's differ only in the checkpoint interval) end up
  to 0.0032 m apart in tracking error (the mean aligned body-position error over the 34 commanded clips), so the
  declaration counts no difference below 0.0032 m between single trainings as evidence; this sentence of the
  declaration was added after results had been read.
- Wider tracking-only reads 0.0262 and 0.0264 m against 0.0274 and 0.0273 m for 1024-wide tracking-only (seeds 0, 1),
  differences of 0.0012 and 0.0009 m from trainings at different code revisions, with no repeat.
- Exploratory, not declared: four single-network recovery policies (two with a 1024-wide world model, two with a
  wider one) were rolled out, and each of the four world models, one trained with each policy, was read on each policy's
  first-row rollouts, in the error scale of the world model trained with that policy. On low-posture states after 8
  controls, the world model trained with a policy has a lower error on that policy's rollouts than the other three, for
  each of the four policies.

The study trains the world model variants with `python -m blinkmimic2.train`, rolls out their checkpoints with
`tools/rollouts.py` on the scene of the [recovery pair study](../recovery_pair/README.md), reads the rollouts with that
study's `readings.py` and with [`groups.py`](../recovery_pair/groups.py), and reads each world model's open-loop error
on recorded states with [`tools/world_model_error.py`](../../tools/world_model_error.py). The link inputs are
DynaMimic's (arXiv 2611.32944, Section 3.2, "Predicting contact wrenches"); the recovery setting is StableMimic's (arXiv
2608.02385, Sections III-B to III-D and Table V) as [RECOVERY.md](../../RECOVERY.md) reduces it. The widths 2048, 2048,
2048, the 18 links, the posture thresholds (pelvis below 0.50 m or the world z component of the pelvis up axis below
0.5), the checkpoint every 500 iterations and every threshold in the decision rules are this study's choices; the
failure penalty's weight 3 and its first iteration 2,000 are those of the BlinkMimic 2 release's own schedule. The pair
study's four trainings are read here; its page calls tracking-only its "control" and single-network recovery its
"recovery", and this page uses the names of the study's declaration. On this page, as on that page, a control is one
0.02 s policy step. The rigid-body integrator comparison has its own study page, that of the rigid-body integrator study
(`studies/rigid_body_integrator`, in a separate pull request of this repository).

## Declaration

### Conditions

Each condition below is single-network recovery with one change, except tracking-only. A **1024-wide** world model has
hidden widths 1024, 1024, 1024 (the default of `--wm_hidden`); a **wider** one has 2048, 2048, 2048. A training made
at another time or place is named by where: the pair study's training; the training made beside the
mixture-of-experts (MoE) conditions of the MoE actor study (`studies/moe_actor`, in a separate pull request of this
repository); and the training made beside the world-model conditions (the
1024-wide single-network recovery trained with the link-input trainings, the second trainings of the wider world model
and the wider-with-link-inputs trainings of this study). Trainings of this study pass
`--ckpt_every 500` and `--bones_lib` set to the pair study's string, so that their library equals the pair study's
scene.

| Condition | Options beyond `PRESET[...]` of [`scripts/presets.sh`](../../scripts/presets.sh) | Seeds | Declared, or added after |
| --- | --- | --- | --- |
| tracking-only, 1024-wide | `PRESET[recovery_control]` (the pair study's training) | 0, 1 | declared |
| single-network recovery, 1024-wide | `PRESET[recovery]` (the pair study's training, and the training made beside the world-model conditions) | 0, 1 | declared |
| failure penalty during recovery, from iteration 0, critic targets include it | `PRESET[recovery] --recovery_fail_penalty 3 0` | 0, 1 | declared |
| failure penalty during recovery, from iteration 0, inside the imagined rollout only | `PRESET[recovery] --recovery_fail_penalty 3 0 --pen_horizon_only` | 0 | the third of three trainings of the penalty from iteration 0 that ended early (with row 3); the declaration does not say whether the penalty also enters the critic targets |
| failure penalty during recovery, from iteration 2,000 | `PRESET[recovery] --recovery_fail_penalty 3 2000 --pen_horizon_only` | 0 | redefined after the three trainings above ended early; could not be trained |
| wider world model | `PRESET[recovery] --wm_hidden 2048,2048,2048`; trained twice (first and second trainings) | 0, 1 | added to the declaration; the declaration does not say whether results had been read when it was added |
| world model with link inputs | `PRESET[recovery] --wm_link_inputs` and the 18 names of `g1_constants.SUPPORT_LINKS` | 0, 1 | declared, as the DynaMimic comparison |
| wider world model with link inputs | `PRESET[recovery] --wm_hidden 2048,2048,2048 --wm_link_inputs` and the 18 names | 0, 1 | added after the wider world model read a lower error and, as then read, a lower tracking error |
| tracking-only with the wider world model | `PRESET[recovery_control] --wm_hidden 2048,2048,2048` | 0, 1 | added after the same results |

The first form of the failure penalty was trained at a revision where its option was spelled differently and took the
weight alone, acting from iteration 0 (Revisions and cards); in this branch's code it is `--recovery_fail_penalty
WEIGHT FIRST_ITERATION` with FIRST_ITERATION 0. Both forms subtract WEIGHT times the soft termination probability
against the target row of `recovery.advance` (the hidden row while recovering, the command row otherwise) from each
imagined control's reward. A further training, which is not one of the declared conditions, applied
the release's failure penalty to tracking-only (`PRESET[recovery_control] --fail_penalty 3 --pen_horizon_only
--iterations 2500`, seed 0); without the release recipe's exam worlds (`--exam_frac`, described in
[RECOVERY.md](../../RECOVERY.md)) the penalty acts from iteration 0, not from 2,000, and the training completed.

**Link inputs** (`--wm_link_inputs`, `networks.link_features`). For each named link, 7 values are appended to the world
model's inputs: the height (world z of the link's origin), the origin's position relative to the base (3) and the
origin's linear velocity (3), the last two in the base's heading frame (`rotations.yaw_mat`, the frame of the world
model's increments), from the trainer's forward kinematics. They are normalized by the world model's input normalizer
with the other inputs. The 18 links are `g1_constants.SUPPORT_LINKS`: the bodies of mjlab's G1 collision model with
collision geoms, all with floor contact in the get-up clips. DynaMimic uses 15 candidate links on its robot; a set of 15
cannot be left-right symmetric here, and the trainer replaces half of each world model training batch by its left-right
mirror (`--wm_mirror 0.5`), so all 18 are used.

### Readings

R, J, F, C and E, the placed starts, twins and first-row trials are defined on the [pair-study
page](../recovery_pair/README.md). This page adds the following.

- **Get-up window** (`get_up_windows` in [`groups.py`](../recovery_pair/groups.py)). In a clip, the rows from 50 before
  to 50 after the end of a sequence of at least 25 consecutive low rows (pelvis below 0.50 m or world z component of the
  pelvis up axis below 0.5). A first-row trial ends in a window when its termination falls inside it.
- **World model error** (`networks.world_model_error_components`, `world_model_normalized_error`). The state error (root
  position in the heading frame, root rotation vector, root linear and angular velocity in the body frame, joint
  positions, joint velocities) divided component-wise by the error scale, squared and averaged over 12 + 2 x 29 = 70
  components. It is unitless, in the trainer's normalized units; the log column `wm_err8` has the same form.
- **Error scale** (`networks.world_model_error_scale`). Per component, the larger of the world model's increment
  standard deviation and the trainer's lower bound. Each checkpoint has its own scale unless `--error_scale_checkpoint`
  names another's, so errors of different checkpoints are in different units.
- **Error after 1 and after 8 controls** (`tools/world_model_error.py`, `open_loop`, `summary`). The world model is
  stepped open loop from a recorded state with the recorded actions; the error of its first and its eighth prediction is
  averaged over segments (`segments`: 8 consecutive controls of one trial, no trial end before the last, `--stride k`
  starts a segment at every k-th control).
- **Low-posture and upright-posture state** (`low_posture` in `tools/world_model_error.py`). A state is low-posture
  when, at the start of a segment, the pelvis is below 0.50 m or the world z component of the pelvis up axis is below
  0.5, and upright-posture otherwise. **Ratio**: low-posture error over upright-posture error.
- **Non-foot contact** (`Contacts` in `tools/world_model_error.py`). At the recorded state before a segment, MuJoCo's
  collision on mjlab's G1 finds a floor contact with penetration on a geom outside the two ankle-roll links; it is
  recomputed, not the simulator's own record. **Reference get-up windows** (`in_get_up_windows` in the same file):
  segments of first-row trials that start in a get-up window.
- **State groups** (`STATE_GROUPS` in `tools/world_model_error.py`). The error restricted to root position (3
  components), root rotation (3), root linear velocity (3), root angular velocity (3), joint positions (29) and joint
  velocities (29). A group's share of the total is its mean times its width over 70, divided by the total.
- **Tracking error over the 34 commanded clips** (`groups.clip_group`). The mean over the 34 commanded clips of E in the
  first-row trials; the value of a checkpoint is the mean over its three sets of rollouts. The same function gives the
  excluded clips (the six get-up clips that are not commands) completed and their E.
- **Ends per get-up window** (`groups.window_ends`). Over the windows a first-row trial enters, the number in which its
  first termination lies, for the five hidden clips, fallAndGetUp1_subject4 and the commanded clips with a get-up.
- **Supplementary reading.** The mean of the tracking error at a training's last checkpoint (three sets) and at the
  checkpoints 500 and 1,000 iterations before it (one set each); it is computed from the tracking errors of the three
  checkpoints (`groups.clip_group`), as in Table 7b.

### Decision rules

The study's declaration was written before the first training and changed afterwards (see Changes to the declaration
below). The rules this page reads are quoted below in their current wording, with those changes applied (EMA is the
exponential moving average of the policy weights). Four of the quoted passages were
added after results had been read: the last sentence of the rule on the world model error by posture, the sentence on
differences below 0.0032 m, the reading of the checkpoints at 5,500 to 7,000, and the last passage on the supplementary
reading.

> The policy read from a training is the averaged (EMA) policy of its last checkpoint. Trials of BlinkMimic 2 are not
> repeatable one by one, so each of its checkpoints is read from three sets of rollouts and all three are reported; in
> every rule below, the value of a BlinkMimic 2 checkpoint is the mean over its three sets.

> **World-model error by posture [...].** For the four world models of the BlinkMimic 2 pair study (tracking-only
> and single-network recovery, seeds 0 and 1): the open-loop prediction error after 8 controls on recorded simulator
> states and actions, in the trainer's normalized state error, on upright-posture states, on low-posture states and
> inside get-up windows. The DynaMimic comparison [the comparison with DynaMimic's world model] **stays on the
> recovery benchmark** if, for each of the four world models, the error on low-posture states is at least twice the
> error on upright-posture states, **and** first-row trials of the condition "failure penalty during recovery" still
> end in get-up windows: in four sets of first-row trials of that condition's seed 0 checkpoint, more than 2 trials in
> total end in a get-up window of the five clips that give the hidden references (91 such windows per set, 364 in four
> sets when every trial reaches every window). Otherwise the DynaMimic comparison **moves to** tracking-only training on
> LAFAN1 segments grouped by non-foot contact, as in DynaMimic's table of tracking by contact and intensity. The second
> condition of this rule could not be read, and the comparison is kept on the recovery benchmark [by the change that
> set this rule aside, listed under Changes to the declaration below].

> **Reported without a rule.** R, J, F, the commanded clips completed, completion and E on the six excluded clips, and
> the first-row trials that end in get-up windows.

The declaration's paragraph also lists StableMimic's post-fall measures (over the three seconds after a fall, the 95th
percentiles of limb speed, of joint-target rate and of joint torque); this page does not read them.

> On BlinkMimic 2, a difference between two conditions that rests on one training per condition and seed and is smaller
> than 0.0032 m is not evidence of a difference.

> One reading is added to tell whether a training's tracking error still moves late in the training: the checkpoints at
> iterations 5,500, 6,000, 6,500 and 7,000 of five trainings (the three trainings of single-network recovery at seed 1
> and the two of the wider world model at seed 0), one set of rollouts each. If in two or more of the five trainings the
> tracking error differs by 0.001 m or more between two of the four checkpoints, a supplementary reading is added for
> every BlinkMimic 2 training: the mean of the tracking errors at iterations 6,000, 6,500 and 7,000. If in every one of
> the five the four checkpoints lie within 0.0005 m, no such reading is added. In between, the decision is taken after
> the reading and reported as such.

> The declared readings and rules stay on the last checkpoint; each rule is read a second time on the supplementary
> reading, and that second reading is exploratory.

**Changes to the declaration.**

- The world model error is classed by the recorded state at the start of each prediction, and get-up windows by the
  clip; the first wording left both open. This was settled before any world model error was read.
- Two world-model settings were withdrawn as conditions: one is switched on in every training, the other has no trainer
  option and its contact loss is multiplied by zero. No training or reading of this page concerns them. The declaration
  does not say whether results had been read when this was changed.
- The failure penalty during recovery was redefined to start at the iteration at which the BlinkMimic 2 release starts
  its own, after three trainings that applied it from the first iteration had ended before iteration 600 and their
  returns had fallen. A counterpart on NablaMimic's trainer was added: single-network recovery without NablaMimic's
  termination penalty.
- The rule's wording names the four world models it reads. It was settled after the first world model error readings had
  been seen, and takes the stricter reading.
- The wider world model was added to the declaration to separate a world model that is too small for low-posture
  states from a world model of this kind being less accurate on them at any size; the declaration does not say whether
  results had been read when it was added, and the first wider trainings came after the first world model error
  readings of the pair study's four trainings.
- Card models: trainings run on RTX 4090 cards; rollouts of BlinkMimic 2 checkpoints run on RTX 3090 cards, because RTX
  4090 cards were not available for them; all rollouts of one comparison come from one card model, and a checkpoint also
  rolled out on an RTX 4090 is reported beside its RTX 3090 reading. The declaration does not say whether results had
  been read when this was added.
- The failure penalty during recovery could not be trained, and the rule was set aside, after the redefined condition
  had been trained at seed 0 and its returns had fallen. The comparison was kept on the recovery benchmark; this choice
  was made after the readings and is not an outcome of the rule.
- The wider world model with link inputs and tracking-only with the wider world model were added after results had been
  read: the wider world model had a lower error on both posture classes and, as then read, a lower tracking error at
  both seeds, at about 1.3 times the duration of a training iteration (0.160 s against 0.121 s).
- After the second trainings of the wider world model had been read, the statement that the wider world model had a
  lower tracking error at both seeds was withdrawn, with the finding on trainings repeated at one seed and the sentence
  quoted above on differences below 0.0032 m. No further trainings on the world model's width or inputs were run.
- After those readings, the reading of the checkpoints at 5,500 to 7,000 was added, and its outcome added the
  supplementary reading, the mean of the tracking errors at 6,000, 6,500 and 7,000 iterations.

## Commands, in order

Run from the repository root. The commands and options are those of this branch's code; each training, rollout and
reading ran at a revision of Revisions and cards, whose last column says how that revision's code differs. The pair
study's revision is cited on the [pair-study page](../recovery_pair/README.md) (Results, "Conditions"), and this
study's base has the same `src`, `tools` and `scripts`. `DIR` holds the 40 LAFAN1 G1
retarget CSVs, `SCENE` is the scene folder of the pair study (the output of its `scene.py`), `BONES_LIB` is the string
that the pair study's trainings record as `bones_lib`, and `NABLAMIMIC` is NablaMimic's `studies/recovery` at the
revision named on the pair study page.

1. **Trainings**, one RTX 4090 with 48 GB each, with the options of the Conditions table.

   ```bash
   source scripts/presets.sh
   python -m blinkmimic2.train --log_dir RUN --seed SEED --bones_list= --motion_dir DIR --bones_lib BONES_LIB \
       ${PRESET[recovery]} --ckpt_every 500 [--wm_hidden 2048,2048,2048] [--wm_link_inputs LINK ...] \
       [--recovery_fail_penalty 3 FIRST_ITERATION] [--pen_horizon_only]
   ```

   The pair study's four trainings were run with the preset's `--ckpt_every` (100) in place of `--ckpt_every 500`.

   `FIRST_ITERATION` is 0 (with or without `--pen_horizon_only`) or 2000 (with it), as in the Conditions table; the
   trainings from iteration 0 ran at a revision where the option was spelled differently (Revisions and cards). The
   18 `LINK` names are those of `g1_constants.SUPPORT_LINKS` (defined in a revision after that of the link-input
   trainings, which were given the same 18 names); tracking-only uses `${PRESET[recovery_control]}`.
2. **Rollouts**, one RTX 3090 each (Table 1 reads the pair study's own RTX 4090 rollouts): three sets at iteration
   7,000, one set at 6,000 and 6,500 and, for the five trainings of Table 7, one set at 5,500, each with the revision
   that trained the checkpoint, except the pair study's checkpoints at 7,000, rolled out with a later revision whose
   rollout tool and modules under `src/blinkmimic2` equal the pair study's except `networks.py` (functions added,
   `Policy` unchanged), and except `recovery.py` and `train.py`, which the tool does not run.

   ```bash
   FIRST_ROW_CONTROLS=$(python -c "import json; print(json.load(open('$SCENE/scene.json'))['first_row_controls'])")
   python tools/rollouts.py --run_dir RUN --ckpt ITER --scene $SCENE/placed.json --termination nonfinite \
       --controls 1000 --out OUT/placed
   python tools/rollouts.py --run_dir RUN --ckpt ITER --scene $SCENE/twins.json --termination nonfinite \
       --controls 1000 --out OUT/twins
   python tools/rollouts.py --run_dir RUN --ckpt ITER --scene $SCENE/first_rows.json --termination recipe \
       --controls $FIRST_ROW_CONTROLS --out OUT/first-rows
   ```

3. **Readings**, on the CPU, per set.

   ```bash
   python studies/recovery_pair/readings.py --pair studies/recovery_pair/pair.json --readings-core $NABLAMIMIC \
       --scene $SCENE --placed OUT/placed --twins OUT/twins --first_rows OUT/first-rows --out READ/repK
   python studies/recovery_pair/groups.py --scene $SCENE --motion_dir DIR --set READ/rep1 OUT1/first-rows \
       [--set READ/rep2 OUT2/first-rows ...] --out READ/groups.json
   ```

4. **World model error**, on the CPU in float32, on set 1 of a checkpoint's rollouts. The second command reads another
   policy's rollouts, every 4th control, in that policy's own error scale (Table 6).

   ```bash
   python tools/world_model_error.py --run_dir RUN --scene $SCENE --motion_dir DIR --first_rows OUT1/first-rows \
       --placed OUT1/placed --contacts --out ERR/world_model_error.json
   python tools/world_model_error.py --run_dir RUN_MODEL --scene $SCENE --motion_dir DIR --first_rows OUT1/first-rows \
       --placed OUT1/placed --stride 4 --error_scale_checkpoint RUN_POLICY/last.pt --out ERR/world_model_error.json
   ```

The tests `tests/test_world_model_error.py`, `tests/test_world_model_link_inputs.py`, `tests/test_recovery.py` and
`tests/test_recovery_pair_groups.py` run on the CPU on built states and records.

## What the study does not show

- **A cause of ends.** It counts first-row trials that end in get-up windows; it does not relate a trial's end to the
  world model's error at that state.
- **The failure penalty.** No checkpoint of the failure penalty during recovery can be read, so the competing
  explanation named in the Question is not tested on BlinkMimic 2.
- **Small differences.** Two seeds, and one training per condition and seed except the repeats; repeated trainings end
  up to 0.0032 m apart in tracking error.
- **A tracking-only repeat.** The 1024-wide tracking-only counterpart was not trained at this study's revision.
- **Comparable errors.** Errors of different checkpoints are in different error scales. Measured by the mean of the
  squared joint-position scale (the divisor of the squared error), the scales of tracking-only trainings are 22 to 28 %
  above those of single-network recovery trainings at the same seed, width and revision (22 % at width 1024 and seed 0;
  24 and 28 % at the wider width, seeds 0 and 1). Those of 1024-wide trainings are 2.5 to 8.5 % above those of wider
  trainings made beside them at the same seed, with or without link inputs, and 11.3 and 11.5 % for the pair study's
  trainings against the first wider trainings, which come from different revisions.
- **Closed-loop error.** The error is open loop on recorded states, in float32 on the CPU; the trainer runs the world
  model in bfloat16 autocast.
- **Other seeds and lengths.** Seeds beyond 0 and 1, and lengths other than 7,000 iterations.

## Results

**Revisions and cards.**

The trainings and readings ran while the code was being written, at the revisions below. These revisions are not in
this branch's history, which holds the final code in a few commits. The last column says, from a diff restricted to
the code that the row's commands execute (`src`, `scripts`, `tools/rollouts.py`, `tools/world_model_error.py` and the
pair study's `readings.py` and `groups.py`), how each revision differs from this branch's code.

| Revision that ran | What it trained, computed or read | How its code differs from this branch's, in the code those commands execute |
| --- | --- | --- |
| f29424a6666c (the pair study's revision, cited on the [pair-study page](../recovery_pair/README.md), Results, "Conditions") | the pair study's four trainings and their rollouts on RTX 4090; before the world model's construction moved into `networks.py` | Compared through this study's base, whose `src`, `tools` and `scripts` the pair-study page gives as this revision's. Trainer: the world model was built and its error's lower bounds were written inside `train.py`, where `networks.build_world_model` and `networks.world_model_error_floor` now do this with the same arguments and values; every training now also holds a zero failure-penalty weight on the card, checks the failure-penalty option and has a link-input path in the world model that is off without `--wm_link_inputs`, so without the two new options the world model, `args.json` and the training path are as before (CPU tests check the world model's state dict and random draws). Rollouts: none (`Policy` and the rollout tool are unchanged; functions were added to `networks.py` and `recovery.py`). |
| 8ad67d9d324c (in the MoE actor study, `studies/moe_actor`, in a separate pull request of this repository) | tracking-only (seeds 0, 1) and single-network recovery (seed 1), made beside the MoE conditions | The trainer, the simulator environment, the replay buffer and `networks.py` held the MoE actor's code (its option, module, losses and log columns, and a per-episode flag of the mixing target), which does not run without the MoE option; the policy optimizer took its networks from a method that, for a single network, returns the encoder and decoder that this branch passes directly, and `mdp.py` had one more function, which these trainings do not call. These trainings and their rollouts ran without the MoE option, so their results rest on none of this code. Every difference of the first row also applies. |
| bfde109dbe65 | `studies/recovery_pair/groups.py`; the groups reading of the pair study's rollout records; a 500-iteration training of single-network recovery, seed 0, before the world model moved into `networks.py` | Groups reading: its output keys, printed keys and one function had other names, and its docstring, which the program writes into its output as its definitions, was worded differently; the computation is the same, so the values read do not depend on this. Training: `train.py` and `networks.py` were the base's, so every trainer difference of the first row applies. |
| 45f8e648238d | the three failure-penalty trainings from iteration 0; the first error readings of the pair's four world models; the 500-iteration training of single-network recovery, seed 0, after the move | Failure-penalty trainings: the option was spelled differently and took the weight alone, acting from iteration 0 (now `--recovery_fail_penalty 3 0`), and its weight was a number fixed in the captured update, where it is now a tensor on the card set at the run's first iteration and at FIRST_ITERATION, with an option check that now also asks for a whole FIRST_ITERATION of at least 0; with FIRST_ITERATION 0 the weight is 3 at every iteration, as then, and Table 2 comes from that code. Error readings: the tool had another file name and other names for its option, functions, output keys and precision string, did not yet compute the state groups, the stride or another checkpoint's error scale, and loaded the world model without the robot's kinematics; with the defaults the errors follow the same formula, so the values do not depend on these changes, but the output file differs. The 500-iteration training: the world-model builder and the error-bound function had other names, and the link-input path and the zero penalty weight came later, with the same result without the new options. |
| 33125c24b8e6 | the first trainings of the wider world model; the delayed failure penalty (seed 0); the release failure penalty on tracking-only; Table 1; RTX 3090 rollouts of the pair study's checkpoints at 7,000 and of the first wider trainings | Trainings: `--wm_link_inputs` and the world model's link-input path did not exist and the world model was built without the trainer's kinematics, which gives the same world model without that option; the failure penalty's code (option, weight, first iteration) is this branch's, and the rollouts reach no changed code. Error reading (Table 1): the error scale was computed inside the error function, segments started at every control with no stride option, no other checkpoint's error scale could be given, the world model was loaded without the robot's kinematics and the posture splits' output keys had other names; with the defaults the errors are the same, but the output file has other key names and lacks the keys of link inputs, stride and error-scale checkpoint. Of `groups.py`, which the reading imports, only the docstring differs. |
| e692ba9baac8 | every other training of Tables 3 and 9 (1024-wide single-network recovery, link inputs, second trainings of the wider world model, wider with link inputs, tracking-only wider); their rollouts and readings; Table 4 and Table 8 | Trainings: one sentence of `--wm_link_inputs`'s help text and `g1_constants.SUPPORT_LINKS`, which no code reads, came later (these trainings were given the same 18 names on the command line), and the code they execute is otherwise this branch's; the rollouts reach no changed code, and of the groups reading only the docstring differs, which it writes into its output as its definitions. Error readings (Tables 4 and 8): segments started at every control with no stride option, the error scale could not be taken from another checkpoint, and the output lacked the keys of stride and error-scale checkpoint. With the defaults the errors are the same. |
| de06c97b28ec | Table 6 (with `--stride` and `--error_scale_checkpoint`); the readings of the checkpoints at 5,500 to 6,500 | Only the docstring of `studies/recovery_pair/groups.py`, which the groups reading writes into its output as its definitions, was worded differently; no computation differs. |

Every training of this study ran on one RTX 4090 with 48 GB; rollouts on one RTX 3090 except where stated; readings on
the CPU. The pair study's trainings ran at a 350 W card power limit and the others at 450 W. After the world model's
construction moved into `networks.py`, CPU tests check that the options being absent leave the world model's state dict
and random stream, `args.json` and the training path as before; the full suite of 110 tests passed at the last revision
of the table. Every training completed its 7,000 iterations except the two 500-iteration trainings of single-network
recovery at seed 0, run before and after the world model's construction moved into `networks.py`, and the following (the
release failure penalty on tracking-only ran 2,500 by design). The three failure-penalty trainings from iteration 0
ended with the trainer's exit code 139 (core dump) before iteration 600. The delayed failure-penalty training was
cancelled by hand after the log row at iteration 3,100. Seed 1 of the delayed penalty was not trained. No
world model error was read on the failure-penalty trainings or on the checkpoints at 5,500 to 6,500.

**Table 1** reads the world model error of the four world models of the pair study, in first-row trials, by posture,
after 1 / after 8 controls (normalized, unitless; each world model's own error scale), on the pair study's RTX 4090
rollouts at 7,000 iterations, with the ratio (low-posture over upright-posture) and the split by non-foot contact.

| Training | Seed | Upright-posture | Low-posture | Reference get-up windows | Non-foot contact | No non-foot contact | Ratio after 1 | Ratio after 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tracking-only | 0 | 0.0143 / 0.081 | 0.0800 / 0.652 | 0.0622 / 0.440 | 0.0843 / 0.678 | 0.0146 / 0.085 | 5.61 | 8.01 |
| tracking-only | 1 | 0.0136 / 0.076 | 0.0843 / 0.775 | 0.0615 / 0.450 | 0.0876 / 0.807 | 0.0140 / 0.081 | 6.22 | 10.19 |
| single-network recovery | 0 | 0.0172 / 0.099 | 0.0977 / 1.107 | 0.0875 / 0.772 | 0.0992 / 1.147 | 0.0178 / 0.105 | 5.67 | 11.20 |
| single-network recovery | 1 | 0.0167 / 0.101 | 0.0956 / 1.029 | 0.0730 / 0.673 | 0.0978 / 1.042 | 0.0172 / 0.108 | 5.73 | 10.20 |

**Table 1b** gives the share of the low-posture error after 8 controls (first-row trials) carried by each state group,
computed from the group means rounded to two decimals, so that a row may not sum to 1; the joint-position share on
upright-posture states, from the unrounded means, is 0.59, 0.59, 0.61 and 0.59:

| Training | Seed | Root position | Root rotation | Root linear velocity | Root angular velocity | Joint positions | Joint velocities |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tracking-only | 0 | 0.03 | 0.09 | 0.04 | 0.01 | 0.70 | 0.13 |
| tracking-only | 1 | 0.02 | 0.08 | 0.03 | 0.01 | 0.73 | 0.12 |
| single-network recovery | 0 | 0.02 | 0.08 | 0.03 | 0.01 | 0.75 | 0.11 |
| single-network recovery | 1 | 0.02 | 0.09 | 0.03 | 0.01 | 0.74 | 0.11 |

**Outcome: the rule on the world model error by posture.** The first condition holds for all four world models: the
error on low-posture states is 8.01 to 11.20 times the error on upright-posture states after 8 controls (5.61 to 6.22
after 1 control). The second condition could not be read: the training of the failure penalty during recovery was
cancelled at iteration 3,100, so it has no last checkpoint to roll out (and, as only a last checkpoint stores the world
model, no world model to read), and the rule gives no outcome. The comparison is kept on the recovery benchmark; this
choice was made after these readings. The competing explanation was read on NablaMimic's trainer, outside this page,
in the NablaMimic fork's MoE actor study
([`studies/moe_actor`](https://github.com/LingzheZhao/NablaMimic/blob/moe-actor-study/studies/moe_actor/README.md),
single-network recovery without the termination penalty): a single-network recovery training made without
NablaMimic's termination penalty completes, at both seeds, all six get-up clips that BlinkMimic 2 excludes from its
commands, and none of its first-row trials ends in a get-up window. This page holds no data for that reading.

**Table 2** gives the failure-penalty trainings: the form, how far it ran, what ended it, and the simulator return
`sim_rew` of its last log row against the pair study's single-network recovery seed 0 at the same iteration.

| Form | Seed | Last log iteration | What ended it | `sim_rew`, with / pair study |
| --- | --- | --- | --- | --- |
| from iteration 0, critic targets include it | 0 | 300 | exit code 139, no message | -0.261582 / 0.679369 |
| from iteration 0, critic targets include it | 1 | 400 | exit code 139, no message | -0.404194 / 0.673862 |
| from iteration 0, inside the imagined rollout only | 0 | 500 | exit code 139; log: `nefc overflow - please increase njmax to 1699` | 0.365632 / 0.653568 |
| from iteration 2,000, inside the imagined rollout only | 0 | 3,100 | training cancelled; no `last.pt`, so no world model | 0.557322 / 0.720809 |
| release failure penalty on tracking-only (not a declared condition) | 0 | 2,500 | completed; against the pair study's tracking-only | 0.697488 / 0.728356 |

**Table 2b** gives, for the delayed penalty, the log against the pair study's single-network recovery seed 0 (with /
without). `imag_rew` and `term_frac` are columns of the trainer's training log (named in
[RECOVERY.md](../../RECOVERY.md)); `term_frac` is the share of ended episodes that ended in a termination.

| Iteration | `sim_rew` | `term_frac` | `wm_err8` |
| --- | --- | --- | --- |
| 2000 | 0.677482 / 0.681184 | 0.220828 / 0.21072 | 1.2274 / 1.20389 |
| 2200 | 0.607039 / 0.678705 | 0.386298 / 0.200826 | 1.62508 / 1.02051 |
| 3000 | 0.471229 / 0.725709 | 0.373585 / 0.137931 | 2.27356 / 0.670738 |
| 3100 | 0.557322 / 0.720809 | 0.331215 / 0.141118 | 2.23196 / 0.644677 |

**Table 3** gives the trainings of single-network recovery by world model width and inputs, at iteration 7,000 on RTX
3090 rollouts, three sets each: the tracking error over the 34 commanded clips (m; mean of the three sets), the
commanded clips completed per set, the excluded clips' E (m; mean of sets) and their completion per set, R per set, ends
in get-up windows / windows entered, summed over the three sets, for three groups of clips in this order (the five clips
that give the hidden references; fallAndGetUp1_subject4; the commanded clips with a get-up), seconds per training
iteration and peak card memory (MiB).

| Training | Seed | Tracking error | Commanded completed | Excluded: E; completed | R | Ends / windows | s per iteration | MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024-wide, the pair study's | 0 | 0.03430 | 34, 34, 34 | 0.0568; 4, 4, 4 | 99, 99, 99 | 4/241; 2/49; 0/129 | 0.1343 | 13672 |
| 1024-wide, the pair study's | 1 | 0.03576 | 34, 34, 34 | 0.0551; 5, 5, 6 | 97, 97, 97 | 2/251; 0/51; 0/129 | 0.1344 | 12606 |
| 1024-wide, beside the MoE conditions | 1 | 0.03576 | 32, 34, 34 | 0.0577; 6, 6, 5 | 98, 98, 98 | 1/253; 0/51; 0/129 | 0.1236 | 12670 |
| 1024-wide, beside the world-model conditions | 0 | 0.03424 | 34, 34, 34 | 0.0536; 5, 5, 5 | 99, 99, 99 | 3/252; 0/51; 0/129 | 0.1228 | 12670 |
| 1024-wide, beside the world-model conditions | 1 | 0.03265 | 33, 33, 33 | 0.0541; 5, 6, 6 | 99, 99, 99 | 1/271; 0/51; 0/126 | 0.1211 | 23310 |
| link inputs, 1024-wide | 0 | 0.03477 | 34, 34, 33 | 0.0557; 6, 4, 6 | 99, 99, 99 | 2/235; 0/51; 0/129 | 0.1498 | 13012 |
| link inputs, 1024-wide | 1 | 0.03328 | 33, 34, 32 | 0.0528; 6, 6, 6 | 99, 99, 99 | 0/273; 0/51; 1/116 | 0.1507 | 13378 |
| wider, first training | 0 | 0.03201 | 34, 34, 34 | 0.0515; 6, 6, 6 | 99, 99, 99 | 0/273; 0/51; 0/129 | 0.1597 | 13872 |
| wider, first training | 1 | 0.03344 | 34, 33, 34 | 0.0528; 6, 5, 6 | 98, 98, 98 | 1/270; 0/51; 0/128 | 0.1613 | 13872 |
| wider, second training | 0 | 0.03518 | 34, 34, 34 | 0.0543; 6, 6, 6 | 99, 99, 99 | 0/273; 0/51; 0/129 | 0.1632 | 13872 |
| wider, second training | 1 | 0.03418 | 34, 33, 34 | 0.0549; 4, 5, 5 | 98, 98, 98 | 4/242; 0/51; 0/129 | 0.1615 | 13872 |
| wider with link inputs | 0 | 0.03343 | 34, 34, 33 | 0.0521; 6, 6, 6 | 99, 99, 99 | 0/273; 0/51; 0/125 | 0.1899 | 14106 |
| wider with link inputs | 1 | 0.03343 | 33, 33, 34 | 0.0536; 6, 6, 6 | 99, 99, 99 | 0/273; 0/51; 0/125 | 0.1881 | 20338 |

In 1024-wide beside the world-model conditions at seed 1 and in wider with link inputs at seed 1, one 5-second sample
(23310 and 20338 MiB) lies above a next-largest sample of 12670 and 14106 MiB; in the pair study's seed 0 and in link
inputs, 1024-wide, seed 1, the largest sample is 1066 and 362 MiB above the next-largest. The table gives the largest
sample. The training made beside the MoE conditions at seed 0 is not read here.

**Exploratory (added after results).** The first trainings of the wider world model read 0.0320 and 0.0334 m, below the
pair study's 0.0343 and 0.0358 m, and a lower tracking error of the wider world model at both seeds was stated. That
statement is withdrawn: the second trainings read 0.0352 and 0.0342 m against 0.0342 and 0.0327 m for the 1024-wide
single-network recovery trained beside them, and the excluded clips completed and the ends in get-up windows do not
separate the two widths. The wider world model takes 0.160 to 0.163 s per iteration against 0.121 to 0.124 s for the
1024-wide single-network recovery trainings without link inputs at 450 W (0.134 s for the pair study's at 350 W). Link
inputs read 0.0348 and 0.0333 m at 1024 width and 0.0334 m at both seeds with the wider world model.

**Table 4** reads the world model error of the trainings of Table 3, in first-row trials, by posture, after 1 / after 8
controls, with the ratio after 8 controls, and on the low-posture states of placed starts (normalized, unitless; each
checkpoint's own error scale), on set 1 of the RTX 3090 rollouts at 7,000. The training made beside the MoE conditions
was not read. The values for the pair study's trainings differ from Table 1, which reads RTX 4090 rollouts.

| Training | Seed | Upright-posture | Low-posture | Ratio after 8 | Placed starts, low-posture |
| --- | --- | --- | --- | --- | --- |
| 1024-wide, the pair study's | 0 | 0.0171 / 0.097 | 0.0965 / 1.054 | 10.84 | 0.1542 / 1.213 |
| 1024-wide, the pair study's | 1 | 0.0167 / 0.101 | 0.0961 / 1.045 | 10.36 | 0.0969 / 0.672 |
| 1024-wide, beside the world-model conditions | 0 | 0.0171 / 0.098 | 0.0933 / 0.993 | 10.17 | 0.1110 / 0.704 |
| 1024-wide, beside the world-model conditions | 1 | 0.0167 / 0.101 | 0.0962 / 1.031 | 10.18 | 0.1187 / 0.801 |
| link inputs, 1024-wide | 0 | 0.0148 / 0.097 | 0.0916 / 1.050 | 10.86 | 0.1111 / 0.824 |
| link inputs, 1024-wide | 1 | 0.0141 / 0.083 | 0.0871 / 0.904 | 10.85 | 0.1076 / 0.743 |
| wider, first training | 0 | 0.0136 / 0.069 | 0.0785 / 0.706 | 10.30 | 0.1189 / 0.715 |
| wider, first training | 1 | 0.0136 / 0.069 | 0.0786 / 0.697 | 10.08 | 0.0889 / 0.657 |
| wider, second training | 0 | 0.0130 / 0.067 | 0.0841 / 0.834 | 12.43 | 0.0989 / 0.703 |
| wider, second training | 1 | 0.0140 / 0.071 | 0.0857 / 0.837 | 11.81 | 0.0927 / 0.700 |
| wider with link inputs | 0 | 0.0119 / 0.073 | 0.0773 / 0.673 | 9.27 | 0.0881 / 0.525 |
| wider with link inputs | 1 | 0.0123 / 0.068 | 0.0767 / 0.724 | 10.69 | 0.0941 / 0.666 |

**Exploratory (added after results).** What holds in every training is the lower error of the wider world model on both
posture classes, with no overlap of ranges after 1 or after 8 controls. The ratio after 8 controls is 9.27 to 12.43 in
all twelve trainings; it is 12.43 and 11.81 in the second trainings of the wider world model, above the 10.17 to 10.86
of the 1024-wide trainings.

**Table 5** gives the logs of two trainings of one seed with identical arguments (first / second): the logged simulator
return `sim_rew`, `imag_rew`, the world model error `wm_err8` (trainer's normalized units) and `term_frac`, at
iterations 100, 1000 and 7000.

| Pair of trainings | Iteration | `sim_rew` | `imag_rew` | `wm_err8` | `term_frac` |
| --- | --- | --- | --- | --- | --- |
| wider, seed 0 (first / second) | 100 | 0.700 / 0.702 | 0.669 / 0.672 | 1.757 / 1.766 | 1.00 / 1.00 |
| wider, seed 0 | 1000 | 0.611 / 0.624 | 0.637 / 0.656 | 0.663 / 0.607 | 0.56 / 0.55 |
| wider, seed 0 | 7000 | 0.798 / 0.800 | 0.885 / 0.882 | 0.267 / 0.257 | 0.04 / 0.04 |
| wider, seed 1 | 100 | 0.691 / 0.699 | 0.660 / 0.665 | 1.816 / 1.823 | 1.00 / 1.00 |
| wider, seed 1 | 1000 | 0.661 / 0.625 | 0.682 / 0.653 | 0.639 / 0.608 | 0.53 / 0.57 |
| wider, seed 1 | 7000 | 0.796 / 0.795 | 0.879 / 0.877 | 0.259 / 0.268 | 0.05 / 0.05 |
| 1024-wide, seed 0 (pair study's / beside world-model) | 100 | 0.696 / 0.695 | 0.662 / 0.659 | 2.413 / 2.401 | 1.00 / 1.00 |
| 1024-wide, seed 0 | 1000 | 0.602 / 0.630 | 0.640 / 0.662 | 0.775 / 0.711 | 0.55 / 0.56 |
| 1024-wide, seed 0 | 7000 | 0.784 / 0.782 | 0.868 / 0.869 | 0.386 / 0.352 | 0.05 / 0.06 |
| 1024-wide, seed 1 (pair study's / beside world-model) | 100 | 0.704 / 0.700 | 0.669 / 0.665 | 2.440 / 2.442 | 1.00 / 1.00 |
| 1024-wide, seed 1 | 1000 | 0.615 / 0.627 | 0.652 / 0.656 | 0.672 / 0.677 | 0.58 / 0.56 |
| 1024-wide, seed 1 | 7000 | 0.780 / 0.795 | 0.867 / 0.875 | 0.392 / 0.370 | 0.06 / 0.05 |

**Exploratory (added after results).** Single-network recovery with the 1024-wide world model has been trained three
times at seed 1 and, among the trainings read here, twice at seed 0 (0.0343 and 0.0342 m); at seed 1 the three read
0.0358, 0.0358 and 0.0327 m. The wider world model, trained twice, reads 0.0320 and 0.0352 m at seed 0 and 0.0334 and
0.0342 m at seed 1. At iteration 100 the logged values of a pair differ by at most 0.012 (`wm_err8` 2.413 and 2.401,
1.757 and 1.766, 2.440 and 2.442); at iteration 1000 by up to 0.036 in `sim_rew` and 0.064 in `wm_err8`. Trainings are
not repeatable bit for bit. Some repeated trainings end within 0.0001 m of each other and others 0.0007 to 0.0032 m
apart, with 25 to 32 of the 34 clips moving the same way; why is not known. At iteration 7,000 the learning rate is
still 0.998 of its starting value, because the preset's schedule decays over 250,000 iterations.

**Outcome: differences below 0.0032 m.** Compared one training against one training, every tracking error of the wider
world model without link inputs differs from every 1024-wide single-network recovery training without link inputs of the
same seed by less than 0.0032 m (the largest difference is 0.0023 m); each such comparison rests on one training per
condition, so by the quoted sentence none is evidence of a difference. That sentence was written after these trainings
had been read.

**Table 6** (6a, then 6b) gives the cross-policy reading on low-posture states and then upright-posture states, error
after 8 controls. Each cell reads the world model of its row on the first-row rollouts of the policy of its column,
every 4th control. The error is divided by the error scale of the world model trained with the column's policy, so the
cells of one column share a unit and the cells of different columns do not. The diagonal, a world model on the rollouts
of its own policy, is in bold. The policies are 1024-wide (the training made beside the world-model conditions) and the
second trainings of the wider world model.

| 6a, low-posture, after 8: world model \ policy | 1024-wide seed 0 | 1024-wide seed 1 | wider seed 0 | wider seed 1 |
| --- | --- | --- | --- | --- |
| 1024-wide seed 0 | **0.988** | 1.581 | 1.681 | 1.585 |
| 1024-wide seed 1 | 1.473 | **1.037** | 1.684 | 1.573 |
| wider seed 0 | 1.202 | 1.336 | **0.841** | 1.242 |
| wider seed 1 | 1.175 | 1.302 | 1.431 | **0.836** |

| 6b, upright-posture, after 8: world model \ policy | 1024-wide seed 0 | 1024-wide seed 1 | wider seed 0 | wider seed 1 |
| --- | --- | --- | --- | --- |
| 1024-wide seed 0 | **0.098** | 0.111 | 0.114 | 0.123 |
| 1024-wide seed 1 | 0.126 | **0.101** | 0.122 | 0.130 |
| wider seed 0 | 0.087 | 0.082 | **0.067** | 0.088 |
| wider seed 1 | 0.085 | 0.081 | 0.081 | **0.071** |

**Exploratory (added after results).** On low-posture states, in every column the diagonal entry is the lowest. On
upright-posture states it is the lowest in the wider columns; in the 1024-wide columns the wider world models read lower
than the diagonal (0.087 and 0.085 against 0.098; 0.082 and 0.081 against 0.101). Off the diagonal, the wider world
models read lower than the 1024-wide ones in every column on both classes. The placed-start panels are not given: this
reading takes every 4th control, so the segments that start at the control of placement make up four times the share
they have in a full reading.

**Table 7** gives the checkpoint-to-checkpoint reading: the tracking error over the 34 commanded clips (m), with in
parentheses the commanded clips completed and the number of the 34 clips whose E is higher than at the checkpoint 500
iterations earlier, at 5,500, 6,000, 6,500 (one set each) and 7,000 (set 1, then the mean of three sets), and the range
of the four checkpoints (set 1 at 7,000).

| Training | Seed | 5,500 | 6,000 | 6,500 | 7,000, set 1 | 7,000, mean | Range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024-wide, the pair study's | 1 | 0.03589 (34) | 0.03580 (34; 18) | 0.03501 (34; 7) | 0.03586 (34; 30) | 0.03576 | 0.00088 |
| 1024-wide, beside the MoE conditions | 1 | 0.03659 (34) | 0.03656 (33; 13) | 0.03528 (34; 5) | 0.03589 (32; 23) | 0.03576 | 0.00131 |
| 1024-wide, beside the world-model conditions | 1 | 0.03401 (33) | 0.03337 (33; 7) | 0.03375 (31; 7) | 0.03265 (33; 15) | 0.03265 | 0.00136 |
| wider, first training | 0 | 0.03354 (34) | 0.03333 (34; 16) | 0.03241 (34; 7) | 0.03200 (34; 12) | 0.03201 | 0.00153 |
| wider, second training | 0 | 0.03568 (34) | 0.03466 (34; 3) | 0.03477 (33; 18) | 0.03519 (34; 24) | 0.03518 | 0.00102 |

**Outcome: the checkpoint reading.** The range of the four checkpoints is 0.00088, 0.00131, 0.00136, 0.00153 and 0.00102
m in the five trainings; it is 0.001 m or more in four of them, so a supplementary reading is added. The rule was
written after the wider world model's trainings had been read. The tracking error moves by up to 0.0013 m between
checkpoints 500 iterations apart, and trainings of one seed end 0.0013 to 0.0032 m apart at each of the four
checkpoints (set 1 at 7,000): 0.0026, 0.0032, 0.0015 and 0.0032 m between the highest and lowest of the three 1024-wide
single-network recovery trainings at seed 1, and 0.0021, 0.0013, 0.0024 and 0.0032 m between the two wider trainings at
seed 0; so a mean over checkpoints does not remove the difference between repeated trainings.

**Table 7b** gives the supplementary reading of every training of Table 3: the tracking error (m) at 6,000 and 6,500
(one set each), at 7,000 (mean of three sets) and the mean of the three.

| Training | Seed | 6,000 | 6,500 | 7,000 | Supplementary reading |
| --- | --- | --- | --- | --- | --- |
| 1024-wide, the pair study's | 0 | 0.03460 | 0.03452 | 0.03430 | 0.03447 |
| 1024-wide, the pair study's | 1 | 0.03580 | 0.03501 | 0.03576 | 0.03553 |
| 1024-wide, beside the MoE conditions | 1 | 0.03656 | 0.03528 | 0.03576 | 0.03587 |
| 1024-wide, beside the world-model conditions | 0 | 0.03461 | 0.03400 | 0.03424 | 0.03428 |
| 1024-wide, beside the world-model conditions | 1 | 0.03337 | 0.03375 | 0.03265 | 0.03326 |
| link inputs, 1024-wide | 0 | 0.03510 | 0.03561 | 0.03477 | 0.03516 |
| link inputs, 1024-wide | 1 | 0.03393 | 0.03419 | 0.03328 | 0.03380 |
| wider, first training | 0 | 0.03333 | 0.03241 | 0.03201 | 0.03258 |
| wider, first training | 1 | 0.03427 | 0.03459 | 0.03344 | 0.03410 |
| wider, second training | 0 | 0.03466 | 0.03477 | 0.03518 | 0.03487 |
| wider, second training | 1 | 0.03458 | 0.03457 | 0.03418 | 0.03444 |
| wider with link inputs | 0 | 0.03388 | 0.03342 | 0.03343 | 0.03358 |
| wider with link inputs | 1 | 0.03504 | 0.03463 | 0.03343 | 0.03437 |

**Exploratory (added after results).** On the supplementary reading the two widths are not separated. At seed 1 the two
trainings of the wider world model (0.03410, 0.03444 m) lie inside the range of the three 1024-wide trainings (0.03326
to 0.03587 m). At seed 0 the first wider training (0.03258 m) is below and the second (0.03487 m) above the two
1024-wide trainings (0.03428 and 0.03447 m).

**Table 8** gives the separation of link inputs from width: the world model error of Table 4 (normalized, unitless; each
checkpoint's own scale), after 1 and after 8 controls, as the range over the trainings without link inputs (four at each
width) and the two values with them (seeds 0, 1), and whether the two values lie below that range or the range of the
two overlaps it.

| Reading | 1024-wide, without | 1024-wide, with | Relation | Wider, without | Wider, with | Relation |
| --- | --- | --- | --- | --- | --- | --- |
| upright-posture, after 1 | 0.0167 to 0.0171 | 0.0148, 0.0141 | below | 0.0130 to 0.0140 | 0.0119, 0.0123 | below |
| low-posture, after 1 | 0.0933 to 0.0965 | 0.0916, 0.0871 | below | 0.0785 to 0.0857 | 0.0773, 0.0767 | below |
| upright-posture, after 8 | 0.0972 to 0.1013 | 0.0967, 0.0834 | below | 0.067 to 0.071 | 0.073, 0.068 | overlaps |
| low-posture, after 8 | 0.993 to 1.054 | 1.050, 0.904 | overlaps | 0.697 to 0.837 | 0.673, 0.724 | overlaps |

**Exploratory (added after results).** With link inputs the error after 1 control is below the range of the trainings
without them on both posture classes at both widths. After 8 controls it is below that range on upright-posture states
at 1024 width (0.0967 and 0.0834 against 0.0972 to 0.1013) and overlaps it in the other three readings. A lower error on
low-posture states after 8 controls with link inputs is therefore not established. The tracking error with link inputs
(Table 3) is 0.0348 and 0.0333 m at 1024 width and 0.0334 m at both seeds with the wider world model.

**Table 9** gives tracking-only with the wider world model against the 1024-wide tracking-only (made beside the MoE
conditions of the MoE actor study), on the quantities of Table 3 and the supplementary reading (m), and then the
world model error as in Table 4 (9b; normalized, each checkpoint's own scale, which differs from that of single-network
recovery).

| Training | Seed | Tracking error | Commanded completed | Excluded: E; completed | R | Ends / windows | s per iteration | MiB | Supplementary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024-wide tracking-only | 0 | 0.02744 | 34, 34, 34 | 0.0403; 6, 6, 6 | 95, 91, 92 | 0/273; 0/51; 0/129 | 0.1173 | 12250 | 0.02769 |
| 1024-wide tracking-only | 1 | 0.02732 | 34, 33, 33 | 0.0392; 6, 6, 6 | 64, 69, 65 | 0/273; 0/51; 0/127 | 0.1177 | 12250 | 0.02778 |
| wider tracking-only | 0 | 0.02622 | 34, 34, 34 | 0.0371; 6, 6, 6 | 95, 96, 97 | 0/273; 0/51; 0/129 | 0.1587 | 13360 | 0.02672 |
| wider tracking-only | 1 | 0.02641 | 34, 34, 34 | 0.0378; 6, 6, 6 | 63, 67, 62 | 0/273; 0/51; 0/129 | 0.1569 | 13360 | 0.02673 |

| Training | Seed | Upright-posture | Low-posture | Ratio after 8 | Placed starts, low-posture |
| --- | --- | --- | --- | --- | --- |
| 1024-wide tracking-only | 0 | 0.0146 / 0.084 | 0.1021 / 0.989 | 11.74 | 2.6431 / 23.069 |
| 1024-wide tracking-only | 1 | 0.0142 / 0.079 | 0.0827 / 0.709 | 8.93 | 1.2239 / 12.124 |
| wider tracking-only | 0 | 0.0118 / 0.057 | 0.0734 / 0.543 | 9.60 | 1.9301 / 18.603 |
| wider tracking-only | 1 | 0.0113 / 0.052 | 0.0743 / 0.586 | 11.18 | 1.0835 / 11.401 |

**Exploratory (added after results).** The wider tracking-only is 0.0012 m (seed 0) and 0.0009 m (seed 1) below the
1024-wide one at the last checkpoint, and 0.0010 m (0.00097 and 0.00105) below on the supplementary reading. Its world
model error is lower on both classes after 1 and after 8 controls at both seeds. The two trainings of a seed come from
different revisions, one training each, and no repeat of tracking-only at one seed sizes the spread; by the sentence
quoted above, a difference of 0.0012 m or less is not evidence of a difference.

**What the results do not show.**

- A test of the competing explanation named in the Question: the failure penalty during recovery produced no readable
  checkpoint.
- A world model error by posture for the training made beside the MoE conditions and for the checkpoints at 5,500 to
  6,500.
- A rate or an interval for ends in get-up windows: the counts are summed over three non-independent sets.
- The seed 0 training of single-network recovery made beside the MoE conditions, which is not read here.
- A comparison of errors across different error scales, apart from the columns of Table 6.
- A comparison of speed across cards or power limits: the pair study's trainings ran at 350 W and the others at 450 W.
