# Recovery from placed lying starts: a pair of trainings

**Question.** Does recovery training from placed lying starts against a hidden get-up reference
([RECOVERY.md](../../RECOVERY.md)) give a BlinkMimic 2 policy that gets up from placed lying starts and rejoins its
command? Does that policy keep tracking the 40 LAFAN1 clips, against the same recipe without recovery?

The [NablaMimic](https://github.com/LingzheZhao/NablaMimic) fork asks the same question of its PyTorch trainer in its
study [`studies/recovery`](https://github.com/LingzheZhao/NablaMimic/blob/369a2833050b/studies/recovery/README.md)
(revision 369a2833050b). This study reads this trainer's rollouts by the same rules, so that the two trainers can be
set side by side: [`readings.py`](readings.py) builds the arrays of that study's `readings_core.py` from the records of
[`tools/rollouts.py`](../../tools/rollouts.py) and calls it, from a checkout of that revision given as
`--readings-core`. The definitions of the reading rules (support, fall rule, get-up, rejoin, R, J, F, C, C_getup, E)
are in that study's README, "Readings"; the declaration below states what is this study's own. In short (pair.json,
"quantities"): **R** counts the 100 placed starts that get up, **J** those that rejoin their command, **J over R** is
the share of the starts that got up with a readable rejoin that rejoin, **F** counts the 100 twins with a counted fall,
**C** the clips of the 40 completed from their first row, **C_getup** states whether each hidden get-up clip is
completed, and **E** is the aligned body-position error in metres.

The study trains a control and a recovery policy at two seeds with `python -m blinkmimic2.train`, rolls out each save
with `tools/rollouts.py`, reads the rollouts and applies the outcome order. As in RECOVERY.md, a **control** is one
0.02 s policy step, a **row** one 0.02 s frame of the motion library, and the **command** the row the policy reads. The
declaration is [pair.json](pair.json), in the structure of the NablaMimic study's `pair.json`: the same top-level keys,
the same fall and recovery values, margins, classes and twelve outcome rows, with the saves at 3,500 and 7,000
iterations in place of 1,500 and 3,000 updates and the rows' texts reworded where they name a trainer's parts. Every
program reads it through `declaration.py`, which refuses a declaration with other top-level keys.

## Declaration

### Trainings and saves

| | |
| --- | --- |
| Conditions | control (`PRESET[recovery_control]` of [`scripts/presets.sh`](../../scripts/presets.sh)) and recovery (`PRESET[recovery]`: the control plus the seven recovery options), each with `--seed SEED --bones_list= --motion_dir DIR`, `DIR` the folder of the 40 LAFAN1 G1 retarget CSVs |
| Seeds | 0 and 1. Each training seeds torch with its seed (`--seed`) |
| Length | 7,000 iterations with a numbered checkpoint every 100, the presets' own values, and a log row every 100 (`--log_every`'s default) |
| Card | chosen at launch, one card model for the four trainings and the 24 rollouts; `args.json` does not record it, `description.json` of each rollout does |
| Saves | the averaged (EMA) policy `pi` of the numbered checkpoint `ckpt_NNNNNN.pt` at 3,500 and 7,000 iterations, rolled out with deterministic actions. A training gives two saves and the pair eight |

Each save is read on its own; nothing is pooled across saves or seeds. The unit of repetition is the seed, two per
condition. The two saves of one training are not repetitions.

**Completion.** This trainer writes no record of a run's end. A training completed its 7,000 iterations when its
checkpoint `ckpt_007000.pt` exists, which `train.py` writes at the last iteration, and the last row of its
`train_log.csv` is at iteration 7,000. `train.py` writes the log header at every start and has no resumption, so a log
that holds a second header, of a second start in the same directory, is refused.

### Scene

`scene.py` writes the scene files of `tools/rollouts.py` (ROLLOUTS.md, "Scene") on the library of a recovery training
of the pair: its `args.json` names the motion folder and the library inputs, and the program builds the library from
them as `tools/rollouts.py` does, on the CPU. It refuses a training whose recovery options are not those of
`PRESET[recovery]`, a library of other than 40 clips, and row counts other than the training's `recovery.json`.

- **Placed lying starts.** 100 trials, the same at every save.
  - *Body rows* are evenly spaced over the start rows of `recovery.build` with the settings of `PRESET[recovery]`.
  - *Command rows* are evenly spaced over the command rows of the same build with 1,000 upright rows ahead in place of
    COMMAND_UPRIGHT's 500: rows outside the command-excluded clips that, with their next 1,000 rows, are upright and
    lie before their clip's last row. A 20 s rollout therefore meets neither a clip end nor a low command.
  - *Evenly spaced* means the rounded points of a float64 linspace over the row indices.
  - *Placement.* One torch CPU generator with seed 7 draws unit uniforms in [-1, 1) in this order: position [100, 3],
    rotation [100, 3], velocity [100, 6] and joints [100, 29]. The same generator then draws a permutation that pairs
    the command rows with the body rows, so a get-up recording and a command clip are not paired by library order. The
    uniforms scale the reset noise's half-widths (ROLLOUTS.md, "Scene"), which are the values of training's reset.
- **Twins.** 100 trials at the same command rows, each started at its command row alone.
- **First rows.** One trial at the first row of each of the 40 clips.

`scene.json` holds the recovery settings, the six row counts of the build, the count of command rows with 1,000
upright rows ahead, `first_row_controls`, every clip with its first and last row, every trial with its rows and clips,
the training it was built from, the robot and the library inputs.

### Rollouts

Every trial starts through `TrackingSim.reset`, which clamps the joints to the joint ranges of the model, and runs in
the evaluation-mode simulator: MuJoCo Warp through mjlab, with mjlab's G1 and its collision model, 128 contacts and
500 constraint rows per world, randomization, reset noise and observation noise off ([ROLLOUTS.md](../../ROLLOUTS.md),
"Environment and policy"). Placed starts and twins run 1,000 controls (20 s) under `--termination nonfinite`, so only
a non-finite state ends a trial. First rows run under `--termination recipe` for the longest clip's rows minus one,
which `scene.py` prints. Each clip is read over its own rows minus one.

Trials are not repeatable one by one here (ROLLOUTS.md, "Repeatability"): the contact order of MuJoCo Warp and the
policy's bf16 inference make two trials of the same start drift apart from the first control. A twin is therefore a
reference run of its placed start's command from the command row, not a copy that a repeat would reproduce.

### Readings

`readings.py` reads one save from its three rollouts. It refuses a rollout without `rollouts.npz`, and a record whose
scene, termination, control count, control period, robot or library inputs are not those of this study's scene and
declaration, whose command or body rows are not its scene file's, or that is of another run, checkpoint or body list
than the save's placed starts. That save then has no reading. The declaration's `actor_key`, `pi`, is not compared with
the records: `tools/rollouts.py` always loads the checkpoint's `pi`. The record already holds the core's arrays under
the core's names; the program hands it those 14 arrays, the declaration's `fall_recovery` section, the control period of
0.02 s, the indices of `pelvis` (0) and `torso_link` (7) among the 14 tracked bodies as `description.json` gives them,
the clip names and first rows of `scene.json`, and the five hidden clips. No model, library or simulator is loaded. It
writes `save_readings.json`, which names the resolved path of the `readings_core.py` that read it (`readings_core`), and
`trials.jsonl`, with the keys of the NablaMimic study's entry but for those that "Differences from the NablaMimic study"
lists.

**Exposure.** `outcome.py` gives, beside the outcome, the imagined tracking-regime controls of each training at 3,500
and 7,000 iterations (RECOVERY.md, "Records"): `recovery_imagined_tracking_total` of the log row at that iteration for
a recovery training, the iterations times `pi_batch` times `horizon` (65,536) for a control training. For each
recovery save it gives the first control iteration whose total reaches it. No outcome row reads the matched iteration.

### Classes, margins and outcomes

The classes, the margins and the outcome order are those of the NablaMimic study's README ("Classes, margins and
outcomes") and `pair.json`, with 3,500 and 7,000 iterations in place of 1,500 and 3,000 updates. `outcome.py` holds
the six rule functions of that study's `outcome.py` (`save_values`, `margins`, `classes`, `tracking_within`,
`outcome_rules`, `outcome`), written out because that file imports the NablaMimic package at its top; they are the
same but for one docstring. Row 1, `run_stopped`, cannot hold here: this trainer does not read MuJoCo Warp's capacity
during training (RECOVERY.md, "Not implemented"), and a training that ends early makes the pair unreadable.

**Unreadable.** No outcome is given, and the reasons are written instead, when any of these holds:

- a declared save has no reading, or its E is unavailable because a clip has no body-position error;
- a training's `args.json` names another seed, other iterations, another checkpoint interval or another log interval
  than declared, or
  recovery options that are not those of its condition's preset (none for the control);
- the two trainings of a seed have `args.json` files that differ outside the seven recovery options and `log_dir`.
  Every other argument counts, `bones_lib` among them, whose default is a path inside the checkout that runs
  `train.py`, so the four trainings run from one checkout path. `args.json` does not hold the card, the source
  revision or the environment, so these are not compared; Results state them;
- a training did not complete its 7,000 iterations;
- the saves were not all read by one `readings_core.py`.

## Differences from the NablaMimic study

- **Trainer.** This trainer learns a model of the robot's dynamics and trains the policy and critic on 32-control
  rollouts imagined in it; the recovery setting differs from NablaMimic's as RECOVERY.md, "Differences from the
  NablaMimic setting", gives.
- **Simulator.** MuJoCo Warp through mjlab with mjlab's G1 collision model (capsule feet, self-collisions on, mjlab's
  solver settings), against NablaEngine. A trial's contacts beyond the pooled contact array are flagged on every world
  of the batch, not on the world that caused them.
- **Length and saves.** 7,000 iterations with a checkpoint every 100, saves at 3,500 and 7,000, and the EMA policy
  `pi`; against 3,000 updates, saves at 1,500 and 3,000, and `actor_ema`.
- **Library and scene.** This library resamples the same 40 clips to other row counts per clip (RECOVERY.md), so the
  scene's rule lands on other rows. The scene is built from a recovery training's `args.json` and checked against its
  `recovery.json`. The joints of a placed start are clamped to the joint ranges of the model. No pelvis yaw at
  placement is recorded.
- **Twins.** Trials are not repeatable one by one here, so a twin is a reference, not a copy.
- **Termination penalty.** NablaMimic's recipe trains control and recovery with a termination penalty
  (`--term-penalty 2 0.08 0.10` in NablaMimic's `configs/base/ours.args`, which `configs/absolute.args` and
  `configs/recovery.args` include: weight 2, growing quadratically once the pelvis-height difference exceeds 0.08 m or
  the up-axis difference exceeds 0.10; at the hidden row during recovery). Here `PRESET[baseline]` sets
  `--fail_penalty 0`, which switches off the release recipe's failure penalty (default 3.0, on the soft termination
  probability at every imagined step), and recovery mode refuses another value, because that penalty reads the command
  row and a penalty at the hidden target row is not implemented (RECOVERY.md, "Host configuration" and "Differences
  from the NablaMimic setting"). It is an untested candidate for the ends on get-up clips (Results). A test needs that
  penalty implemented and a control and a recovery training at both seeds, read as ends per reference get-up over
  repeated rollouts.
- **Anchor.** This trainer's reward anchor is the pelvis, NablaMimic's the torso link; both trainers' terminations read
  the pelvis. E's alignment keeps the reading's anchor, `torso_link`.
- **Records.** Completion is read from the last checkpoint and the last log row, not from a terminal record; there is
  no wall-time limit to declare; the identity of a seed's two trainings is read from `args.json`, which holds neither
  the card nor the source revision.
- **Exposure.** In imagined tracking controls, 65,536 per control iteration, not in window rows; this trainer logs no
  switch yaw, so none is given.
- **Outcome rules.** Copied into `outcome.py` from the NablaMimic study's `outcome.py` at 369a2833050b, where they are
  the same but for one docstring; the copy does not follow later revisions of that file. The reading rules are not
  copied but called.
- **Output keys.** `scene.json` holds `settings` in place of `recovery_settings`, and `library` (the `args.json` values
  `motion_dir`, `bones_list`, `bones_lib` and `motion_store`, with the clip and row counts) in place of
  `library_inputs` (environment variables); it adds `run_dir` and `robot` and gives no pelvis yaw. In
  `save_readings.json` the save names its `iteration` in place of `completed_updates` and no actor key, each rollout
  names its `card` in place of `gpu`, `readings_core` is added, and `pelvis_yaw_at_placement` is absent, as is the yaw
  in `trials.jsonl`. `outcome.json` has no `stopped` list, since `run_stopped` cannot hold; its saves are labelled in
  iterations, not updates; each training gives `run_dir` and `end` (`completed`, `last_checkpoint`,
  `last_log_iteration`) in place of `root`, `end` (`completed`, `cause`, `completed_updates`, `detail`) and the card,
  commit, environment and torch version; and it holds `exposure` and `readings_core`, with no `exposure.json`.

## Commands, in order

Run from the repository root of one committed revision in this repository's environment. `RUNS` and `STUDY` are output
roots that do not exist yet, and every program writes to an output directory that must not exist. `DIR` holds the 40
LAFAN1 G1 retarget CSVs. `NABLAMIMIC` is a checkout of [NablaMimic](https://github.com/LingzheZhao/NablaMimic) at
revision 369a2833050b, or its `studies/recovery` taken with `git archive` at that revision; only
`studies/recovery/readings_core.py` is read. `REV` is the revision of this repository that runs.

1. **The four trainings**, one card each, with `python -m blinkmimic2.train`, each from one checkout path and with one
   `DIR` (`args.json` records both through `bones_lib` and `motion_dir`). A start stamp, `process_start.txt`, is
   written before the trainer's process starts. `scripts/train.sh` is not used: after the training it averages the
   last checkpoints into `policy/`, which this study does not read, since the rollouts read the numbered checkpoints.

   ```bash
   source scripts/presets.sh
   mkdir $RUNS
   for seed in 0 1; do
     for condition in control recovery; do
       run=$RUNS/${condition}_s$seed; preset=${condition/#control/recovery_control}
       mkdir $run && date +%s > $run/process_start.txt
       python -m blinkmimic2.train --log_dir $run --seed $seed --bones_list= --motion_dir $DIR ${PRESET[$preset]}
     done
   done
   ```

2. **The scene**, on the CPU, from the recovery training of seed 0. The program prints `first_row_controls`, the control
   count of the first-row rollouts.

   ```bash
   python studies/recovery_pair/scene.py --pair studies/recovery_pair/pair.json --run_dir $RUNS/recovery_s0 \
       --out $STUDY/scene
   ```

3. **The rollouts**, three per save, one card each.

   ```bash
   export RUN_SOURCE_REVISION=$REV SOURCE_REVISION=$REV
   FIRST_ROW_CONTROLS=$(python -c "import json; print(json.load(open('$STUDY/scene/scene.json'))['first_row_controls'])")
   for run in control_s0 recovery_s0 control_s1 recovery_s1; do
     for iter in 3500 7000; do
       python tools/rollouts.py --run_dir $RUNS/$run --ckpt $iter --scene $STUDY/scene/placed.json \
           --termination nonfinite --controls 1000 --out $STUDY/rollouts/$run-$iter-placed
       python tools/rollouts.py --run_dir $RUNS/$run --ckpt $iter --scene $STUDY/scene/twins.json \
           --termination nonfinite --controls 1000 --out $STUDY/rollouts/$run-$iter-twins
       python tools/rollouts.py --run_dir $RUNS/$run --ckpt $iter --scene $STUDY/scene/first_rows.json \
           --termination recipe --controls $FIRST_ROW_CONTROLS --out $STUDY/rollouts/$run-$iter-first-rows
     done
   done
   ```

   A rollout records the revisions of the training and of itself only from `RUN_SOURCE_REVISION` and
   `SOURCE_REVISION` (`train.py` records none), so the first line sets both.

4. **The readings**, on the CPU.

   ```bash
   for save in {control,recovery}_s{0,1}-{3500,7000}; do
     python studies/recovery_pair/readings.py --pair studies/recovery_pair/pair.json \
         --readings-core $NABLAMIMIC/studies/recovery --scene $STUDY/scene --placed $STUDY/rollouts/$save-placed \
         --twins $STUDY/rollouts/$save-twins --first_rows $STUDY/rollouts/$save-first-rows --out $STUDY/readings/$save
   done
   ```

5. **The outcome**, with the exposure, on the CPU.

   ```bash
   python studies/recovery_pair/outcome.py --pair studies/recovery_pair/pair.json \
       --training control:0=$RUNS/control_s0 --training recovery:0=$RUNS/recovery_s0 \
       --training control:1=$RUNS/control_s1 --training recovery:1=$RUNS/recovery_s1 \
       $(for save in {control,recovery}_s{0,1}-{3500,7000}; do echo --save $STUDY/readings/$save; done) \
       --out $STUDY/outcome
   ```

The tests, `tests/test_recovery_pair_scene.py`, `tests/test_recovery_pair_readings.py` and
`tests/test_recovery_pair_outcome.py`, run on the CPU on generated libraries, records, run directories and logs; the
reading test gives the entry a stand-in for `readings_core.py`, so it needs no NablaMimic checkout.

## What the study does not show

- **Small differences.** It cannot carry small differences, or any statement beyond its two seeds.
- **A validation.** It is not a validation of getting up. The placed starts are drawn from the recovery training's own
  start rows, and the five hidden get-up clips are training data in both conditions: commands of the control and the
  hidden reference of the recovery training.
- **Other interference.** It cannot separate interference beyond the lower tracking exposure of recovery training and
  the iterations that tracking and recovery share.
- **Pushes, contact and speed.** It covers no pushes or falls during execution, and no contact, since the readings are
  kinematic. Its outcome makes no speed comparison; the times beside the NablaMimic study in Results cross
  simulators, collision models and units of training and hold for one card.
- **Heading and position.** Rejoining is a posture reading against the twin, not a reading of heading or world
  position.
- **Single trials.** A twin is a reference, not a copy; a single trial's rejoin is read against a run of its command
  that may differ from an exact repeat.
- **A ranking of the trainers.** Set beside the NablaMimic study, the pairs differ in trainer, simulator, collision
  model, length and saves; each outcome is read against its own control.

## Results

**Conditions.** The trainings, scene, rollouts, readings and outcome ran at revision f29424a6666c of this repository,
whose `src`, `scripts` and `tools` are those of pull request 2 (f7e140c); the reading rules are NablaMimic's
`readings_core.py` at 369a2833050b. The pair's four trainings and the rollouts of the eight declared saves ran on one
NVIDIA GeForce RTX 4090 with 48 GB, with empty compile caches at each start, on MuJoCo Warp 3.11.0 through mjlab 1.6.0
with 128 contacts per world (pooled) and 500 constraint rows; the largest counts in a rollout of the eight saves are 38
contacts and 152 constraint rows, and no trial exceeded the capacity. Control seed 1's saves other than 3,500 and 7,000
were rolled out on an NVIDIA RTX 6000 Ada. Every training completed its 7,000 iterations. No training, rollout or
reading failed or was skipped. Three further sets of three rollouts (placed starts, twins, first rows) of two saves, 18
rollouts in all (below), ran at the same revision on an RTX 4090 with 48 GB; none failed.

| Save | Get up of 100 (R) | Rejoin of 100 (J) | Rejoined over got up (readable) | Twins that fall (F) | Clips completed of 40 (C) | Body-position error E (m) |
| --- | --- | --- | --- | --- | --- | --- |
| control, seed 0, 3,500 | 93 | 92 | 92 / 93 | 0 | 39 | 0.0321 |
| control, seed 0, 7,000 | 60 | 58 | 58 / 60 | 0 | 40 | 0.0296 |
| control, seed 1, 3,500 | 97 | 95 | 95 / 97 | 0 | 37 | 0.0332 |
| control, seed 1, 7,000 | 91 | 90 | 90 / 91 | 1 | 39 | 0.0292 |
| recovery, seed 0, 3,500 | 95 | 94 | 94 / 95 | 0 | 38 | 0.0438 |
| recovery, seed 0, 7,000 | 99 | 98 | 98 / 98 | 1 | 36 | 0.0388 |
| recovery, seed 1, 3,500 | 99 | 99 | 99 / 99 | 0 | 38 | 0.0428 |
| recovery, seed 1, 7,000 | 97 | 97 | 97 / 97 | 0 | 39 | 0.0387 |

The fall rule at placement reads the state after the first control. Placed start 99 is outside it at every save; placed
start 25 sits at the 60° tilt threshold there and is outside it at six of these saves (not recovery seed 0 at 7,000 and
seed 1 at 3,500), so 98 or 99 starts are inside the rule; a start outside counts as not got up. At recovery seed 0,
7,000 one rejoin is unavailable because its twin fell. In these rollouts the control saves complete all five hidden
get-up clips; the recovery saves miss fallAndGetUp1_subject1 (seed 0, 3,500), fallAndGetUp2_subject2 and
fallAndGetUp3_subject1 (seed 0, 7,000), fallAndGetUp2_subject2 (seed 1, 3,500) and fallAndGetUp2_subject3 (seed 1,
7,000). A separate recomputation from the rollout records, with code not part of this repository, agrees with R, J,
rejoined over got up and F at all 56 saves (every 500 iterations of the four trainings), and with C, the hidden get-up
clips and E at the 32 saves with first-row rollouts. Differences below are taken before rounding.

**Outcome: `some_get_up`** (row 7; rows 1 to 6 do not hold). Both seeds are in class "some". `pair.json` reads this
row as "some get-up, not established as the hidden target's" and sends forward "the per-start reading of which starts
rise (scene.json and trials.jsonl), before any change". The class gets_up needs R ≥ 50 and R at least the larger
control R at 7,000 plus 20; the controls get up from 60 and 91 of 100 without recovery training, so that bound is 111
of 100 and cannot be reached. Read after the numbers: the declared classes do not fit a control that already gets up,
and the outcome does not say that recovery training failed. Row 3, control_within (the control's R at 7,000 within 10
of the recovery R at each seed), fails at seed 0 only (60 against 99; 91 against 97 at seed 1). The tracking terms the
outcome computed, recovery at 7,000 against each control save of its seed (margins 0.005 m on E, 2 clips on C; F is
within at every term):

| Seed | Against the control at | E difference (m) | Outside the E margin by (m) | C | C within |
| --- | --- | --- | --- | --- | --- |
| 0 | 7,000 | +0.0092 | 0.0042 | 36 against 40 | no |
| 0 | 3,500 | +0.0067 | 0.0017 | 36 against 39 | no |
| 1 | 7,000 | +0.0096 | 0.0046 | 39 against 39 | yes |
| 1 | 3,500 | +0.0055 | 0.0005 | 39 against 37 | yes |

No outcome row that holds reads these terms; the rows that read them need both seeds in gets_up. Each term rests on one
rollout per save; four first-row rollouts of seed 0 at 7,000 read C 36 to 38 for the recovery save and 39 to 40 for the
control ("Repeated rollouts of one checkpoint" below). E reads each clip up to its first termination, so where the
completed clips differ the clips are read over different spans: at 7,000 the recovery save misses
fallAndGetUp1_subject4, fallAndGetUp2_subject2, fallAndGetUp3_subject1 and fightAndSports1_subject1 against none
(seed 0), and fallAndGetUp2_subject3 against run1_subject5 (seed 1); the controls at 3,500 miss walk2_subject3
(seed 0) and dance1_subject2, fightAndSports1_subject1 and walk2_subject1 (seed 1).

**Exposure.** The recovery saves at 7,000 have 321,061,586 (seed 0) and 319,211,717 (seed 1) imagined tracking
controls, which a control reaches at iterations 4,900 and 4,871; a control has 229,376,000 at 3,500 and 458,752,000 at
7,000. *Exploratory:* the controls at 5,000 iterations read E 0.0306 m with C 38 (seed 0; it misses
fallAndGetUp1_subject1 and fallAndGetUp1_subject5) and 0.0317 m with C 40 (seed 1, on the RTX 6000 Ada), and R 58 and
96; the recovery saves at 7,000 are 0.0081 m and 0.0071 m above them. This is at about equal exposure, not at equal
iterations, and the completed clips differ (36 against 38, 39 against 40).

**Saves at every 500 iterations (exploratory for the outcome).** Every save from 500 to 7,000 was rolled out on the
same scene files. `readings.py` read the controls' 28 saves and the recovery runs' four declared saves. The recovery
runs' 24 other saves were read from placed starts and twins only, with the same `readings_core.py` called by a program
that is not part of this repository, so they have no C, no E and no record of contact or constraint capacity. Each
cell gives, for a control, R / C / E (m) and the seconds from launch at which the timing training of the same condition
and seed wrote that checkpoint ("Training time" below), and R for a recovery run:

| Iterations | control, seed 0 | control, seed 1 | recovery, seed 0 | recovery, seed 1 |
| --- | --- | --- | --- | --- |
| 500 | 10 / 13 / 0.1008, 564 s | 9 / 1 / 0.1094, 567 s | 3 | 2 |
| 1,000 | 26 / 26 / 0.0652, 622 s | 27 / 26 / 0.0619, 627 s | 8 | 5 |
| 1,500 | 93 / 33 / 0.0466, 681 s | 98 / 32 / 0.0472, 686 s | 59 | 56 |
| 2,000 | 98 / 36 / 0.0390, 740 s | 92 / 36 / 0.0390, 746 s | 87 | 82 |
| 2,500 | 97 / 37 / 0.0343, 798 s | 95 / 35 / 0.0367, 805 s | 87 | 95 |
| 3,000 | 93 / 39 / 0.0331, 857 s | 93 / 39 / 0.0337, 864 s | 96 | 97 |
| 3,500 | 93 / 39 / 0.0321, 916 s | 97 / 37 / 0.0332, 924 s | 95 | 99 |
| 4,000 | 90 / 40 / 0.0313, 974 s | 83 / 38 / 0.0325, 983 s | 98 | 98 |
| 4,500 | 75 / 40 / 0.0306, 1,033 s | 83 / 38 / 0.0321, 1,043 s | 99 | 98 |
| 5,000 | 58 / 38 / 0.0306, 1,092 s | 96 / 40 / 0.0317, 1,102 s | 98 | 97 |
| 5,500 | 85 / 40 / 0.0297, 1,150 s | 81 / 40 / 0.0306, 1,161 s | 99 | 99 |
| 6,000 | 76 / 40 / 0.0299, 1,209 s | 89 / 40 / 0.0300, 1,220 s | 98 | 98 |
| 6,500 | 62 / 39 / 0.0295, 1,268 s | 87 / 40 / 0.0294, 1,280 s | 98 | 98 |
| 7,000 | 60 / 40 / 0.0296, 1,327 s | 91 / 39 / 0.0292, 1,339 s | 99 | 97 |

The controls get up from 92 to 98 of 100 at every save from 1,500 to 3,500 iterations. At seed 0, from 4,000 on, the
control's R lies between 58 and 90 and is 60 at 7,000, while its E lies between 0.0295 and 0.0313 m.

**Training time.** The times come from timing trainings: the pair's training commands and seeds run again from empty
compile caches, each on one NVIDIA GeForce RTX 4090 with 48 GB at a 450 W power limit in a host with two 64-core CPUs,
four trainings at once beside other jobs on the host; their saves are not read. The pair's own trainings ran on a card
of the same model under a lower power limit, and their times are not used. The table gives the seconds from
`process_start.txt` to the file time of `ckpt_007000.pt`; the steady rate, between the first and the last numbered
checkpoint; and the start-up, the first numbered checkpoint's time (iteration 100, 517 to 529 s from launch) minus its
100 iterations at the steady rate:

| Training | Launch to the last checkpoint (s) | Start-up (s) | Steady (s per iteration) |
| --- | --- | --- | --- |
| control, seed 0 | 1,326.9 | 505.0 | 0.1174 |
| control, seed 1 | 1,339.4 | 507.6 | 0.1188 |
| recovery, seed 0 | 1,362.5 | 513.5 | 0.1213 |
| recovery, seed 1 | 1,388.9 | 516.3 | 0.1247 |

The times of the four timing trainings sum to 5,417.7 s.

**Beside the NablaMimic study.** The final saves of both studies:

| Final save | Recovery R / J / C / E (m) | Control C / E (m) / R |
| --- | --- | --- |
| NablaMimic, seed 0, 3,000 updates | 99 / 98 / 39 / 0.0387 | 40 / 0.0286 / 31 |
| NablaMimic, seed 1, 3,000 updates | 99 / 99 / 40 / 0.0382 | 40 / 0.0291 / 32 |
| this study, seed 0, 7,000 iterations | 99 / 98 / 36 / 0.0388 | 40 / 0.0296 / 60 |
| this study, seed 1, 7,000 iterations | 97 / 97 / 39 / 0.0387 | 39 / 0.0292 / 91 |

Besides the trainer, the two studies differ in the simulator and the robot's collision model (MuJoCo Warp with mjlab's
G1 against NablaEngine with Menagerie's G1), in the motion library's resampling of the same 40 clips, so the scene's
rule lands on other rows, and in the unit of training (iterations against updates), so equal counts are not equal
training. C and the span over which E is averaged follow each trainer's termination: the same formula, each against
its own library's rows ("Ends on get-up clips" below).

On a card of the same model and power limit, from launch to the last checkpoint, the NablaMimic study's training
commands, run again in the same way, take 1,458 and 1,450 s (control, seeds 0 and 1) and 1,462 and 1,500 s (recovery);
against this study's totals above the ratios, taken before the seconds are rounded, are 0.91, 0.92, 0.93 and 0.93.
NablaMimic's seconds are computed by the same rule from the file times of these trainings' launch and checkpoints; its
README gives only the from-launch times at its criteria and the trainer's own totals, which start about 20 s later. By
the NablaMimic study's control criterion (the first save on its grid, every 500 iterations here and 250 updates there,
with C ≥ 38 and E ≤ 0.0341 m), a control first tracks at 3,000 iterations here and at 1,750 updates there; the timing
trainings wrote those checkpoints 857 and 864 s from launch here and 908 and 904 s there: ratios 0.94 and 0.96. Both
trainers started with empty compile caches. NablaMimic's recovery criterion (R ≥ 50, R at least the larger control R
at the last save plus 20, J over R ≥ 0.5) needs R ≥ 111 here, so it gives no time.

**With pre-built compile caches (exploratory).** For each trainer and condition at seed 0, a short training of the
same configuration (100 iterations here, 250 updates there) filled the compile caches from empty; they were archived
and unpacked in another container on a card of the same model and power limit, where the training command of the pair
then ran unchanged. One training per cell; the first value is the timing training from empty caches, the second the one
with the caches:

| Training, seed 0 | Launch to the last checkpoint (s) | Start-up (s) | Steady (s per iteration or update) |
| --- | --- | --- | --- |
| control, here | 1,326.9 / 883.1 | 505.0 / 47.8 | 0.1174 / 0.1193 |
| recovery, here | 1,362.5 / 906.0 | 513.5 / 48.0 | 0.1213 / 0.1226 |
| control, NablaMimic | 1,458.1 / 1,321.6 | 137.1 / 28.3 | 0.4403 / 0.4311 |
| recovery, NablaMimic | 1,462.4 / 1,358.4 | 146.7 / 29.8 | 0.4385 / 0.4429 |

With the caches this trainer compiled none of its 44 Warp modules and its cache folder (1.1 GB) did not grow;
NablaMimic's trainer read 11 of 11 graph-cache hits and 7 of 7 autograd-cache hits, did not build its engine library,
and its cache folder (0.3 GB) did not grow. The ratio of this trainer's seconds to NablaMimic's, launch to the last
checkpoint, is 0.91 (control) and 0.93 (recovery) from empty caches and 0.67 and 0.67 with them. As with the timing
trainings, the saves of these trainings were not rolled out; the pair's control of seed 0 first tracked at iteration
3,000 here and at 1,750 updates there, and the training with the caches of that condition and seed wrote its checkpoint
of the same iteration at 406 s here and at 784 s there. A cache built on one card model was not fully used on another
(the Warp modules loaded on a card of the same compute capability and were compiled again on one of another), so a
cache is built per card model.

**Repeated rollouts of one checkpoint (exploratory).** Control seed 0 and recovery seed 0 at 7,000 each have three
further sets of three rollouts (placed starts, twins, first rows) on the same scene files, read by `readings.py`; the
first value in each cell is from the study's own set:

| Save | R | J | F | C | E (m) |
| --- | --- | --- | --- | --- | --- |
| control, seed 0, 7,000 | 60, 64, 65, 62 | 58, 62, 61, 59 | 0, 0, 1, 0 | 40, 40, 40, 39 | 0.0296 in all four |
| recovery, seed 0, 7,000 | 99 in all four | 98, 97, 97, 98 | 1, 2, 1, 1 | 36, 38, 36, 38 | 0.0388, 0.0377, 0.0382, 0.0377 |

The control's get-up count is a property of the checkpoint (60 to 65 of 100), but per start it is not fixed: 40 starts
get up in all four sets and 12 in none. The recovery policy gets up from the same 99 starts in all four. Its C reads
36, 38, 36 and 38, so a difference of two clips between saves is inside the spread of rollouts of one checkpoint. Of
its 12 clip ends over the four first-row rollouts, 10 are on the six get-up clips excluded from its commands (below);
fallAndGetUp3_subject1 ends in all four.

**Nearly the same placed starts on both trainers (exploratory).** Placed start i of the two studies comes from the same
body clip in 100 of 100 trials; the absolute differences of the placed states have medians of 0.017 m in pelvis height
and 3.2° in tilt; the pelvis-height quartiles are 0.128, 0.213 and 0.394 m here against 0.137, 0.200 and 0.417 m there;
and the counts per height bin, tilt bin and lying orientation agree within 3. So the controls' get-ups here (60 and 91
at 7,000, 98 at seed 0, 2,000 iterations) against 31 and 32 there at 3,000 updates are not a difference of the placed
starts.

**Commanded and excluded clips (exploratory).** The recovery policies' commands exclude six get-up clips (the five
hidden clips and fallAndGetUp1_subject4). The final saves, one rollout each; E over a group is the mean of its clips'
errors, each read up to the clip's first termination:

| Final save, seed 0 / seed 1 | Completed of 34 commanded | E over 34 (m) | Completed of 6 excluded | E over 6 (m) |
| --- | --- | --- | --- | --- |
| this study, control, 7,000 | 34 / 33 | 0.0279 / 0.0273 | 6 / 6 | 0.0393 / 0.0397 |
| this study, recovery, 7,000 | 33 / 34 | 0.0349 / 0.0358 | 3 / 5 | 0.0606 / 0.0554 |
| NablaMimic, control, 3,000 updates | 34 / 34 | 0.0268 / 0.0272 | 6 / 6 | 0.0389 / 0.0400 |
| NablaMimic, recovery, 3,000 updates | 33 / 34 | 0.0362 / 0.0357 | 6 / 6 | 0.0527 / 0.0528 |

**Ends on get-up clips (exploratory).** In the recovery first-row rollouts here, trials end on the excluded clips 10
times over the four rollouts of seed 0 at 7,000 (8 on the tilt term, 2 on the height term; not independent), once at
seed 1 at 7,000 (height), once at seed 0 at 3,500 (tilt) and once at seed 1 at 3,500 (height). In each the body had
gone down with the reference; in the four height-term ends it then rose and sank back (three) or rose more slowly than
the reference (one). The end came before the reference left its low span or 13 to 36 controls after, inside a get-up
as defined below, and on the term that ended the trial the reference was higher or more upright than the body. Both
trainers end a trial by one formula: pelvis-height difference above 0.25 m or up-axis z difference above 0.8, against
the command row after the control. Applied to the other study's records at 7,000 and 3,500 iterations here and 3,000
updates there, each rule gives that study's ends on these clips: none in NablaMimic (the closest trial 0.0024 m under
the height threshold, against its own library's rows), and here each end at the same control or 1 to 6 controls
before.

Ends per get-up of the reference, a get-up being the window from 50 rows before to 50 rows after the end of a low span
of at least 25 rows (a low row: pelvis below 0.50 m or up-axis z below 0.5; overlapping windows counted separately),
over the windows the trials reached. The four rollouts of one checkpoint are summed and are not independent;
NablaMimic gives one rollout per seed:

| Save | Five hidden clips | fallAndGetUp1_subject4 | 12 commanded clips with a get-up |
| --- | --- | --- | --- |
| recovery, 7,000: seed 0 (four rollouts) / seed 1 (one) | 8 in 284 / 1 in 80 | 2 in 66 / 0 in 17 | 0 in 167 / 0 in 43 |
| control, 7,000: seed 0 (four rollouts) / seed 1 (one) | 0 in 364 / 0 in 91 | 0 in 68 / 0 in 17 | 0 in 169 / 0 in 43 |
| NablaMimic recovery, 3,000 updates: seed 0 / seed 1 | 0 in 92 / 0 in 92 | 0 in 19 / 0 in 19 | 0 in 43 / 0 in 44 |
| NablaMimic recovery, 1,500 updates: seed 0 / seed 1 | 0 in 74 / 1 in 87 | 1 in 17 / 0 in 19 | 0 in 44 / 0 in 43 |

The same pattern appears at other saves: this study's controls end 14 times on the excluded clips between 2,000 and
6,500 iterations (7 at each seed; 13 of them inside a get-up window), each with the reference higher or more upright
on the ending term, and NablaMimic's recovery policies end on fallAndGetUp1_subject1, fallAndGetUp1_subject4 and
fallAndGetUp3_subject1 at 1,500 updates (one of these inside a low span, outside any window) and on none at 3,000. The
cause of these ends is not established.

**What the results do not show.** Beyond "What the study does not show" above:

- A class for the recovery policies' getting up (see the outcome).
- A tracking comparison at equal exposure and equal iterations.
- The saves at every 500 iterations, the exposure comparison, the times and the four exploratory readings after them
  are not part of the outcome.
- A cause of the ends on get-up clips, or a rate of them with an interval: the repeated rollouts are of one checkpoint
  per condition, at seed 0.
- The times hold for one card model and power limit, the RTX 4090 with 48 GB at 450 W, four trainings at once beside
  other jobs on the host, and come from timing trainings whose saves are not read; the trainings with pre-built caches
  are one run per condition at seed 0.
