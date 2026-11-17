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
- **Anchor.** This trainer's reward and termination anchor is the pelvis; E's alignment keeps the reading's anchor,
  `torso_link`.
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
  start rows, and the five hidden get-up clips are tracking commands in both conditions.
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
`readings_core.py` at 369a2833050b. Every training and the rollouts of the eight declared saves ran on one NVIDIA
GeForce RTX 4090 with 48 GB under a 350 W power limit, with empty compile caches at each start, on MuJoCo Warp 3.11.0
through mjlab 1.6.0 with 128 contacts per world (pooled) and 500 constraint rows; the largest counts in a rollout of the
eight saves are 38 contacts and 152 constraint rows, and no trial exceeded the capacity. Control seed 1's saves other
than 3,500 and 7,000 were rolled out on an NVIDIA RTX 6000 Ada. Every training completed its 7,000 iterations. No
training, rollout or reading failed or was skipped.

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

The fall rule at placement reads the state after the first control. Placed start 99 is outside it at every save;
placed start 25 sits at the 60° tilt threshold there and is outside it at six of these saves (not recovery seed 0 at
7,000 and seed 1 at 3,500), so 98 or 99 starts are inside the rule; a start outside counts as not got up. At recovery
seed 0, 7,000 one rejoin is unavailable because its twin fell. The control saves complete all five hidden get-up clips;
the recovery saves miss fallAndGetUp1_subject1 (seed 0, 3,500), fallAndGetUp2_subject2 and fallAndGetUp3_subject1
(seed 0, 7,000), fallAndGetUp2_subject2 (seed 1, 3,500) and fallAndGetUp2_subject3 (seed 1, 7,000). A separate
recomputation from the rollout records, with code written from the reading rules' words that is not part of this
repository, agrees with every reading: R, J, rejoined over got up and F at all 56 saves (every 500 iterations of the
four trainings), and C, the hidden get-up clips and E at the 32 saves with first-row rollouts. Differences below are
taken before rounding.

**Outcome: `some_get_up`** (row 7; rows 1 to 6 do not hold). Both seeds are in class "some". `pair.json` reads this
row as "some get-up, not established as the hidden target's" and sends forward "the per-start reading of which starts
rise (scene.json and trials.jsonl), before any change". The class gets_up needs R ≥ 50 and R at least the larger
control R at 7,000 plus 20; the controls get up from 60 and 91 of 100 without recovery training, so that bound is 111
of 100 and cannot be reached. Read after the numbers: the declared classes do not fit a control that already gets up,
and the outcome does not say that recovery training failed. At 7,000 the recovery save against its control reads R 99
against 60, J 98 against 58, C 36 against 40, E 0.0388 against 0.0296 m and F 1 against 0 (seed 0), and R 97 against
91, J 97 against 90, C 39 against 39, E 0.0387 against 0.0292 m and F 0 against 1 (seed 1). Row 3, control_within
(the control's R at 7,000 within 10 of the recovery R at each seed), fails at seed 0 only (60 against 99; 91 against
97 at seed 1). At 3,500 the recovery saves get up from 95 and 99 and the controls from 93 and 97. The tracking terms
the outcome computed, recovery at 7,000 against each control save of its seed (margins 0.005 m on E, 2 clips on C; F
is within at every term):

| Seed | Against the control at | E difference (m) | Outside the E margin by (m) | C | C within |
| --- | --- | --- | --- | --- | --- |
| 0 | 7,000 | +0.0092 | 0.0042 | 36 against 40 | no |
| 0 | 3,500 | +0.0067 | 0.0017 | 36 against 39 | no |
| 1 | 7,000 | +0.0096 | 0.0046 | 39 against 39 | yes |
| 1 | 3,500 | +0.0055 | 0.0005 | 39 against 37 | yes |

No outcome row that holds reads these terms; the rows that read them need both seeds in gets_up. E reads each clip up
to its first termination, so where the completed clips differ the clips are read over different spans: at 7,000 the
recovery save misses fallAndGetUp1_subject4, fallAndGetUp2_subject2, fallAndGetUp3_subject1 and
fightAndSports1_subject1 against none (seed 0), and fallAndGetUp2_subject3 against run1_subject5 (seed 1); the
controls at 3,500 miss walk2_subject3 (seed 0) and dance1_subject2, fightAndSports1_subject1 and walk2_subject1
(seed 1).

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
cell gives R / C / E (m) for a control, R for a recovery run, and the seconds from launch to the checkpoint's file time:

| Iterations | control, seed 0 | control, seed 1 | recovery, seed 0 | recovery, seed 1 |
| --- | --- | --- | --- | --- |
| 500 | 10 / 13 / 0.1008, 663 s | 9 / 1 / 0.1094, 658 s | 3, 672 s | 2, 670 s |
| 1,000 | 26 / 26 / 0.0652, 728 s | 27 / 26 / 0.0619, 723 s | 8, 739 s | 5, 737 s |
| 1,500 | 93 / 33 / 0.0466, 792 s | 98 / 32 / 0.0472, 787 s | 59, 806 s | 56, 804 s |
| 2,000 | 98 / 36 / 0.0390, 857 s | 92 / 36 / 0.0390, 852 s | 87, 873 s | 82, 871 s |
| 2,500 | 97 / 37 / 0.0343, 922 s | 95 / 35 / 0.0367, 917 s | 87, 940 s | 95, 938 s |
| 3,000 | 93 / 39 / 0.0331, 986 s | 93 / 39 / 0.0337, 981 s | 96, 1,008 s | 97, 1,006 s |
| 3,500 | 93 / 39 / 0.0321, 1,051 s | 97 / 37 / 0.0332, 1,046 s | 95, 1,075 s | 99, 1,073 s |
| 4,000 | 90 / 40 / 0.0313, 1,115 s | 83 / 38 / 0.0325, 1,111 s | 98, 1,142 s | 98, 1,140 s |
| 4,500 | 75 / 40 / 0.0306, 1,180 s | 83 / 38 / 0.0321, 1,175 s | 99, 1,209 s | 98, 1,207 s |
| 5,000 | 58 / 38 / 0.0306, 1,245 s | 96 / 40 / 0.0317, 1,240 s | 98, 1,276 s | 97, 1,274 s |
| 5,500 | 85 / 40 / 0.0297, 1,309 s | 81 / 40 / 0.0306, 1,305 s | 99, 1,343 s | 99, 1,341 s |
| 6,000 | 76 / 40 / 0.0299, 1,374 s | 89 / 40 / 0.0300, 1,369 s | 98, 1,410 s | 98, 1,408 s |
| 6,500 | 62 / 39 / 0.0295, 1,438 s | 87 / 40 / 0.0294, 1,434 s | 98, 1,477 s | 98, 1,475 s |
| 7,000 | 60 / 40 / 0.0296, 1,505 s | 91 / 39 / 0.0292, 1,500 s | 99, 1,546 s | 97, 1,544 s |

The controls get up from 92 to 98 of 100 at every save from 1,500 to 3,500 iterations. At seed 0, from 4,000 on, the
control's R lies between 58 and 90 and is 60 at 7,000, while its E lies between 0.0295 and 0.0313 m.

**Training time.** Seconds from `process_start.txt` to the file time of `ckpt_007000.pt`; before the first iteration,
that total minus the checkpoint's `wall_time` (the trainer's clock, which starts after its warm-up collection); the
steady rate, between the first and the last numbered checkpoint; and the start-up, the first numbered checkpoint's time
(606 to 619 s from launch) minus its 100 iterations at the steady rate:

| Training | Launch to the last checkpoint (s) | Before the first iteration (s) | Start-up (s) | Steady (s per iteration) |
| --- | --- | --- | --- | --- |
| control, seed 0 | 1,504.8 | 140.3 | 598.2 | 0.1295 |
| control, seed 1 | 1,500.5 | 140.5 | 593.1 | 0.1296 |
| recovery, seed 0 | 1,546.1 | 141.2 | 605.8 | 0.1343 |
| recovery, seed 1 | 1,544.1 | 140.8 | 603.5 | 0.1344 |

The four trainings together take 6,095.5 s.

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
training. C and the span over which E is averaged follow each trainer's own termination, pelvis-anchored here.

The bands and the time criteria were fixed before these trainings, outside `pair.json`: the bands from NablaMimic's
final saves, the criteria with NablaMimic's numbers. The control band and the control criterion were fixed after
control trainings of this configuration on another card had been read, and this trainer's rate (7.6 iterations per
second) and its start-up of about 10 minutes from launch to the 100th checkpoint were known before the two clocks were
fixed, so neither the control comparisons nor the clocks are blind. The recovery bands and criterion were fixed before
the get-ups of any recovery policy of this trainer had been read.

| Band at 7,000 iterations | Seed 0 | Seed 1 |
| --- | --- | --- |
| recovery R over the starts inside the fall rule ≥ 0.90 | 99 / 99, holds | 97 / 98, holds |
| recovery rejoined over got up ≥ 0.89 | 98 / 98, holds | 97 / 97, holds |
| recovery F ≤ 2 | 1, holds | 0, holds |
| recovery C ≥ 37 | 36, missed | 39, holds |
| recovery minus control E ≤ +0.0151 m | +0.0092 m, holds | +0.0096 m, holds |
| control C ≥ 38 and E ≤ 0.0341 m | 40, 0.0296 m, holds | 39, 0.0292 m, holds |

The two studies were to be called comparable only if every band held at both seeds; one band is missed at seed 0 (C 36
against at least 37), so "comparable" is not called; the other five hold at both seeds.

Times on the same card run from launch to the checkpoint's file time; after start-up is that time minus the training's
start-up (above; 157 to 171 s for NablaMimic). NablaMimic's times after start-up, to the last checkpoint and its
start-ups are computed by the same rule from the file times of its trainings' launch and checkpoints; its README gives
only the from-launch times at its criteria and the trainer's own totals, which start about 20 s later. A control
tracks at the first save on its grid (500 iterations here, 250 updates there) with C ≥ 38 and E ≤ 0.0341 m. The
recovery criterion (R ≥ 50, J over R ≥ 0.5 and R at least the larger control R at 7,000 plus 20) cannot be met here,
as for the class gets_up above. The exploratory rows put R ≥ 80 in place of its last term; 80 was chosen after the
readings were seen, and the controls of this study meet it first, at 1,500 iterations (R 93 and 98, J over R 91 / 93
and 98 / 98; 792 s and 787 s from launch), so these rows do not time a get-up attributable to recovery training.

| Seconds, seed 0 / seed 1 | Here | NablaMimic | Ratio |
| --- | --- | --- | --- |
| control tracks, from launch | 986 / 981 (3,000 iterations; C 39 / 39, E 0.0331 / 0.0337 m) | 1,107 / 1,105 (1,750 updates) | 0.89 / 0.89 |
| control tracks, after start-up | 388 / 388 | 950 / 947 | 0.41 / 0.41 |
| *exploratory:* recovery R ≥ 80, from launch | 873 / 871 (2,000 iterations; R 87, 82) | 442 / 443 (500 updates; R 97, 97) | 2.0 / 2.0 |
| *exploratory:* recovery R ≥ 80, after start-up | 267 / 268 | 272 / 272 | 0.98 / 0.98 |

By the declared call, a ratio below 1 at both seeds, this trainer's control is faster on both clocks; the recovery
criterion is not met, so it has no time and no call. From launch to the last checkpoint NablaMimic's trainings take
1,784 and 1,781 s (control) and 1,804 and 1,806 s (recovery), ratios 0.84, 0.84, 0.86 and 0.86. Every ratio, here
and in the table, is taken before the seconds are rounded.

**What the results do not show.** Beyond "What the study does not show" above:

- A class for the recovery policies' getting up (see the outcome).
- A tracking comparison at equal exposure and equal iterations.
- The saves at every 500 iterations, the exposure comparison, the bands and the times are not part of the outcome.
- The times hold for one card, the RTX 4090 with 48 GB at 350 W.
