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
  kinematic. It makes no speed comparison.
- **Heading and position.** Rejoining is a posture reading against the twin, not a reading of heading or world
  position.
- **Single trials.** A twin is a reference, not a copy; a single trial's rejoin is read against a run of its command
  that may differ from an exact repeat.
- **A ranking of the trainers.** Set beside the NablaMimic study, the pairs differ in trainer, simulator, collision
  model, length and saves; each outcome is read against its own control.

## Results

Results are added after the runs: the revision the trainings, scene, rollouts, readings and outcome ran at; the card;
each training's total time, the seconds from `process_start.txt` to the file time of `ckpt_007000.pt`, with the time
before the first iteration (the total minus that checkpoint's `wall_time`, the trainer's clock, which starts after its
warm-up collection; the difference then includes the writing of that checkpoint); the readings of the eight saves; the
outcome and the exposure; and every failed or skipped part.
