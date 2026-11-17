# Rollouts of a checkpoint from placed starts

[`tools/rollouts.py`](tools/rollouts.py) runs the policy of one numbered checkpoint closed-loop in the
evaluation-mode simulator (MuJoCo Warp through mjlab), one world per trial of a scene file, and records every
control. A trial can start from a body placed at any library row (a lying row of a get-up clip, say) while the
command starts at another row, from small explicit offsets of the reset, and can carry a push. "The release" below is
BlinkMimic 2 v1.0.0, the code this fork starts from; the tool reads its environment, termination and reset as they are.

A **control** is one 0.02 s policy step (50 Hz, four physics substeps of 5 ms); a **row** is one 0.02 s frame of the
motion library. The **command** is the reference row the policy tracks: the policy's goal and observation read it,
it advances one row per control and holds at its clip's last row.

```bash
python tools/rollouts.py --run_dir logs/run_s1 --ckpt 6500 --scene scene.json --termination nonfinite \
    --controls 1000 --out rollouts/run_s1_6500
```

| Option | Meaning |
| --- | --- |
| `--run_dir` | run directory of `blinkmimic2.train` (`args.json`, `ckpt_NNNNNN.pt`) |
| `--ckpt` | iteration N of the checkpoint `ckpt_NNNNNN.pt` (six digits) |
| `--scene` | scene file (JSON), below |
| `--termination` | `recipe` (the training recipe's termination) or `nonfinite` (only a non-finite state), below |
| `--controls` | controls per trial, at least 1 |
| `--out` | output directory; it must not exist |

## Requirements and refusals

- One CUDA device (`cuda:0`).
- `args.json` with the keys `robot`, `motion_dir`, `bones_list`, `bones_lib` and `motion_store`, which `train.py`
  writes. The motion library is built from them as `tools/evaluate.py` builds it, so the scene's rows index the
  run's library.
- A checkpoint with `pi`, `iteration` and `args` (`goal_frames`, `hist`, `enc_hidden`, `dec_hidden`, `amp`,
  `pi_ref_skip`).
- A missing file or key raises. The scene is checked against the library before the output directory is made.

## Environment and policy

The simulator is `TrackingSim(make_sim(robot), T, robot, motion, mdp, "cuda:0", eval_mode=True)`, as
`tools/evaluate.py` builds it, with T the number of trials, and never with the recovery options of
[RECOVERY.md](RECOVERY.md), whatever the run's `args.json` holds:

- The release's solver settings and capacities: per world 128 contacts (`nconmax`; MuJoCo Warp pools them into one
  array of 128 x T contacts for the batch) and 500 constraint rows (`njmax`).
- `randomize` is off (`DomainRandomizer` in [`dr.py`](src/blinkmimic2/dr.py) in evaluation mode). It writes a sliding
  friction of 1 to every robot geom, keeps the joint encoder offsets at zero and leaves the centre of mass of
  `torso_link` at the model's own. It also writes 1 to the geoms of the world body, but the floor is not one of them:
  mjlab's plane terrain puts the floor plane on a body of its own (`terrain`) and gives it no friction, so it keeps
  MuJoCo's default sliding friction of 1. A contact takes the friction of the geom of higher priority (the feet's
  capsules have priority 1), else the larger of the two, so every robot-floor contact has a sliding friction of 1. The
  observation has no noise and no delay, and the release's random pushes are off.
- `reset_noise` is off: a trial's body is placed exactly at its row's reference state plus the scene's offsets.

The policy is loaded as `tools/evaluate.py` loads it: the averaged (EMA) weights `pi` of the checkpoint, its MLP
matrix products in bf16 autocast when the run trained with `--amp bf16`, TF32 matrix products elsewhere. As in
`train.py` and unlike `tools/evaluate.py`, the 3x3 rotation products of the kinematics, the observation and the goal
are written elementwise (`rotations.ELEMENTWISE`), so they are float32 products without TF32 rounding; the
observation therefore differs from `tools/evaluate.py`'s by rounding. The actions are the policy's output without
noise. The observation history (`hist` observations) starts filled with the first
observation; the goal is `motion.goal` of the command row.

## Scene

`{"trials": [TRIAL, ...]}`, at least one trial; unknown keys are refused at every level. A trial holds:

| Key | Meaning |
| --- | --- |
| `command_row` | library row the command starts at (required) |
| `body_row` | library row whose reference state the body is placed at; absent: the command row |
| `reset_uniforms` | `{"position" [3], "rotation" [3], "velocity" [6], "joints" [J]}`, every key, values in [-1, 1]; absent: zeros |
| `push` | `{"control", "velocity_xy"}`: a planar world velocity change (m/s) of the root at that control |

Rows are integers in [0, rows of the library), push controls integers in [0, `--controls`), and push velocities
two finite numbers.

The uniforms scale the half-widths of the release's reset noise (`DomainRandCfg` in
[`dr.py`](src/blinkmimic2/dr.py)), applied as the training reset applies the noise:

| Uniforms | Half-widths | Applied to |
| --- | --- | --- |
| `position` | 0.05, 0.05, 0.01 m | the root position (world) |
| `rotation` | 0.1, 0.1, 0.2 rad | a world-frame rotation vector, premultiplied on the root rotation |
| `velocity` [0:3] | 0.5, 0.5, 0.2 m/s | the root's world linear velocity |
| `velocity` [3:6] | 0.52, 0.52, 0.78 rad/s | the root's world angular velocity |
| `joints` | 0.1 rad | the joint positions, in the robot's joint order |

Every trial starts through `TrackingSim.reset(ids, command_rows, uniforms, body_rows)`: the body takes the body
row's reference root pose, root velocities, joint positions and joint velocities plus the offsets above, the joint
positions are then clamped to the joint ranges of the model, the previous action is zero, and the command starts at
the command row. Zero uniforms place the body exactly at the reference state.

A push at control `c` adds `velocity_xy` to the root's world x and y velocity (the free joint's `qvel[0:2]`) of the
state control `c` starts from, after the action of control `c` is computed and before its physics substeps; the rest
of the state is unchanged. The policy sees the push from control `c + 1` on.

## Terminations

- `--termination recipe`: the release's termination (`MDP.termination` in [`mdp.py`](src/blinkmimic2/mdp.py))
  against the command row after the control ends a trial: the root height differs from the command root's by more
  than 0.25 m, or the z component of the root's z axis (the cosine of its tilt from upright) differs from the
  command root's by more than 0.8. A non-finite state also ends a trial.
- `--termination nonfinite`: only a non-finite state ends a trial.

Under `recipe` a body placed lying at an upright command ends at its first control, so placed lying starts are run
with `nonfinite`. Neither a clip's end nor the training episode limit ends a trial.

A trial that has ended is placed again at its start before every later control and acts with zero actions, so the
batch keeps stepping it with a finite state; its results are not recorded and a later push is not applied. The
control that ends a trial is recorded (`available` unless its state is not finite). Placing again runs
`TrackingSim.reset` on the ended worlds at every later control, a cost that grows with the number of ended trials and
the controls left.

## Repeatability

Trials are not repeatable one by one. Two trials of one batch that start from the same row with the same uniforms
drift apart from the first control: MuJoCo Warp fills its shared contact array in an order that varies, so the
simulated states differ in their last bits, and the policy's bf16 inference turns such a difference into an action
step of 1/256 within a few controls. A trial placed at the same row as another is therefore a reference with the
same start and command, not an identical copy; compare counts and averages over trials, not single trials.

## Record

`description.json`, written before the rollouts, holds:

- `run_dir`, `checkpoint` (its path) and `iteration`; `run_args`, the run's `args.json`;
- `source_revision`: `run` and `rollouts` from the environment variables `RUN_SOURCE_REVISION` and
  `SOURCE_REVISION` (null when unset; `train.py` records none);
- `scene_path` and `scene` (its content), `termination` and `controls`;
- `library`: the four library inputs of `args.json`, the number of clips and of rows;
- `card`: the CUDA device name;
- `simulator`: `nconmax`, `njmax`, `naconmax_pooled` (the pooled contact capacity), the solver settings, the physics
  time step and the substeps per control;
- `bodies`: the names of the robot's tracked bodies in the order of the record, and the indices of `pelvis` and
  `torso_link` among them;
- `settings`: evaluation mode, randomization and reset noise (off), the action noise (0), the policy's goal frames,
  history and autocast, TF32, the elementwise rotation products, the control period, the termination thresholds, the half-widths of the uniforms and the
  versions of torch, warp-lang, mujoco-warp and mjlab.

`rollouts.npz` holds, per control and trial (`[C, T, ...]` with `C` controls, `T` trials and `B` tracked bodies;
after a trial's end NaN, -1 or false):

| Field | Shape, type | Meaning |
| --- | --- | --- |
| `body_pos_w`, `body_quat_w` | [C, T, B, 3], [C, T, B, 4] float32 | world positions and orientations (WXYZ quaternions) of the tracked bodies after the control, by the forward kinematics of the state (`blinkmimic2.kinematics` with elementwise float32 products, as in training) |
| `command_body_pos_w`, `command_body_quat_w` | [C, T, B, 3], [C, T, B, 4] float32 | the reference bodies of the command row after the control (`motion.ref_bodies`); NaN where the state is not available |
| `command_row_before`, `command_row_after` | [C, T] int32 | the command row before and after the control |
| `clip_last_row` | [C, T] int32 | the last row of the command's clip (the clip end is `command_row_after == clip_last_row`) |
| `available` | [C, T] bool | the state after the control is recorded and finite |
| `done` | [C, T] bool | the control ends the trial |
| `nonfinite` | [C, T] bool | the state after the control is not finite |
| `termination_height`, `termination_tilt` | [C, T] bool | the two terms of the release's termination against the command row after the control, recorded under either termination |
| `contacts`, `constraint_rows` | [C, T] int32 | the contacts and constraint rows the trial's world used in the control's last physics substep |
| `contacts_beyond_capacity`, `rows_beyond_capacity` | [C, T] bool | in a substep of the control, contacts beyond the pooled capacity or constraint rows beyond `njmax` were dropped |
| `action` | [C, T, J] float32 | the policy's action |
| `state` | [C, T, 13 + 2J] float32 | the state after the control in the release's layout (`mdp.flatten_state`): root position, root WXYZ quaternion, root linear and angular velocity (world), joint positions, joint velocities; the world angular velocity is `TrackingSim.state`'s product of the root rotation and the free joint's body-frame angular velocity, with TF32 rounding |
| `push_applied` | [C, T] bool | the trial's push is applied at the control |

and per trial:

| Field | Shape, type | Meaning |
| --- | --- | --- |
| `command_row`, `body_row` | [T] int32 | the trial's rows |
| `push_control` | [T] int32 | the push's control, -1 without a push |
| `push_velocity_xy` | [T, 2] float32 | the push's velocity change, zero without one |
| `start_state` | [T, 13 + 2J] float32 | the placed state before the first control |
| `start_body_pos_w`, `start_body_quat_w` | [T, B, 3], [T, B, 4] float32 | its tracked bodies |

The counts and flags come from MuJoCo Warp's data after the control: `contacts` counts the stored contacts of the
world (`contact.worldid` over the first `nacon` entries, at most the pooled capacity), `constraint_rows` is `nefc`
at most `njmax` (MuJoCo Warp's `nefc` counts on past `njmax` when rows are dropped; the rows used are `njmax` then,
and `rows_beyond_capacity` is set). The overflow bitmask `overflow` is cleared before each control
and read after it, so the flags cover the control's four substeps: `rows_beyond_capacity` is its NEFC bit, set for
the world concerned, and `contacts_beyond_capacity` its BROADPHASE or NARROWPHASE bit, which MuJoCo Warp sets on
every world of the batch when the pooled contact array overflows.

## Tests

`python -m unittest discover -s tests` runs on the CPU, without a simulator or GPU, with a library of generated clips
(`tests/synthetic.py`), the release's kinematics and MDP, and a stand-in for the simulator. The tests cover:

- the placement (`env.placed_state`): the random reset noise drawn in the release's order with the release's values,
  for a few reset worlds and for every world as `reset_masked` draws it; no draw without noise; the uniforms times
  the half-widths, with no draw; zero uniforms give the reference exactly; `TrackingSim.reset` without the added
  arguments writes the release's draws, and with them places the body row while the command starts at the command
  row;
- the scene's defaults and refusals;
- the rollout loop: each record field's shape and type; the record holds the stepped state, the command's bodies and
  the step's counts and flags; the uniforms reach the placement; the command row advances and holds at the last row
  of its own clip without ending a trial, in three clips of one batch; the height and tilt terms each end a trial under
  `recipe`, also just beyond their thresholds and not just within them, and equal `MDP.termination`; a non-finite
  state ends a trial under both terminations, the trial is placed again and its later records are unavailable; an
  ended trial leaves the other trials' records equal to a batch without it; a push changes only its trial's planar
  root velocity, from its control on, and is not applied after the trial's end;
- the command line's refusals of an existing output directory and of zero controls;
- `train.py --motion_dir`: refused with a non-empty `--bones_list`; the library `tools/evaluate.py` builds from such
  a run's `args.json` is the folder's clips; a folder without motion files is refused.

They leave out what needs the simulator and a GPU: `rollouts.Simulator` (the placement through `TrackingSim.reset`,
pushes written to the simulator, the command row held through `TrackingSim.step`, and the reading of MuJoCo Warp's
counts and overflow bits), `main` (loading the run, the checkpoint and the library, the rounding of the products
under TF32, and the files written) and `train.py` passing `--motion_dir` to the library (it trains once parsed).

## Rollouts on MuJoCo C

[`tools/mujoco_rollouts.py`](tools/mujoco_rollouts.py) runs the same rollouts with the physics stepped by MuJoCo C
(the `mujoco` package) in place of MuJoCo Warp, and writes the same record.

```bash
python tools/mujoco_rollouts.py --run_dir logs/run_s1 --ckpt 6500 --scene scene.json --termination nonfinite \
    --controls 1000 --threads 16 --out rollouts/run_s1_6500_mujoco
```

The options are those of `tools/rollouts.py` plus `--threads`, the number of threads that step the trials (required,
at least 1), and `--checks RECORD` (below).

**What is the same.** The program loads the run, the checkpoint, the library and the scene as `tools/rollouts.py`
does, runs the policy on `cuda:0` at the same precision (bf16 autocast for `--amp bf16`, TF32, elementwise rotation
products), builds the evaluation-mode `TrackingSim` as that tool builds it, and runs its loop `rollouts.roll_out`:
the history, the goal, the command, the terminations, the re-placement of ended trials and the record. The simulator
object (`MujocoC`) replaces only `rollouts.Simulator`. It computes the placement (`TrackingSim.reset` in evaluation
mode, the joint clamp included), the state tuple (`TrackingSim.state`), the observation (no noise, no delay), the push
and the actuator target (default pose plus the joint offsets of `xi` plus the action scale times the action) with the
same float32 torch expressions on the same device, and holds its own previous action (zero at a placement, the
action after a control) and its own `xi` (that of the evaluation: friction 1, the rest 0). A run whose simulator has
randomization, reset noise, stateful sensing, the recovery options, delayed actuators or actuators other than mjlab's
built-in joint position actuators is refused.

**The model.** MuJoCo C steps the `MjModel` the training simulator is built from (`TrackingSim.sim.mj_model`,
compiled by mjlab with `make_sim`'s options applied) and nothing else, with the entries mjlab wrote into MuJoCo
Warp's per-world model after compilation. In evaluation mode these are the sliding frictions of 1 that
`DomainRandomizer.startup` writes to the robot's geoms (the feet's compiled 0.6 among them); the floor keeps its
compiled 1. A field whose worlds differ raises. The options are those of MuJoCo Warp's model, which are the compiled
model's except the solver tolerance: MuJoCo Warp's `put_model` raises it to at least 1e-6 for float32, so for a
robot whose `solver` sets less (`mjlab_g1`: 1e-8) MuJoCo C is given MuJoCo Warp's value, recorded under
`simulator.options_written`; any other option in which the two models differ raises (the check mode lists it
instead). Every other field is the compiled model's: the geoms' contact flags (`contype`, `conaffinity`, `condim`,
`priority`, margins), the excluded body pairs, friction, the solver (Newton, `iterations`, `ls_iterations` and
`ccd_iterations` of the robot's `solver`), the `implicitfast` integrator, the pyramidal cone, the 5 ms time step with
four substeps per control, gravity, and the actuators (fixed gain kp, bias -kp q - kd qdot, force range, ctrl not
limited). The qpos, qvel and ctrl addresses are found by joint name and must equal mjlab's indexing and its path from the position target to `ctrl`, which writes the target
of each joint, unclamped, to that joint's actuator; a mismatch raises. The check mode dumps the model and sets it
beside MuJoCo Warp's model, option by option and field by field.

**What differs.** These set-up differences remain, besides the engine itself (its collision routines, constraint
solver implementation, Jacobian representation under the `auto` option and arithmetic; MuJoCo Warp, for one, puts a
floor of 1e-6 under a line search's gradient tolerance, which no option reaches):

- precision: MuJoCo C holds the model and the state in float64, MuJoCo Warp in float32. Each trial has one `MjData`
  for the whole rollout; between controls nothing is written to it from the float32 state, and the solver's warm
  start (`qacc_warmstart`) is carried from step to step as MuJoCo C carries it and cleared by `mj_resetData` at a
  placement, as mjlab's reset clears it. Float32 values enter the `MjData` at a placement (exact in float64), as
  `ctrl`, and as a push's velocity change, which is added in float64 where MuJoCo Warp adds in float32. The policy
  reads the state rounded to float32.
- capacities: MuJoCo C has no fixed per-world contact or constraint-row capacity; it allocates both from the
  `MjData` arena (`narena` bytes, in the description). `contacts` and `constraint_rows` are MuJoCo C's `ncon` and
  `nefc` after the control's last substep. `contacts_beyond_capacity` and `rows_beyond_capacity` are set where the
  `CONTACTFULL` or `CNSTRFULL` warning count of the trial's data rose during the control, that is, where the arena ran
  full and contacts or constraint rows were dropped. The description's `nconmax`, `njmax` and `naconmax_pooled` are
  null.
- bad states: in a step that meets a position, velocity or acceleration that is not finite or beyond 1e10, MuJoCo C
  warns and resets the data to the model's initial state, which would otherwise continue the trial from that state.
  The state after a control in which one of these warnings or the warning of a bad `ctrl` rose, or in which the
  data's time went back, is recorded as not finite, so the trial ends as a non-finite state ends it on MuJoCo Warp.
- stepping: the trials are split into contiguous chunks over `--threads` threads (`mj_step` releases Python's
  interpreter lock), each thread stepping its own trials' `MjData`; the policy runs batched on the card. The record
  does not depend on the number of threads.

**Record.** `rollouts.npz` has the fields, shapes, types and meanings of the record above. `description.json` has the
keys and values of `tools/rollouts.py`'s, except `simulator`, which names the engine (`"engine": "MuJoCo C"`, the
`mujoco` version, the threads, the entries written after compilation, the options taken from MuJoCo Warp's model,
the state's precision, the null MuJoCo Warp capacities, `narena`, the robot's solver setting as `tools/rollouts.py`
records it, the physics time step, the substeps and the options MuJoCo C steps with), and `settings.versions`, which
adds `mujoco`. `card` is the device of the policy, not of the physics.

### Check mode

`--checks RECORD`, with RECORD a `tools/rollouts.py` output on the same scene file and library, writes `checks.json`
in place of the rollouts. It is meant to be read before any MuJoCo C rollout is. It holds the description and:

- (a) placement: every trial placed on MuJoCo C, with its state and tracked bodies (the release's kinematics) against
  the record's `start_state`, `start_body_pos_w` and `start_body_quat_w`, the largest difference per field; the
  training simulator's placement in the same process against the record; `qpos` and `qvel` of the two engines
  after placement; MuJoCo C's own body positions and orientations (`mj_kinematics`) against the record's.
- (b) one control from the same placed state with the same action (the policy's first action, one tensor for both):
  on the trials lifted 2 m, where at most self-contacts and joint limits act, which tests the actuators, gains,
  integrator and gravity; and on the scene's placed trials with their floor contact. The contacts and constraint rows
  of both engines are given (MuJoCo C's also as the largest over the control's substeps, with its joint-limit rows;
  the training simulator's after the last substep only), and two subsets are reported apart: the trials without
  contact (none in any substep on MuJoCo C, none after the last substep on the training simulator) and the trials
  without constraint rows (the same, for rows), where no constraint acts. Beside each are the difference between two steps of the training
  simulator from that same state (its own repeatability), that between two steps of MuJoCo C, and the state's change
  over the control on the training simulator (the scale of the differences). The `ctrl` of the two engines after the
  control must be equal.
- (c) the model: MuJoCo C's options, per-geom contact settings, explicit and excluded pairs, actuators and dofs; the
  fields and options where it differs from the training simulator's `MjModel` (only the entries and options written,
  as it is a copy of that model); the entries and options written. The models the two engines step are compared in
  two further entries, which are where a difference between them appears: each option the two hold, MuJoCo C's value
  beside MuJoCo Warp's, equal at the float32 rounding of MuJoCo C's value or not (`impratio` through MuJoCo Warp's
  `impratio_invsqrt`), with the options only one of them holds named; and every array field of MuJoCo Warp's model
  against MuJoCo C's field of that name, the entries that differ in any world, with the fields of another layout and
  those the `MjModel` does not hold named. Last, the largest difference between mjlab's default pose and the robot's.
- (d) two rollouts of the first 50 controls (fewer when `--controls` is smaller) of the scene's trials on MuJoCo C,
  compared field by field, with their durations.

Differences are reported with the trial where the largest is, the median over trials and the trials left out as not
finite; no tolerance is applied. A RECORD of another scene or library, a record whose tracked bodies or rows are not
the robot's and the scene's, and `ctrl` that differs between the engines (a joint-to-actuator mismatch) raise.

### Comparing the two engines' records

The two records have the same fields, so one reader reads both, and a difference between them is one of engine, not
of model, scene, policy or loop (apart from what the check mode lists). Trials are not repeatable on MuJoCo Warp
(Repeatability, above), so a MuJoCo C trial and the MuJoCo Warp trial from the same start are two runs from one start,
not one trial computed twice: compare counts and averages over trials, and set the trials that change outcome between
the engines beside those that change among repeated MuJoCo Warp rollouts of the same checkpoint. A comparison does
not separate MuJoCo C's float64 arithmetic from its algorithms, and it says nothing about a physical robot.

Nothing in a record names its engine but `simulator` (and the `mujoco` entry of `settings.versions`). A reader that
checks the scene, termination, controls, library, control period, robot and the run's identity of the records it
reads together, and not `simulator`, accepts a MuJoCo C record, and also a set of records that mixes the two engines
for one checkpoint. Records read together are to be taken from one engine by their directories, and a reading that
lists its records' directories and `card` names its engine only through those directories (`card` is the policy's
device on both engines).

### Tests of the MuJoCo C rollouts

`tests/test_mujoco_rollouts.py` runs on the CPU with the robot's model as mjlab compiles it (the scene of
`make_scene_cfg` with `make_sim`'s options, built on the CPU), small generated models and the synthetic library. It
covers the addresses by joint name and their refusals; the model from the training simulator's per-world fields and
its refusals; the options taken from MuJoCo Warp's model (the tolerance raised as `put_model` raises it), the option
comparison and their refusals; the comparison of model fields; the placement against `env.placed_state`; a trial
placed again after controls and a bad state, which holds what a trial placed once holds and steps as it does; the
previous action and `xi` owned by the object, and `ctrl` read through each actuator's joint; one state chain (after several controls the object's data equal, bitwise, a plain
`mj_step` loop from the placed data); the report; pushes and lifts; the record's fields, shapes and types against the
CPU stand-in's through the same `roll_out`; the same record with one and with three threads; a bad state ending a
trial as not finite; the checks with a second MuJoCo C object as the training simulator, and their refusals; the
reports; the description's keys against `tools/rollouts.py`'s; and the command line's refusals. They leave out what
needs MuJoCo Warp and a GPU: `MujocoC.from_training` and the model taken from a built `TrackingSim`, the training
simulator's side of the checks, `main`, and the arena-full flags, which these models do not reach.
