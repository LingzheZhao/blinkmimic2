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
`tools/evaluate.py` builds it, with T the number of trials:

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

### Added entities, sensors and a function after each substep

`TrackingSim` takes three more keyword arguments; `tools/rollouts.py` passes none of them, and without them the
simulator's mjlab scene and its step are the release's.

- `scene_entities` (name -> mjlab `EntityCfg`) and `scene_sensors` (mjlab `SensorCfg`s) are added to the mjlab scene
  by `make_scene_cfg` after the plane terrain and the robot, so the robot keeps its indices in the compiled model. The
  names `robot` and `terrain` are refused. The randomizer writes the friction of the robot's geoms and of the world
  body's geoms only ([`dr.py`](src/blinkmimic2/dr.py)), so an added entity's geoms keep their own contact
  parameters. Camera and ray-cast sensors are refused, since `TrackingSim` does not run mjlab's sense pipeline.
  `reset_masked` resets the robot only and refuses an mjlab scene with added entities or sensors.
- `after_substep` is called with the simulator after each of the four physics substeps of a step, after
  `scene.update` (which clears the cache of mjlab's sensor data) and before the sensing filter's update. MuJoCo's
  step computes the derived quantities, sensor data included, before it integrates, so the sensor data the function
  reads describe the state at the start of that substep, not the state after it. It runs inside `step_core`; to keep
  the step capturable in a CUDA graph, the function must write in place into tensors allocated before the capture.

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
  a run's `args.json` is the folder's clips; a folder without motion files is refused;
- the added entities, sensors and function: with the defaults, `make_scene_cfg` and `TrackingSim` give the release's
  mjlab scene configuration field by field, and MuJoCo compiles it on the CPU to the release's model byte for byte; an
  added mocap box and contact sensor leave the robot's part of the compiled model unchanged; the names `robot` and
  `terrain`, `reset_masked` with additions, and camera or ray-cast sensors are refused, and `reset_masked` passes its
  new check on the default scene; `TrackingSim.step_core` without a function makes the release's calls in the
  release's order, and with one calls it after each scene update and before the sensing filter's update and returns
  the same values. The release's scene configuration is written out in the test, not run from the release's code.

They leave out what needs the simulator and a GPU: `rollouts.Simulator` (the placement through `TrackingSim.reset`,
pushes written to the simulator, the command row held through `TrackingSim.step`, and the reading of MuJoCo Warp's
counts and overflow bits), `main` (loading the run, the checkpoint and the library, the rounding of the products
under TF32, and the files written), `train.py` passing `--motion_dir` to the library (it trains once parsed), and a
step on MuJoCo Warp with added entities, sensors or a function (its model and data with them, a mocap pose written
per world, the sensor data read in the function, the step captured in a CUDA graph).
