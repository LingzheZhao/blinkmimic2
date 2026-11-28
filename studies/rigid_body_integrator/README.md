# A rigid-body integrator against BlinkMimic 2's world models: one-control error and step time

**Question.** "Before any force predictor is trained: does a rigid-body integrator, given the simulator's own recorded
contact wrenches, reproduce the simulator's next state, and what does a physics-informed world-model step cost in a
training iteration?" (from the study's declaration, written before the first training of the studies declared with
it; see Declaration). The declaration's question on DynaMimic's physics-informed world model against BlinkMimic 2's
world model is asked only if this one passes.

**Answer.**

- The integrator comparison does not pass for either recording policy with either link set (16 or 18 links), and the
  estimated training iteration is longer than the limit of 2.5 times 0.12 s, so the physics-informed world model was not
  trained.
- Fed the recorded contact wrenches without the joint-limit force, the integrator's one-control error on upright-posture
  states is below all four world models' on the single-network recovery policy's recordings and above all four on the
  tracking-only policy's.
- On low-posture states it is above all four on the single-network recovery policy's recordings; on the tracking-only
  policy's it is above all four with 16 links and below two (the single-network recovery world models) with 18 links.
- With the recorded joint-limit force added, 18 links and two 10 ms steps, the integrator's one-control error is 0.0045
  to 0.0049 on upright-posture states and 0.0082 to 0.0111 on low-posture states, against 0.0279 to 0.0365 and 0.1735 to
  0.2646 for the world models; the rule does not read this variant.
- In every comparison of table 3 the integrator's median error, 0.0000 to 0.0023, is below the world model's, 0.0098 to
  0.1027. For the integrator at two 10 ms steps without the joint-limit force, the low-posture controls with a joint
  limit (0.097 of the single-network recovery policy's low-posture controls, 0.052 of the tracking-only policy's) hold
  0.98 and 0.96 of its summed error (table 5).
- A training iteration with the physics-informed step is estimated at 0.634 s and 0.997 s (5.29 and 8.31 times 0.12 s)
  for the faster configuration of each of the two timing builds, both above the limit of 0.30 s.

The study has no training. It uses `src/blinkmimic2/rigid_body.py` (the rigid-body integrator), records simulator
wrenches with `tools/contact_wrench_recording.py`, compares the integrator and the world models with
`tools/rigid_body_step_comparison.py` and `tools/world_model_error_by_contact_class.py`, and times a physics-informed
step with `tools/world_model_step_timing.py`. DynaMimic's (arXiv 2611.32944, Section 3.2) are the 15 candidate support
links, the two integration timesteps of 10 ms per 20 ms control step, the contact decoder and the seven-step fitting;
the link sets, the four 5 ms schedule, the recordings, the timed force predictor and the batch sizes are this study's
choices. The four world models and the 0.12 s are those of [the pair study](../recovery_pair/README.md), which calls
tracking-only its "control" condition and single-network recovery its "recovery" condition; this page uses the
declaration's names, tracking-only and single-network recovery. A **control** is one 0.02 s policy step.

## Declaration

This study has no training of its own. Its declaration was written together with those of the world-model study
(`studies/world_model`, in a separate pull request of this repository) and of the mixture-of-experts (MoE) actor study
(`studies/moe_actor`, in another pull request of this repository, and
[`studies/moe_actor`](https://github.com/LingzheZhao/NablaMimic/blob/moe-actor-study/studies/moe_actor/README.md) in
NablaMimic), before the first training of any of them. This section reproduces what this page reads of it, with its
later changes applied; "Changes to the declaration" below lists those changes.

### What is compared

| Item | What it is | Whose value |
| --- | --- | --- |
| Rigid-body integrator | `RigidBody` of `src/blinkmimic2/rigid_body.py`: batched, differentiable dynamics of the trainer's robot; one integration timestep of MuJoCo's implicitfast integrator (equation below); constants from the simulator's model; checked against MuJoCo 3.11 (table 1) | this study; the form of the step follows DynaMimic's integrator (arXiv 2611.32944, Section 3.2) |
| 16 candidate support links | the 18 links below without the two elbow links; a link set has to be mirror-symmetric because the trainer fits the world model on mirrored windows | this study |
| 18 candidate support links | the 18 bodies that carry collision geoms in the trainer's model and have a non-zero near-contact share in reference playback (table 2) | this study; DynaMimic uses 15 links on its robot |
| Four 5 ms steps; two 10 ms steps | four: one integrator step per simulator physics substep, each fed that substep's recorded wrench; two: two steps of 10 ms per control, each fed the mean of two recorded 5 ms wrenches | four: this study; two: DynaMimic's two integration timesteps per control step |
| Variants that the rule reads | net contact wrenches of the candidate support links (floor and robot-robot, force in the world frame, moment about the body origin), no residual generalized force (DynaMimic's), at each schedule | the declared rule: "recorded contact wrenches and the actuator model" |
| Variants with the joint-limit force | the rule's variants plus the recorded joint-limit generalized force; not read by a rule. A further variant feeds MuJoCo's own constraint force of each physics substep, as a check of the recording | this study |
| Four world models | the world model of the last checkpoint of tracking-only and of single-network recovery, seeds 0 and 1, of the pair study; each is read with its own normalizer | the declaration |
| Recordings of get-up clips | six get-up clips of the 40 LAFAN1 clips; in each clip a trial starts every 300 rows from a noisy placement and runs 100 controls, with training noise and no termination, stepped by MuJoCo's C library in float64 | this study |
| Policies of these recordings | the averaged policy of the 7,000-iteration checkpoint of (a) single-network recovery, seed 0 (8185 upright-posture and 7915 low-posture controls); (b) tracking-only, seed 0 (8100 and 8000 controls) | this study |
| Recording of non-get-up clips | six non-get-up clips, driven by the single-network recovery seed 0 policy of (a); trials start every 1500 rows; 4900 controls, upright-posture states only | this study |
| Timing configurations | the world model, and a physics-informed step: an untrained force predictor, DynaMimic's contact decoder and the integrator with 2 or 4 integration timesteps, on a dense or a tree path | this study; the batch of 2,048 and the 32-control update are the trainer's |
| Force predictor and solves timed | a multilayer perceptron (MLP) of 1024, 1024, 1024 on the world model's inputs plus each link's height, root-frame position and world-frame velocity; solves: batched Cholesky, LᵀDL (MuJoCo's tree-sparse factorization), Cholesky column by column, lower-upper (LU) | this study |

The integrator's step is (M + dt·D)(ν⁺ − ν) = dt·(τ_act + τ_pass − h + τ_ext), with the actuators clamped to their force
range, then the position update. M is the mass matrix including rotor armature, D the damping (the passive damping plus
the actuators' damping, left out for an actuator that is clamped to its force range, as MuJoCo does), ν the generalized
velocity, ν⁺ its value after the step, dt the step length, τ_act the actuator force, τ_pass the passive force, h the
bias force of gravity, Coriolis and centrifugal terms, and τ_ext the generalized force of the fed contact wrenches (and,
in the variants with the joint-limit force, of the recorded joint-limit force). A physics substep is one 5 ms step of
the simulator, four per control; an integration timestep is one step of the integrator, of 10 ms or 5 ms.

The two timing builds differ in what is compiled, which links are used and which card runs them. The graph-broken build
compiled a whole control with torch.compile, numpy indices breaking the graph at every body, with 16 links, on a card
with 24 GB of memory. The per-timestep build compiled the force predictor and one integration timestep separately,
each as one graph, with 18 links, on a card with 48 GB of memory.

The link sets, the recordings, the variants with the joint-limit force, the generalized-force and contact-class tables,
the non-get-up clips and the use of a posture's mean over its controls as its value are not in the declaration.

### Readings

[The pair-study page](../recovery_pair/README.md) defines the readings of the rollouts; this page adds these.

| Reading | Definition | Computed by |
| --- | --- | --- |
| One-control error | for one recorded control, the mean over 70 components of the squared frame-local state error divided by a per-component scale; dimensionless; the per-control value of `train.py`'s `_wm_err` | `WorldModel.state_error` and `err_floor`, used in `tools/rigid_body_step_comparison.py` |
| Value of a posture | the mean of the one-control error over the posture's controls; a variant passes for a world model when its value is below that world model's on both postures | the `passes_rule` key of the report of `tools/rigid_body_step_comparison.py` |
| Posture of a recorded state | low-posture if the pelvis is below 0.50 m or the world z component of the pelvis up axis is below 0.5; else upright-posture | `tools/rigid_body_step_comparison.py`, `tools/world_model_error_by_contact_class.py` |
| Contact classes of a control | joint limit; robot-robot contact; non-foot floor contact; foot contact only; no contact (definitions below); share of summed error = the class's summed one-control errors over the posture's | `contact_classes`, `tools/world_model_error_by_contact_class.py` |
| Projected iteration | the duration of a training iteration with the physics-informed step (formula below); ratio = projection / 0.12 s | `--aggregate` of `tools/world_model_step_timing.py` |

The 70 components are root position, rotation, linear and angular velocity, 29 joint positions and 29 joint velocities;
a component's scale is the larger of the normalizer's `inc_norm.std()` and a lower bound. A force is present in a
control when its largest absolute entry over its four physics substeps exceeds 1e-6 N or N·m. Non-foot floor contact is
a floor wrench on a body other than the two ankle-roll links; foot contact only, one on those two links and on no other
body; no contact, no floor and no robot-robot wrench. Joint limit, robot-robot and non-foot floor contact can hold
together.

The projected iteration is 0.12 s + (policy update with the physics-informed step − policy update with the world model)
+ 5 × (one fit with the physics-informed step − one fit with the world model). The policy update is 32 controls at batch
2,048; a fit is 8 controls at batch 2,048, five per iteration (the trainer's `--wm_batches 5`, default; the timing
tool's `--fit_updates 5`, default). Each forward and backward is a CUDA-graph replay after torch.compile.

### Decision rules

The rules of the integrator question are quoted from the study's declaration with its changes applied. The second
quote is the declaration's question on DynaMimic's physics-informed world model, in which "the wider world model" is
BlinkMimic 2's world model with hidden widths 2048, 2048, 2048 in place of 1024, 1024, 1024, a condition of the
world-model study (`studies/world_model`, in a separate pull request of this repository). In both quotes a question
number is replaced by a bracketed description, and in the second a parenthesis that refers to a change by number is
replaced by "[...]". The changes follow the quotes.

> **The rigid-body integrator ([the question above]).** *Integrator comparison with recorded wrenches.* Given the
> simulator's recorded contact wrenches and the actuator model, one control of the rigid-body integrator **passes** if
> its normalized state error is below the one-control error of each of the four world models of the BlinkMimic 2 pair
> study on the same recorded states, taken separately on upright-posture states and on low-posture states; both
> comparisons have to be met. *World-model step timing.* The forward and backward time of the physics-informed
> world-model step over 32 controls is read at the batch size of the trainer's policy optimization. The duration of a
> training iteration with it is estimated by replacing, in 0.12 s (the steady duration of one training iteration in the
> BlinkMimic 2 pair study), the time of BlinkMimic 2's world-model steps by the time of the physics-informed ones. The
> physics-informed world model **is not trained** if that estimate is more than 2.5 times 0.12 s.

> **The physics-informed world model.** Only if [the question on the rigid-body integrator] passes its two rules: does
> DynaMimic's world model lower the world-model error on low-posture states and on states with non-foot contact, and
> the number of first-row trials that end in get-up windows, against BlinkMimic 2's world model given DynaMimic's link
> inputs, and against the wider world model [...]?

**Changes to the declaration.** Each change says whether results had been read when it was made.

- The declaration first named its questions, conditions and readings by earlier working names; they were replaced by
  the papers' terms and by descriptive names, with no condition, reading, threshold or rule changed. The declaration
  does not say whether results had been read when this was changed.
- Four phrases were made definite: the world-model error is classed by the robot's recorded state at the start of each
  prediction (and get-up windows by the clip); the timing rule's outcome, which first read "stop", is that the
  physics-informed world model is not trained, the iteration's duration being estimated as the rule states; 0.12 s is
  the pair study's steady duration (0.117 to 0.125 s); the rule names the four world models. The last was settled after
  the first readings of the world-model error by posture, and the declaration does not say whether before or after
  the integrator readings of this page; for the other three, it does not say whether results had been read when they
  were added.
- The question on DynaMimic's physics-informed world model first spoke of the world-model error on contact rows; it is
  written as low-posture states and states with non-foot contact. The declaration does not say whether results had
  been read when this was changed.
- The comparison with DynaMimic's physics-informed world model was kept on the recovery benchmark; this choice was made
  after the readings of the world-model error by posture and is not an outcome of the declared rule (a rule of the
  world-model study).
- The question on DynaMimic's physics-informed world model also compares it with the wider world model, besides the
  world model with link inputs as declared; added after the wider world model's results had been read. Those results
  are on the world-model study's page, not on this one. The question was not reached.

## Commands, in order

Run from the repository root. `LAFAN1` holds the 40 LAFAN1 G1 retarget CSV files of the pair study;
`RUN_TRACKING_ONLY_SEED0`, `RUN_TRACKING_ONLY_SEED1`, `RUN_SINGLE_NETWORK_SEED0`, `RUN_SINGLE_NETWORK_SEED1` are its
four training directories; `LINKS` lists the 16 or the 18 link names of table 2. Commands 1 to 4 and 6 ran on a CPU
runner (MuJoCo 3.11.0, torch 2.9.0 on the CPU, 8 threads), command 5 on one NVIDIA RTX 4090 per build.

1. Agreement with MuJoCo: `python tools/rigid_body_comparison.py --out comparison.json --motion_dir LAFAN1 --n_random
   200 --n_library 400`.
2. The recordings, one run per get-up clip and policy, with `RUN` the single-network recovery seed 0 run
   (`RUN_SINGLE_NETWORK_SEED0`, `--seed` 1 to 6) or the tracking-only seed 0 run (`RUN_TRACKING_ONLY_SEED0`, `--seed` 21
   to 26): `python tools/contact_wrench_recording.py --run_dir RUN --ckpt 7000 --motion_dir LAFAN1 --clips CLIP --stride
   300 --controls 100 --seed S --store --out CLIP.npz`. The 34 other clips ran only with the single-network recovery
   seed 0 run, without `--store` (stride 300, 100 controls, seeds 11 and 12), and six non-get-up clips with `--store`,
   stride 1500, seed 31, with the same run. Commands 2 to 4 first ran at revisions at which the tools had other file
   names and the option `--recordings` of command 4 was spelled differently (see the revisions table).
3. Floor-contact counts: `python tools/floor_contact_counts.py --motion_dir LAFAN1 --stats <14 npz files of command 2>
   --out contacts.json`.
4. The integrator comparison, for each policy's six get-up recordings and both link sets: `python
   tools/rigid_body_step_comparison.py --recordings <6 npz files> --runs RUN_TRACKING_ONLY_SEED0 RUN_TRACKING_ONLY_SEED1
   RUN_SINGLE_NETWORK_SEED0 RUN_SINGLE_NETWORK_SEED1 --links LINKS --out comparison.json`; and, with 18 links, the same
   command with `--recordings other-clips.npz`, the one recording of the six non-get-up clips.
5. World-model step timing, one process per timing configuration `ROW` (a configuration name of the tool, such as
   `physics_informed_2_substeps_tree_columns`), then the aggregate: `python tools/world_model_step_timing.py
   --fullgraph --compile_unit physics_substep --rows ROW --links LINKS --out row-ROW.json`, then `python
   tools/world_model_step_timing.py --aggregate row-*.json --out timing.json`. In the option value `physics_substep`
   and in the configuration names, "substep" means one integration timestep of the integrator. The per-timestep build
   (card of 48 GB) ran this with 18 links. The graph-broken build (card of 24 GB) ran `python
   tools/world_model_step_timing.py --rows ROW --links LINKS --out row-ROW.json` (a whole control compiled; the tool's
   file name and the configuration names were different then) with 16 links, at a revision whose timing tool and
   integrator module differ from this branch's (see the revisions table). This branch's integrator holds the body and
   dof indices as Python integers, where the numpy indices of that revision broke the compiled graph; the per-timestep
   build compiled one integration timestep of it with `--fullgraph`, under which a graph break raises an error. No
   timing of this branch's code with a whole control compiled is reported, so the page does not show whether this
   code reproduces the graph-broken build.
6. Error by contact class with 18 links, on each policy's six recordings: `python
   tools/world_model_error_by_contact_class.py --recordings POLICY:FILE1.npz,...,FILE6.npz --runs NAME:RUN ... --links
   LINKS --out controls.json`. `--runs` lists the four world models above and two single-network recovery world models
   of hidden widths 2048, 2048, 2048 (seeds 0 and 1), the first trainings of the wider world model of the world-model
   study (`studies/world_model`, in a separate pull request of this repository), which the rule does not use.

## What the study does not show

- **A trained force predictor.** None was trained, and DynaMimic's wrench loss, residual regularization and seven-step
  fitting were not implemented. The timed predictor's input (the world model's inputs plus each link's height,
  root-frame position and world-frame velocity) and width are this study's choices, and its output has 467 values
  against DynaMimic's 395. The comparison feeds the simulator's wrenches and shows nothing about predictor accuracy.
- **Timing configurations without a time.** No hand-fused kernel was written or timed. Batched Cholesky and LU failed in
  CUDA-graph capture; a whole control compiled as one graph gave no time within about half an hour and was stopped. The
  two timing builds differ in link set, card memory and compile unit at once; which of these makes the per-timestep
  build slower was not measured.
- **Other conditions.** No analytic joint-limit or robot-robot contact model was added; the variants with the
  joint-limit force feed the recorded force. Flat ground, one robot (mjlab's Unitree G1, self-collision on, frictionless
  non-foot geoms), two policies on six get-up clips; the six non-get-up clips were read on upright-posture states only.
  The recordings are stepped by MuJoCo's C library and the world models were trained on MuJoCo Warp (an earlier
  comparison, by a program that is not part of this repository, gives root-position differences up to 1e-6 m standing
  and 1.7e-4 m lying after one control). The world models were read in float32 on the CPU; their training ran the MLP
  in bf16 on the card.

## Results

**Revisions and cards.** The recordings, counts, comparisons and timings ran while the code was being written, at the
revisions below. These revisions are not in this branch's history, which holds the final code in a few commits. The
last column says, from a diff restricted to the code that the row's commands execute, how each revision differs from
this branch's code.

| Revision that ran | What it trained, computed or read | How its code differs from this branch's, in the code those commands execute |
| --- | --- | --- |
| 2e75d101ccbc | the recordings of both policies on the six get-up clips, and of the single-network recovery policy on the 34 other clips (command 2) | The recording tool had another file name, and one stored key (whether a trial was still running at the end) and one word of its progress log had other names; its code, and that of the model builder it calls, is otherwise this branch's. No tool or table reads that key. |
| 0596dd2ca87c | floor-contact counts (command 3, table 2); the four integrator comparisons and their generalized forces (command 4, tables 3 and 4) | The two tools had other file names, and the comparison's recordings option, row names and report keys had other names; the counting tool's code is otherwise this branch's. A recording set without one of the two postures then stopped the comparison with an error, where it now leaves that posture out; these recordings hold both postures, so tables 3 and 4 do not depend on it. In the integrator, a solve other than batched Cholesky then ran LU on the dense path and one other than LᵀDL ran batched Cholesky on the tree path (so a tree-path LU solve ran Cholesky), where now each solve runs its own method and an unknown one raises an error, and the body and dof indices were numpy integers where they are now Python integers of the same values; these commands use the dense path with batched Cholesky, which runs the same calls, and no cited result uses an LU solve on the tree path. |
| 3d17ffac74d0 | timing, graph-broken build, 16 links (command 5, table 6) | The timing tool had another file name, and its configuration names, class name and report keys and the three options of the baseline duration, the fit's length and the fits per iteration had other names; the integrator module is that of the row above, with the same differences (no graph-broken configuration uses an LU solve); its numpy body and dof indices broke the compiled graph at every body, which defines the graph-broken build, so table 6's graph-broken times depend on them; this branch's code holds Python integers there, and no timing of it with a whole control compiled is reported, so the page does not show whether it reproduces that build. The timing tool then read the peak memory after the eager policy update and the eager fit (now after the eager policy update alone, so table 6's peak memory of the graph-broken rows covers both), ran the eager fit before the policy's CUDA-graph capture and also when that capture failed (now after it, and only when it did not fail), counted the compiled graphs before the captures (now after the eager fit), checked the accuracy on the dense path with batched Cholesky only (now also on the tree path with LᵀDL and with the column-by-column solve) and wrote its report only at the end (now after every measurement). It had no column-by-column solve and neither `--fullgraph` nor `--compile_unit`, whose defaults compile a whole control as it did. |
| 630ae08b5c48 | timing, per-timestep build, 18 links (command 5, table 6); the recording of the six non-get-up clips (command 2) | None in the code: the timing tool, the integrator module and the recording tool are this branch's; only comments, docstrings and option help differ. |
| 92d503b408ea | comparison with MuJoCo (command 1, table 1); comparison on the non-get-up recording (command 4) | None in the code: the two comparison tools and the integrator module are this branch's; only comments and docstrings differ. |
| 2bf4e049c16e | the error by contact class (command 6, table 5) | None in the code: the contact-class tool, the comparison tool it imports and the integrator module are this branch's; only comments and docstrings differ. |

Command 5 ran on RTX 4090 cards of 24,564 MiB (24.0 GiB, graph-broken build) and 49,140 MiB (48.0 GiB, per-timestep
build). A tree-path timing run was cancelled after a CUDA-graph capture failure, and a first timing run, with the timing
tool's first version, failed at start; neither gave a time.

**Table 1.** Agreement with MuJoCo: the largest relative difference over 200 random states (79 with the pelvis below 0.5
m) and 400 library states (203 with the pelvis below 0.5 m), float64. In float32 the quantities differ by at most
2.6e-06 and the velocity after one physics substep by at most 1.0e-05. A variant of the integrator that keeps the
damping of the clamped actuators differs from MuJoCo by 1.1e+00 in that velocity.

| mass matrix | bias force | body Jacobians | actuator force | passive force | velocity after one `mj_step` fed `qfrc_constraint` |
| --- | --- | --- | --- | --- | --- |
| 3.5e-16 | 5.6e-15 | 2.0e-15 | 1.5e-15 | 0.0 absolute | 1.9e-14 (dense path), 2.2e-14 (tree path, largest of three solves) |

**Table 2.** The 18 candidate support links: floor-force share (a body's summed floor contact force norm over the
recorded physics substeps over the sum over all bodies; 128,800 substeps on the get-up clips, both policies' recordings
pooled, 660,000 on all 40 clips), share of substeps in floor contact, and share of the 46,734 get-up library rows with a
penetrating reference pose. 16 links leave out the two elbow links and cover 0.9965 of the floor force over all clips
and 0.9837 over get-up clips:

| Body | Floor force, get-up / all | Contact, get-up | Penetrating | Body | Floor force, get-up / all | Contact, get-up | Penetrating |
| --- | --- | --- | --- | --- | --- | --- | --- |
| left_ankle_roll_link | 0.3233 / 0.4612 | 0.590 | 0.227 | right_ankle_roll_link | 0.3114 / 0.4498 | 0.584 | 0.251 |
| pelvis | 0.0543 / 0.0132 | 0.088 | 0.069 | left_hip_roll_link | 0.0422 / 0.0088 | 0.072 | 0.076 |
| torso_link | 0.0286 / 0.0072 | 0.045 | 0.055 | left_knee_link | 0.0212 / 0.0071 | 0.045 | 0.084 |
| right_knee_link | 0.0223 / 0.0067 | 0.040 | 0.079 | right_wrist_yaw_link | 0.0251 / 0.0065 | 0.220 | 0.127 |
| left_wrist_yaw_link | 0.0249 / 0.0062 | 0.207 | 0.136 | right_hip_roll_link | 0.0257 / 0.0059 | 0.043 | 0.052 |
| left_wrist_pitch_link | 0.0250 / 0.0057 | 0.116 | 0.063 | right_wrist_pitch_link | 0.0235 / 0.0053 | 0.103 | 0.072 |
| left_hip_yaw_link | 0.0185 / 0.0042 | 0.049 | 0.097 | right_hip_yaw_link | 0.0171 / 0.0040 | 0.045 | 0.087 |
| left_shoulder_yaw_link | 0.0111 / 0.0025 | 0.046 | 0.072 | right_shoulder_yaw_link | 0.0095 / 0.0022 | 0.034 | 0.069 |
| left_elbow_link | 0.0081 / 0.0018 | 0.036 | 0.065 | right_elbow_link | 0.0081 / 0.0018 | 0.030 | 0.069 |

**Table 3.** The integrator comparison with 18 links: the one-control error (dimensionless; mean / median) of each world
model and of the integrator variants, upright-posture ; low-posture states, on the recordings of the single-network
recovery policy (8185 and 7915 controls) and of the tracking-only policy (8100 and 8000). The columns "four 5 ms steps"
and "two 10 ms steps" are the variants that the rule reads; "+ joint-limit force" adds the recorded joint-limit force.
MuJoCo's own constraint force reads 0.0000 everywhere; the largest absolute difference between it and the sum of the
recorded floor, robot-robot and joint-limit forces is 9.1e-13 (N or N·m):

| Recordings | World model | World model, upright ; low | four 5 ms steps | two 10 ms steps | four 5 ms steps + joint-limit force | two 10 ms steps + joint-limit force |
| --- | --- | --- | --- | --- | --- | --- |
| single-network recovery policy | tracking-only, seed 0 | 0.0344 / 0.0136 ; 0.2646 / 0.1010 | 0.0039 / 0.0000 ; 0.5861 / 0.0000 | 0.0088 / 0.0018 ; 0.6276 / 0.0023 | 0.0000 / 0.0000 ; 0.0000 / 0.0000 | 0.0049 / 0.0018 ; 0.0111 / 0.0021 |
|  | tracking-only, seed 1 | 0.0339 / 0.0140 ; 0.2551 / 0.1027 | 0.0037 / 0.0000 ; 0.5684 / 0.0000 | 0.0084 / 0.0017 ; 0.6110 / 0.0022 | 0.0000 / 0.0000 ; 0.0000 / 0.0000 | 0.0047 / 0.0017 ; 0.0107 / 0.0020 |
|  | single-network recovery, seed 0 | 0.0365 / 0.0143 ; 0.1735 / 0.0646 | 0.0045 / 0.0000 ; 0.4277 / 0.0000 | 0.0090 / 0.0018 ; 0.4592 / 0.0021 | 0.0000 / 0.0000 ; 0.0000 / 0.0000 | 0.0045 / 0.0018 ; 0.0085 / 0.0019 |
|  | single-network recovery, seed 1 | 0.0356 / 0.0143 ; 0.1932 / 0.0730 | 0.0045 / 0.0000 ; 0.4130 / 0.0000 | 0.0090 / 0.0018 ; 0.4447 / 0.0021 | 0.0000 / 0.0000 ; 0.0000 / 0.0000 | 0.0045 / 0.0018 ; 0.0087 / 0.0019 |
| tracking-only policy | tracking-only, seed 0 | 0.0292 / 0.0098 ; 0.2162 / 0.0653 | 0.0406 / 0.0000 ; 0.2322 / 0.0000 | 0.0311 / 0.0017 ; 0.2496 / 0.0021 | 0.0000 / 0.0000 ; 0.0000 / 0.0000 | 0.0048 / 0.0016 ; 0.0105 / 0.0020 |
|  | tracking-only, seed 1 | 0.0279 / 0.0099 ; 0.2195 / 0.0748 | 0.0386 / 0.0000 ; 0.2248 / 0.0000 | 0.0295 / 0.0016 ; 0.2427 / 0.0020 | 0.0000 / 0.0000 ; 0.0000 / 0.0000 | 0.0046 / 0.0016 ; 0.0102 / 0.0019 |
|  | single-network recovery, seed 0 | 0.0340 / 0.0121 ; 0.1942 / 0.0763 | 0.0481 / 0.0000 ; 0.1764 / 0.0000 | 0.0355 / 0.0017 ; 0.1876 / 0.0019 | 0.0000 / 0.0000 ; 0.0000 / 0.0000 | 0.0046 / 0.0017 ; 0.0082 / 0.0019 |
|  | single-network recovery, seed 1 | 0.0333 / 0.0121 ; 0.1958 / 0.0766 | 0.0473 / 0.0000 ; 0.1725 / 0.0000 | 0.0349 / 0.0016 ; 0.1842 / 0.0019 | 0.0000 / 0.0000 ; 0.0000 / 0.0000 | 0.0046 / 0.0016 ; 0.0084 / 0.0019 |

With 16 links the variants' low-posture means are 0.5006 to 0.6776 (four 5 ms steps) and 0.5300 to 0.7162 (two 10 ms
steps) on the single-network recovery policy's recordings and 0.2152 to 0.2770 and 0.2253 to 0.2926 on the tracking-only
policy's. On the six non-get-up clips, driven by the single-network recovery seed 0 policy (4900 controls,
upright-posture states only, 18 links, two 10 ms steps) the integrator reads 0.0058 to 0.0064 against 0.0297 to 0.0348
for the world models; no rule outcome is read here.

**Table 4.** Generalized forces on the joints in the recordings (the same in the comparisons with 16 and with 18 links;
root-mean-square (RMS) over controls and physics substeps, largest entry, N·m; share of controls with a non-zero force):

| Recordings | Posture | Robot-robot: RMS / largest / share | Joint limit: RMS / largest / share |
| --- | --- | --- | --- |
| single-network recovery policy | upright-posture states | 1.46 / 160.8 / 0.112 | 0.26 / 48.8 / 0.026 |
| single-network recovery policy | low-posture states | 2.65 / 181.4 / 0.299 | 0.77 / 107.7 / 0.097 |
| tracking-only policy | upright-posture states | 0.91 / 133.8 / 0.121 | 0.68 / 180.7 / 0.021 |
| tracking-only policy | low-posture states | 2.50 / 219.2 / 0.245 | 0.87 / 272.3 / 0.052 |

**Table 5.** One-control error by contact class (18 links; the world model of tracking-only seed 0 and the integrator
variant at two 10 ms steps without the joint-limit force; the other three world models and the other variants are not
shown): the mean over the posture's controls, then each class's share of the summed error. A row's shares add to more
than 1 because the classes overlap. "Share of controls": foot contact only / non-foot floor / joint limit / robot-robot
/ no contact:

| Recordings, posture (controls) | Share of controls | Model | All | Foot only | Non-foot floor | Joint limit | Robot-robot | No contact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single-network recovery policy, upright-posture states (8185) | 0.979 / 0.001 / 0.026 / 0.112 / 0.016 | world model | 0.034 | 0.93 | 0.00 | 0.05 | 0.29 | 0.04 |
| | | integrator | 0.009 | 0.90 | 0.01 | 0.48 | 0.14 | 0.07 |
| single-network recovery policy, low-posture states (7915) | 0.106 / 0.889 / 0.097 / 0.299 / 0.004 | world model | 0.265 | 0.07 | 0.92 | 0.13 | 0.31 | 0.01 |
| | | integrator | 0.628 | 0.02 | 0.98 | 0.98 | 0.32 | 0.00 |
| tracking-only policy, upright-posture states (8100) | 0.977 / 0.001 / 0.021 / 0.121 / 0.018 | world model | 0.029 | 0.89 | 0.03 | 0.12 | 0.26 | 0.04 |
| | | integrator | 0.031 | 0.98 | 0.00 | 0.86 | 0.25 | 0.01 |
| tracking-only policy, low-posture states (8000) | 0.111 / 0.885 / 0.052 / 0.245 / 0.003 | world model | 0.216 | 0.06 | 0.90 | 0.14 | 0.32 | 0.03 |
| | | integrator | 0.250 | 0.03 | 0.96 | 0.96 | 0.11 | 0.00 |

**Table 6.** Timing in seconds of a physics-informed step against the world model's 0.12 s (limit 0.30 s; ratio =
projection / 0.12 s; "policy alone" replaces only the policy update; the pair study's single-network recovery training,
seed 0, peaked at 13.35 GiB, 13,672 MiB, in a log of card memory that is not part of this repository); the builds are
those defined in the declaration. The peak memory is read after the eager policy update in the per-timestep build and
after the eager policy update and the eager fit in the graph-broken build:

| Build, configuration | Policy update | One fit | Projection | Ratio | Policy alone: projection / ratio | Peak memory of the eager runs (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| graph-broken, world model | 0.0099 | 0.0032 | 0.120 | 1.00 | 0.120 / 1.00 | 0.65 |
| graph-broken, 2 integration timesteps, tree path, LᵀDL | 0.2463 | 0.0588 | 0.634 | 5.29 | 0.356 / 2.97 | 5.05 |
| graph-broken, 4 integration timesteps, tree path, LᵀDL | 0.4369 | 0.1063 | 1.062 | 8.85 | 0.547 / 4.56 | 8.95 |
| graph-broken, 2 integration timesteps, tree path, batched Cholesky | no time (graph capture failed) | | | | | 4.91 |
| graph-broken, 2 integration timesteps, dense path, batched Cholesky | no time (graph capture failed) | | | | | 16.46 |
| graph-broken, 4 integration timesteps, dense path, batched Cholesky | no time (graph capture failed; out of memory in the eager reference) | | | | | 21.95 |
| per-timestep, 2 integration timesteps, LU | no time (graph capture failed) | | | | | 16.63 |
| per-timestep, world model | 0.0096 | 0.0032 | 0.120 | 1.00 | | 0.65 |
| per-timestep, 2 integration timesteps, tree path, columns | 0.4048 | 0.0996 | 0.997 | 8.31 | 0.515 / 4.29 | 16.01 |
| per-timestep, 2 integration timesteps, tree path, LᵀDL | 0.6133 | 0.1474 | 1.445 | 12.04 | 0.724 / 6.03 | 4.84 |
| per-timestep, 4 integration timesteps, tree path, columns | 0.7771 | 0.1938 | 1.841 | 15.34 | 0.887 / 7.40 | 31.23 |
| per-timestep, 2 integration timesteps, dense path, columns | 0.5373 | 0.1318 | 1.291 | 10.76 | 0.648 / 5.40 | 27.85 |
| per-timestep, 2 integration timesteps, tree path, columns, TF32 (TensorFloat-32) off | 0.4155 | 0.1028 | 1.024 | 8.53 | 0.526 / 4.38 | 15.99 |
| per-timestep, 4 integration timesteps, tree path, LᵀDL | 1.2116 | 0.2969 | 2.791 | 23.26 | 1.322 / 11.02 | 8.62 |

**Outcome: integrator comparison with recorded wrenches does not pass.** The rule reads net contact wrenches at four 5
ms and at two 10 ms steps, for each of the four world models, on both postures. A setting is one recording policy with
one link set. No variant passes for any world model in any of the four settings (0 of 8 pairs of world model and
schedule per setting). On upright-posture states a variant is below 4 of 4 world models on the single-network recovery
policy's recordings and below 0 of 4 on the tracking-only policy's, in all four settings. On low-posture states it is
below 0 of 4 on the single-network recovery policy's recordings; on the tracking-only policy's recordings it is below 0
of 4 with 16 links and below 2 of 4 (the two single-network recovery world models) with 18 links. The counts are the
same at both schedules.

**Outcome: world-model step timing, the physics-informed world model is not trained.** The estimate is more than 2.5
times 0.12 s = 0.30 s for both timing builds: smallest 0.634 s (5.29 times) and 0.997 s (8.31 times); with only the
policy update replaced, 0.356 s (2.97 times) and 0.515 s (4.29 times). The question on DynaMimic's physics-informed
world model is read only if the integrator question passes its two rules (quoted above); neither is met, so the
physics-informed world model was not trained and `train.py` is unchanged.

**Exploratory (added after results).** Table 5 and the non-get-up comparison come from revisions later than the
integrator comparison; table 4 comes from the same files as table 3. None was declared or is read by a rule.

**What the results do not show.** Beyond the limits above: a cause of the variants' error on low-posture states (the
contact classes overlap), or a rule outcome for the joint-limit variants or the non-get-up clips.
