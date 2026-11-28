# StableMimic's MoE actor, a wider single network and the training length in recovery training

**Question.** Does recovery training with StableMimic's mixture-of-experts (MoE) actor and its mixing losses lower the
tracking cost relative to recovery training with the single network, while the policy still gets up? Is the tracking
cost a matter of training length? The tracking cost of a recovery condition is its mean tracking error over the 34 clips
used as commands, minus that of tracking-only training (the same recipe without recovery) at the same seed and length.

**Answer.**

- **The MoE actor.** The declared rule, read at three seeds, gives "undecided" for MoE recovery (recovery training with
  StableMimic's MoE actor), on its first and on its second trainings, and for MoE recovery with full-width experts (each
  expert as wide as the single network's decoder; added after results). Every MoE condition keeps getting up.
- **What the rule read.** Tracking cost of single-network recovery minus that of the MoE condition at seeds 0, 1 and 2:
  -0.00016, +0.00109 and -0.00405 m (first trainings); -0.00168, +0.00342 and -0.00405 m (second trainings); +0.00116,
  +0.00448 and +0.00040 m (full-width experts). The rule asks for at least 0.003 m at every seed to lower the cost and
  for less than 0.001 m at every seed not to lower it.
- **Five seeds (exploratory; the reading was declared before seeds 2 to 4 and the wider decoder were trained).** MoE
  recovery with full-width experts against single-network recovery with a wider decoder, which has at least as many
  actor parameters: the mean over five seeds of the wider decoder's cost minus the MoE condition's is +0.00001 m,
  positive at 3 of 5 seeds, "does not lower it". Against single-network recovery, the mean over five seeds of
  single-network recovery's cost minus the cost of MoE recovery with full-width experts (+0.00174 m) and minus that of
  the wider decoder (+0.00173 m) each reads "undecided", positive at 4 of 5 seeds.
- **Per condition, without a rule** (mean and standard deviation of the tracking error over the 34 commanded clips,
  number of trainings): tracking-only 0.02734 and 0.00021 m (5); single-network recovery 0.03414 and 0.00134 m (9, four
  of them made outside this study, two in the pair study and two in the world-model study, `studies/world_model`, in
  a separate pull request of this repository); MoE recovery 0.03483 and 0.00173 m (5); MoE recovery with full-width
  experts 0.03232 and 0.00085 m (5); single-network recovery with a wider decoder 0.03233 and 0.00091 m (5).
- **Training length.** The rule reads "undecided": the tracking cost of single-network recovery at twice the training
  length is 0.989 times its cost at the declared length at seed 0 and 0.782 times at seed 1. The sensitivity reading
  (added after results; each condition's value at a seed replaced by the mean over its trainings with identical
  arguments) reads "is not", with ratios 0.981 and 0.891.
- **How far trainings differ.** Trainings with identical arguments at one seed differ in tracking error over the 34
  commanded clips by up to 0.00311 m. In the supplementary reading (added after results, exploratory), the tracking
  errors of one training at its last checkpoint and at the two checkpoints 500 and 1,000 iterations before it span up to
  0.00224 m (median 0.00069 m over 33 trainings). On that reading, with this study's own trainings of tracking-only and
  single-network recovery, the MoE rule reads "does not lower it" for the first trainings and "undecided" for the
  second; with the sensitivity reading it reads "does not lower it" for both (Table 11).
- **Seconds per training iteration** (Table 7): single-network recovery 0.122 to 0.126; MoE recovery 0.141 to 0.145;
  single-network recovery with a wider decoder 0.145 to 0.151; MoE recovery with full-width experts 0.172 to 0.179.

This study trains BlinkMimic 2 with `python -m blinkmimic2.train`, rolls out the checkpoints with `tools/rollouts.py` on
the scene of the pair study, reads the rollouts with `studies/recovery_pair/readings.py` and with
`studies/recovery_pair/groups.py`, added by the pull request this one depends on, and applies the declared rules to the
readings with a program that is not part of this repository; Tables 1 and 10 hold the per-seed values it reads, so that
a reader can recompute each rule outcome from them. The MoE actor, its mixing network (StableMimic's
"gate"), its three mixing losses and their weights 0.1, 0.01 and 0.01 are StableMimic's (arXiv 2608.02385, Section III-C
and the implementation-details table); the softmax temperature, the forms of two of the losses, the mixing network's
widths, the experts' widths, the training lengths, the seeds and every threshold are this study's choices, as the
declaration below gives them. The page continues the [pair
study](../recovery_pair/README.md), which trained at two seeds a "control" condition and a "recovery" condition: they
are tracking-only and single-network recovery here, the names this page uses from here on. As on that
page, a **control** is one 0.02 s policy step, a **row** one 0.02 s frame of the motion library and a **command** the
row the policy reads. R counts the 100 placed starts that get up, J those that rejoin their command, F the 100 twins
with a counted fall and E is the aligned body-position error in metres. NablaMimic is the
[fork](https://github.com/LingzheZhao/NablaMimic) that trains through a differentiable physics simulator and asks the
same questions in its MoE-actor study,
[`studies/moe_actor`](https://github.com/LingzheZhao/NablaMimic/blob/moe-actor-study/studies/moe_actor/README.md). A
wider decoder has hidden widths 3200, 1600, 800 and 400. The tracking error over the 34 commanded
clips is the mean of E over the 34 clips that recovery training uses as commands, and the tracking cost of a recovery
condition is its tracking error over the 34 commanded clips minus that of tracking-only at the same seed and training
length.

## Declaration

### Conditions

Three names are defined here. **MoE recovery** is single-network recovery with `--moe`: StableMimic's MoE actor in
place of the single network's decoder, and the three mixing losses added to the actor's loss. **MoE recovery with
full-width experts** is MoE recovery with each expert's hidden widths set to the widths of the single network's
decoder. **Single-network recovery with a wider decoder** is single-network recovery with decoder hidden widths 3200,
1600, 800 and 400 in place of 2048, 1024, 512 and 256. The MoE actor as implemented is described in
[RECOVERY.md](../../RECOVERY.md), "Mixture-of-experts actor (`--moe`)"; what a reader needs to follow the conditions
is this.

- **Experts.** The tracking expert and the recovery expert are two decoders that read the latent of the shared motion
  encoder (32 values) and the 10 normalized observations. Their hidden widths are 1024, 512, 256, 128 in MoE recovery,
  half of the single network's decoder, as StableMimic's experts have half the width of its single network, and 2048,
  1024, 512, 256 with full-width experts, which this study added.
- **Mixing network.** A multilayer perceptron (MLP) with hidden widths 256 and 128 (this study's choice) on the first
  93 entries of each of the 10 normalized observations, so 930 inputs: joint positions, joint velocities, angular
  velocity, gravity in the body frame and previous action, which are StableMimic's proprioceptive inputs. It reads no
  goal, latent, reference-relative orientation or regime. Its two outputs give the mixing weights by a softmax at
  temperature 1.0 (StableMimic gives none; 1.0 is this study's), and the action is the sum of the experts' actions
  weighted by the mixing weights.
- **Mixing target.** Training only: (1, 0) on tracking controls, (0, 1) on recovery controls (written as (weight of
  the tracking expert, weight of the recovery expert)) and, over the 75 controls of the return to the command
  (StableMimic's "transition", this repository's blend), a linear change from the recovery expert to the tracking
  expert (StableMimic's regimes; the linear form is this study's). An episode placed with its pelvis at 0.8 or more of
  its command row's pelvis height keeps the tracking target (StableMimic's rule).
- **Mixing losses.** Cross-entropy of the mixing weights to the mixing target, weight 0.1, with blend controls
  weighted 4 inside it; consistency of the mixing weights between consecutive controls of one regime, weight 0.01;
  alignment of the experts' actions on blend controls, weight 0.01. The three weights are StableMimic's; the weight 4
  is StableMimic's "transition 4.0", which StableMimic does not define and this study reads as the blend weight; the
  forms of consistency and alignment are this study's.
- **Actor parameters.** Single network (decoder 2048, 1024, 512, 256): 9,116,477 (encoder 4,259,616; decoder
  4,856,861). MoE actor with half-width experts: 8,011,740 (each expert 1,740,317; mixing network 271,490). MoE actor
  with full-width experts: 14,244,828 (each expert 4,856,861). Single network with decoder 3200, 1600, 800, 400:
  14,267,645; a test checks that this has at least as many parameters as the MoE actor with full-width experts and
  that decoder 3192, 1596, 798, 399 (14,225,867) has fewer.

Every training is `python -m blinkmimic2.train --seed SEED --bones_list= --motion_dir DIR` with the pair study's
`--bones_lib`, a preset of [`scripts/presets.sh`](../../scripts/presets.sh) and `--ckpt_every 500`, 7,000 iterations
unless stated.

| Condition | What it changes, against which args or preset | Seeds | Declared before the trainings, or added after which result |
| --- | --- | --- | --- |
| tracking-only | `PRESET[recovery_control]` | 0 to 4 | declared; seeds 2 to 4 as the sentence below the table says |
| single-network recovery | `PRESET[recovery]`: tracking-only plus the seven recovery options | 0 to 4 | declared; seeds 2 to 4 as the sentence below the table says |
| MoE recovery | `PRESET[recovery]` plus `--moe`; first training and second training at seeds 0 and 1, one training at seed 2 | 0 to 2 | first trainings declared; second trainings at seeds 0 and 1 added after the first trainings, for the reason below the table; seed 2 as the sentence below the table says |
| MoE recovery with full-width experts | `PRESET[recovery]` plus `--moe --moe_expert_hidden 2048,1024,512,256` | 0 to 4 | added after results: NablaMimic's MoE recovery had been read, for the reason below the table |
| tracking-only at twice the training length | `PRESET[recovery_control]` plus `--iterations 14000`, every schedule option unchanged | 0, 1 | declared |
| single-network recovery at twice the training length | `PRESET[recovery]` plus `--iterations 14000`, every schedule option unchanged | 0, 1 | declared |
| single-network recovery with a wider decoder | `PRESET[recovery]` plus `--dec_hidden 3200,1600,800,400` | 0 to 4 | added after results, declared before it was trained, for the reason below the table |

Seeds 0 and 1 were declared. Seed 2 is the third seed that the rule asks for when its reading is undecided. Seeds 3
and 4 were added after results and declared before they were trained: they are trained for tracking-only,
single-network recovery, MoE recovery with full-width experts and single-network recovery with a wider decoder, not
for MoE recovery. MoE recovery is trained twice at seeds 0 and 1 because, after the first trainings, the mixing
network's attribute in the policy, stored checkpoint keys, per-episode flag and log columns were given other names in
the code; the second trainings come from code with this branch's names (revision 55e1cddc68e2) with the design,
options, seeds and card model unchanged, and the rules are read on each. MoE recovery with full-width experts
was added because, on NablaMimic, the tracking error of MoE recovery had been read and its tracking cost was higher
than that of single-network recovery at both seeds while its mixing weights followed the regimes; halving the experts'
widths also halves the capacity that acts on tracking controls, and the added condition separates that from the
mixture. The wider decoder is for a single network whose actor has at least as many parameters as the MoE actor with
full-width experts; StableMimic compares its MoE actor with a wider single network for the same reason. Two trainings
of single-network recovery made outside this study, at seeds 0 and 1 each, enter the sensitivity reading, the
per-condition report (Table 4), the comparison of repeated trainings (Table 5) and the supplementary reading, and no
declared rule: the pair study's training and the world-model study's training of single-network recovery. Their
arguments are identical to single-network recovery's; the pair study's checkpoint interval is 100 iterations.

### Readings

The pair-study page defines R, J, F, C and E. This page adds these, computed by `clip_group`, `read_set` and `pooled`
of `studies/recovery_pair/groups.py` (added by the pull request this one depends on) from the records of
`tools/rollouts.py` and `readings.py`, and by a program that is not part of this repository for the rest, whose
inputs are the values of Tables 1 and 10.

- **Tracking error over the 34 commanded clips** (m): the mean over the commanded clips of each clip's E, read up to
  the clip's first termination, in one set of first-row rollouts (`clip_group`). The value of a checkpoint in a rule
  is the mean over its three sets of rollouts; the range of the three sets is reported. The spread between the sets of
  one checkpoint is the largest minus the smallest of the three.
- **Excluded clips.** The six get-up clips that recovery training does not use as commands (the five hidden get-up
  clips and fallAndGetUp1_subject4): the clips completed and the mean of their E, in the same way.
- **Ends per get-up window.** A get-up window is the rows from 50 before to 50 after the end of a sequence of at least
  25 consecutive low rows of a clip. A first-row trial enters a window unless it ended before it, and ends in it when
  its termination falls inside it. Counts are given as ends in windows entered, for the five hidden clips, for
  fallAndGetUp1_subject4 and for the commanded clips with a window, summed over the three sets (`read_set`, `pooled`),
  so they are not independent.
- **Primary reading** (added after results, with the sensitivity reading). The reading of this study's own trainings:
  for tracking-only and single-network recovery, those trained beside the MoE conditions. **Sensitivity reading** (added
  after results). The same comparison with each condition's value at a seed replaced by the mean over all of its
  completed trainings with identical arguments at that seed. The pair study's trainings differ from the others in one
  argument, the interval between checkpoints (100 iterations against 500), and count as trainings with identical
  arguments here.
- **Supplementary value** (added after results; exploratory). The mean of three values: the tracking error at a
  training's last checkpoint (mean of its three sets) and at the checkpoints 500 and 1,000 iterations before it (one
  set each).
- **Per condition** (added after results, without a rule). The number of completed trainings and the mean and standard
  deviation (divisor n - 1, n the number of trainings) of their tracking errors over the 34 commanded clips, at the
  last checkpoint.

### Decision rules

The study's declaration was written before the first training. Its rules are quoted below, with each reference to a
change replaced by "[...]" and each reference to one of the two questions above replaced by the question in brackets.
The thresholds of both rules are those of the declaration as first written.

**The MoE actor.** The text is the declared rule with later changes applied. Its sentence on a third seed, the
outcome "undecided" and the answer to the question were made definite after the first readings of the MoE actor on
NablaMimic had been seen, taking the stricter reading; its sentences on a first and a second training refer to the
second trainings, which were added after the first trainings of MoE recovery; its last sentence applies the same rules
to MoE recovery with full-width experts, a condition added after results had been read.

> **The MoE actor ([the question on the MoE actor]).** On one trainer, MoE recovery **lowers the tracking cost** if the
> tracking cost of single-network recovery minus that of MoE recovery is at least 0.003 m at both seeds. It **does not
> lower it** if that difference is below 0.001 m at both seeds. Otherwise the reading is **undecided** and a third seed
> (seed 2) is trained; the rule is then applied with "at all three seeds" in place of "at both seeds", and a reading
> that is still neither is reported as undecided. MoE recovery **keeps getting up** if its R is at least the R of
> single-network recovery minus 5 at both seeds, and **does not keep getting up** otherwise. The answer to [the question
> on the MoE actor] on a trainer is yes if MoE recovery lowers the tracking cost and keeps getting up, and no if it does
> not lower the tracking cost; if it lowers the tracking cost and does not keep getting up, both readings are reported
> and the answer is no. MoE recovery has a first and a second training at each seed [...]: the rules are read once on
> the first trainings of the two seeds and once on the second trainings. Where the two readings differ, the reading is
> undecided; where one of the two lacks a seed because a training failed [...], the rules are read on the other alone.
> The same rules apply to MoE recovery with full-width experts.

**Training length.** The text is the declared rule with the outcome "undecided" between its thresholds made definite
after the first readings of the MoE actor on NablaMimic had been seen, taking the stricter reading.

> **Training length ([the question on the training length]).** The tracking cost **is a matter of training length** if
> the cost of single-network recovery at twice the training length is at most 0.5 times its cost at the declared length
> at both seeds, and **is not** if it is above 0.8 times at both seeds. Otherwise the reading is **undecided**. Each
> cost is taken against the tracking-only condition of the same length.

**Five-seed reading,** written after the rules above had read "undecided", before the trainings of seeds 2 to 4 and
of the wider decoder that it reads (the trainings at seeds 0 and 1 of the other conditions had been read):

> A reading over five seeds is given beside the declared rule's and is exploratory: of two conditions, the first
> lowers the tracking cost relative to the second if the mean over seeds 0 to 4 of the second's cost minus the first's
> is at least 0.003 m and that difference is positive at four or more of the five seeds; it does not if the mean is
> below 0.001 m; otherwise the reading is undecided. It is read for MoE recovery with full-width experts against
> single-network recovery, for MoE recovery with full-width experts against single-network recovery with a wider
> decoder, and for single-network recovery with a wider decoder against single-network recovery.

**Sensitivity reading and baselines,** written after the seed 2 trainings of tracking-only and single-network recovery
had been read:

> The baselines of [...] are fixed, and no threshold is changed. The trainings of tracking-only and of single-network
> recovery that enter the declared rule and the five-seed reading are those trained beside the MoE conditions. A
> sensitivity reading is given beside each: the same comparison with every condition's value at a seed replaced by the
> mean over all of that condition's completed trainings at that seed with identical arguments.

**Per-condition report,** written at the same time:

> For every BlinkMimic 2 condition the number of completed trainings and the mean and standard deviation of their
> tracking errors are reported, without a rule.

**Differences below 0.0032 m,** written at the same time:

> On BlinkMimic 2, a difference between two conditions that rests on one training per condition and seed and is
> smaller than 0.0032 m is not evidence of a difference. The outcomes recorded so far on BlinkMimic 2 all read
> "undecided" and are unchanged.

**Supplementary reading,** written after a reading of the checkpoints of five trainings had been made; the declared
readings stay on the last checkpoint and the second reading of each rule is exploratory:

> The supplementary reading is the mean of three values: the tracking error at a training's last checkpoint (the mean
> of its three sets of rollouts) and at the checkpoints 500 and 1,000 iterations before it (one set each). It is given
> for every BlinkMimic 2 training that enters a comparison. The declared readings and rules stay on the last
> checkpoint; each rule is read a second time on the supplementary reading, and that second reading is exploratory.

**Changes to the declaration.**

- **Wording and phrases made definite.** The declaration's questions, conditions and readings were renamed from
  earlier working names to the papers' terms and to descriptive names, with no condition, reading, threshold or rule
  changed; the declaration does not say whether results had been read when this was done. Several phrases were made
  definite where they touch these rules: the value of a BlinkMimic 2 checkpoint in a rule is the mean over its three
  sets of rollouts; the training-length rule has the outcome "undecided" between its thresholds; the MoE rule says how
  a third seed is read and what answers the question; the rule that keeps the tracking target when the pelvis is at
  0.8 or more of the command row's pelvis height says which heights are compared and when, and the mixing network's
  input and the decoder's widths are written out. The training-length and MoE-rule phrases were settled after the
  first readings of the MoE actor on NablaMimic and the first readings of the world-model study's world-model error had
  been seen, each taking the stricter reading; the declaration does not say when the other phrases were settled
  relative to the first readings on BlinkMimic 2.
- **Trainings that end on non-finite values.** A training that ends on non-finite values is reported as failed and is
  trained once more at the same seed and revision with nothing changed; a second such end at the same seed is reported
  and not followed by a further attempt; the rules are read on the trainings that completed. This was written after a
  NablaMimic training ended on non-finite losses, and the declaration does not say when it was written relative to the
  first readings on BlinkMimic 2. No training of this study failed. Two 300-iteration trial runs, one made before
  the trainings of 8ad67d9d324c and one before those of 55e1cddc68e2, at revisions that are not in the revisions
  table in Results, stopped on code defects that these two revisions fix and that this branch's code does not have;
  they are not trainings of this study.
- **Card models.** Trainings run on RTX 4090 cards. Rollouts of BlinkMimic 2 checkpoints run on RTX 3090 cards,
  because RTX 4090 cards were not available for them. All rollouts that enter one comparison come from one card model,
  and a checkpoint that was also rolled out on an RTX 4090 is reported beside its RTX 3090 reading. The declaration
  does not say whether results had been read when this was added.
- **MoE recovery with full-width experts added, after results had been read.** The results read were NablaMimic's MoE
  recovery. Its results are marked, wherever they are written, as those of a condition added after results had been
  read.
- **MoE recovery trained twice.** After the first trainings of MoE recovery, the second trainings at each seed, from
  revision 55e1cddc68e2, were added, and the rules are read on the first trainings and on the second trainings. Where
  the two readings differ, the reading is undecided. The declaration does not say whether results had been read when
  this was added.
- **More seeds and the wider decoder, declared before they were trained and after results.** On BlinkMimic 2 the MoE
  rule read "undecided" for MoE recovery (first and second trainings) and for MoE recovery with full-width experts,
  and the training-length rule read "undecided" (0.99 and 0.78). Two trainings of one condition at one seed differed
  by 0.0015 m and 0.0023 m, and the three sets of one checkpoint by at most 0.0004 m, so two seeds cannot decide
  differences of the size the rule asks about. Seed 2 was added for four conditions, seeds 3 and 4 for three, the
  wider decoder at seeds 0 to 4, and the five-seed reading is exploratory.
- **Trainings repeated at one seed and the baselines.** Written after the seed 2 trainings of tracking-only and
  single-network recovery had been read. Single-network recovery trained three times at seeds 0 and 1 with identical
  arguments gave tracking errors of 0.0343, 0.0342 and 0.0342 m at seed 0 and 0.0358, 0.0358 and 0.0327 m at seed 1;
  the declaration gained the sentence on differences below 0.0032 m, the baselines were fixed to the trainings beside
  the MoE conditions with no threshold changed, and the sensitivity reading beside each rule and the per-condition
  report were added.
- **Checkpoint reading and supplementary reading.** A reading of the checkpoints at 5,500, 6,000, 6,500 and 7,000
  iterations of five trainings was declared before it was made (the three trainings of single-network recovery at
  seed 1 in Table 5, and two trainings of single-network recovery with a world model with hidden widths 2048, 2048,
  2048 (the world-model study's) at seed 0; the latter are not in this study's tables), one set each, with the rule
  for adding a supplementary reading. It found that the tracking error of one training spans 0.0009, 0.0013, 0.0014,
  0.0015 and 0.0010 m from iteration 5,500 to 7,000, that it moves by up to 0.0013 m between checkpoints 500
  iterations apart, and that the condition for a supplementary reading was met in four of the five. The supplementary
  reading and the second reading of each rule on it were added after results.

## Commands, in order

Run from the repository root, in this repository's environment. The trainings and rollouts ran at the revisions that
Table 1 and the revisions table in Results give for each training, and the readings at those that the revisions table
gives; the last column of that table says how each differs from this branch's code.
`RUNS`, `STUDY` and `OUT` are output roots that do not exist yet (trainings, rollouts, readings), and `DIR` holds the 40
LAFAN1 G1 retarget CSVs. `NAME` is a training's name and `S` its seed. `SCENE` is the pair study's scene, reused because
the library inputs in each training's `args.json` equal its `scene.json` "library". `NABLAMIMIC` is NablaMimic's
`studies/recovery` at revision 369a2833050b, of which only `readings_core.py` is read. `READER` is the root of a
checkout of this branch, which holds `studies/recovery_pair/groups.py` from the pull request this one depends on; the
readings ran at 33125c24b8e6, de06c97b28ec and e692ba9baac8, whose reading code is that of this branch.
`TRAINING_REVISION` and `ROLLOUT_CODE_REVISION` are full revision hashes.

1. **Trainings**, each on one 48 GB RTX 4090 (`--ckpt_every 500` keeps every 500-iteration checkpoint); `BONES_LIB` is
   the pair study's `--bones_lib` string. After the first line, `...` stands for its common options, `--log_dir
   $RUNS/NAME --seed $S --bones_list= --motion_dir $DIR --bones_lib $BONES_LIB`:

   ```bash
   source scripts/presets.sh
   python -m blinkmimic2.train --log_dir $RUNS/NAME --seed $S --bones_list= --motion_dir $DIR \
       --bones_lib $BONES_LIB ${PRESET[recovery_control]} --ckpt_every 500   # tracking-only
   python -m blinkmimic2.train ... ${PRESET[recovery]} --ckpt_every 500   # single-network recovery
   python -m blinkmimic2.train ... ${PRESET[recovery]} --ckpt_every 500 --moe   # MoE recovery
   python -m blinkmimic2.train ... ${PRESET[recovery]} --ckpt_every 500 --moe \
       --moe_expert_hidden 2048,1024,512,256   # MoE recovery with full-width experts
   python -m blinkmimic2.train ... ${PRESET[recovery]} --ckpt_every 500 \
       --dec_hidden 3200,1600,800,400   # single-network recovery with a wider decoder
   python -m blinkmimic2.train ... ${PRESET[recovery_control]} --ckpt_every 500 \
       --iterations 14000   # tracking-only, twice the training length
   python -m blinkmimic2.train ... ${PRESET[recovery]} --ckpt_every 500 \
       --iterations 14000   # single-network recovery, twice the training length
   ```

2. **Rollouts**, one 24 GB RTX 3090 per batch of rollouts (six processes at once), with the code at the revision that
   trained the checkpoint (for the pair study's trainings, 33125c24b8e6 at 7,000 iterations and f29424a6666c at 6,500
   and 6,000; for the world-model study's, e692ba9baac8): three sets at the last checkpoint and one set at each of the
   two checkpoints 500 and 1,000 iterations before it. `--ckpt` is the last checkpoint, 7000, or 14000 at twice the
   training length. The two earlier checkpoints, 6500 and 6000 (13500 and 13000), are rolled out the same way with `for
   k in 1` and an output folder of their own, read with `readings.py` once, and passed to `groups.py` as a single
   `--set`. The two exported variables are only recorded in the rollouts' descriptions: the revision that trained the
   checkpoint, and the revision of the code that runs `tools/rollouts.py`.

   ```bash
   export RUN_SOURCE_REVISION=$TRAINING_REVISION SOURCE_REVISION=$ROLLOUT_CODE_REVISION
   FIRST_ROW_CONTROLS=$(python -c "import json; print(json.load(open('$SCENE/scene.json'))['first_row_controls'])")
   for k in 1 2 3; do
     python tools/rollouts.py --run_dir $RUNS/NAME --ckpt 7000 --scene $SCENE/placed.json \
         --termination nonfinite --controls 1000 --out $STUDY/NAME/rep$k/placed
     python tools/rollouts.py --run_dir $RUNS/NAME --ckpt 7000 --scene $SCENE/twins.json \
         --termination nonfinite --controls 1000 --out $STUDY/NAME/rep$k/twins
     python tools/rollouts.py --run_dir $RUNS/NAME --ckpt 7000 --scene $SCENE/first_rows.json \
         --termination recipe --controls $FIRST_ROW_CONTROLS --out $STUDY/NAME/rep$k/first-rows
   done
   ```

   Three sets of the first training of MoE recovery at seed 0 were also rolled out on a 24 GB RTX 4090.

3. **Readings**, on the CPU: `readings.py` once per set (`K` is 1, 2 and 3), then the groups over the three sets:

   ```bash
   PYTHONPATH=$READER/src python $READER/studies/recovery_pair/readings.py \
       --pair $READER/studies/recovery_pair/pair.json --readings-core $NABLAMIMIC \
       --scene $SCENE --placed $STUDY/NAME/repK/placed --twins $STUDY/NAME/repK/twins \
       --first_rows $STUDY/NAME/repK/first-rows --out $OUT/NAME/repK
   PYTHONPATH=$READER/src python $READER/studies/recovery_pair/groups.py --scene $SCENE --motion_dir $DIR \
       --set $OUT/NAME/rep1 $STUDY/NAME/rep1/first-rows --set $OUT/NAME/rep2 $STUDY/NAME/rep2/first-rows \
       --set $OUT/NAME/rep3 $STUDY/NAME/rep3/first-rows --out $OUT/NAME/groups.json
   ```

4. **Rules**, on the CPU, from the `groups.json` files, by a program that is not part of this repository. Tables 1 and
   10 hold the per-seed values it reads, so that each outcome of Tables 2, 3, 6 and 11 to 13 can be recomputed from
   them.

## What the study does not show

- **Few trainings.** Every condition has one training per seed at seeds 2 to 4; at seeds 0 and 1, two or three
  trainings exist only for single-network recovery and MoE recovery, and no condition has a third training at a seed
  except as listed.
- **Small differences.** Trainings with identical arguments at one seed differ by up to 0.00311 m (Results), so
  differences of that size are not read as differences between conditions.
- **Critic inputs.** StableMimic's critic inputs, the command and the active get-up reference, are not implemented;
  the critic reads the hidden row in place of the command row while recovering ([RECOVERY.md](../../RECOVERY.md)).
  StableMimic's proximal policy optimization (PPO) training, its start-frame weighting and its success and power
  terms are not implemented either.
- **Post-fall measures.** StableMimic's post-fall measures, the 95th percentiles of limb speed, joint-target rate and
  joint torque over the three seconds after a fall, were not read, and no rollout includes pushes.
- **Earlier checkpoints.** One set of rollouts exists at each of the two checkpoints before the last; the last
  checkpoints of the pair study's and the world-model study's trainings of single-network recovery enter with the
  rollouts that the revisions table lists for them (at 33125c24b8e6 for the pair study's, at e692ba9baac8 for the
  world-model study's) and the readings of those rollouts, and were not rolled out again.
- **Simulation only.** The readings are kinematic readings of simulated trials in MuJoCo Warp; no hardware was used.

## Results

**Revisions and cards.** The trainings, rollouts and readings ran while the code was being written, at the revisions
below. These revisions are not in this branch's history, which holds the final code in a few commits; 369a2833050b is
a revision of NablaMimic. The last column says, from a diff restricted to the code that each revision's trainings,
rollouts and readings execute, how each revision differs from this branch's code (with
`studies/recovery_pair/groups.py` from the pull request this one depends on).

| Revision that ran | What it trained, computed or read | How its code differs from this branch's, in the code those commands execute |
| --- | --- | --- |
| 8ad67d9d324c | Seeds 0 and 1 of tracking-only, single-network recovery and MoE recovery (first trainings); tracking-only and single-network recovery at twice the training length, seeds 0 and 1; the rollouts of their checkpoints, the three sets on an RTX 4090 included | The mixing network's attribute in the policy, functions, keyword arguments, stored checkpoint keys, per-episode flag and buffer field, and log columns had other names, and one option-check message was worded differently; the log columns held the same values in the same order. `--moe_expert_hidden` did not exist; without it this branch's code writes no such key to `args.json` and builds the experts with every decoder width halved, as this revision did; no cited value depends on these differences, but this branch's code cannot load the MoE checkpoints of the first trainings, whose keys have the other names. |
| 55e1cddc68e2 | MoE recovery with full-width experts at seeds 0 and 1; MoE recovery (second trainings) at seeds 0 and 1; the rollouts of their checkpoints | none |
| 806f97fbea35 | Seeds 2 to 4 of tracking-only and single-network recovery, seed 2 of MoE recovery, seeds 2 to 4 of MoE recovery with full-width experts; single-network recovery with a wider decoder at seeds 0 to 4; the rollouts of their checkpoints | none |
| f29424a6666c | The pair study's trainings of single-network recovery (seeds 0 and 1), as the pair-study page says; the rollouts of their checkpoints at 6,000 and 6,500 iterations, and at 5,500 for seed 1 in the checkpoint reading | Its `src`, `scripts` and `tools` are those of f7e140c, as the pair-study page says. Against them this branch adds the MoE actor behind `--moe` and `--moe_expert_hidden` and restructures statements on the path of a single-network training and of a rollout (the policy's construction and optimizer groups, the per-episode flag of the simulator and of the replay buffer, the actor's loss), which without those options give the same parameters, random draws and values; no change of behaviour. |
| e692ba9baac8 | The world-model study's trainings of single-network recovery (seeds 0 and 1); the rollouts of their checkpoints; the two readings of their last checkpoints | This revision holds the world-model study's code, which this branch lacks: DynaMimic's link inputs to the world model, a failure penalty at the target row and a reading of the world-model error, behind options these trainings did not pass, and the world model built through a function with the same arguments; this branch adds the MoE actor behind `--moe`. Without those options both train and roll out the same, with the same random draws, and `readings.py` and `groups.py` have this branch's executable code; no change of behaviour. |
| 33125c24b8e6 | The rollouts of the pair study's checkpoints at 7,000 iterations (three sets each, on RTX 3090 cards); `readings.py` and `groups.py` for every reading on this page except those made at e692ba9baac8 and de06c97b28ec | `tools/rollouts.py` is this branch's; the modules under `src/blinkmimic2` lack the MoE actor and hold world-model functions that the rollouts do not call, so a single-network checkpoint is rolled out the same. `readings.py` and the modules that `groups.py` imports are this branch's, and `groups.py` has this branch's executable code; no change of behaviour. |
| de06c97b28ec | `readings.py` and `groups.py` for the readings of the checkpoints at 6,000 and 6,500 iterations of the pair study's and the world-model study's trainings of single-network recovery and of this study's single-network recovery at seed 1, and of the checkpoints at 5,500 iterations of the checkpoint reading | The reading code is that of 33125c24b8e6: `readings.py`, `groups.py` and the modules it imports have this branch's executable code, except that the robot's constants define one more tuple of link names, which the readings do not read; no change of behaviour. |
| 369a2833050b | NablaMimic's `studies/recovery/readings_core.py`, which `readings.py` calls at every reading | none: `readings.py` loads the same file from NablaMimic at this revision, and the file is the same on NablaMimic's published `recovery-pair-study` branch |

The scene files are those of the pair study. The revision that built the scene files is not known. Trainings ran on
one NVIDIA GeForce RTX 4090 with 48 GB at a 450 W power limit each, except the pair study's trainings (350 W);
rollouts of the last checkpoints ran on one RTX 3090 with 24 GB per batch of rollouts and are the readings of every
rule below, and three sets of one checkpoint also ran on an RTX 4090 with 24 GB. Every training exited with code 0 and
its last log line has 0 non-finite updates; no training, rollout or reading failed or was skipped. Two 300-iteration
trial runs, one made before the trainings of 8ad67d9d324c and one before those of 55e1cddc68e2, at revisions that are
not in the table above, stopped on code defects that these two revisions fix and that this branch's code does not
have; they are not trainings of this study.
The revisions 8ad67d9d324c, 55e1cddc68e2 and 806f97fbea35 trained this page's conditions with the same computation at
equal options, as the table's last column says; tracking-only and single-network checkpoints carry no MoE keys.

Table 1 gives every training's last checkpoint (7,000 iterations, or 14,000 at twice the training length). The tracking
error over the 34 commanded clips is in metres, the mean of three sets with the range of the sets in brackets;
"Revision" is the revision that trained the checkpoint (revisions table); "Commanded completed", R, J, F and "Excluded
completed" give the three sets; "Excluded E" is the tracking error over the six excluded clips in metres, and "Ends"
counts ends in get-up windows entered, for the five hidden clips, fallAndGetUp1_subject4 and the commanded clips with a
window. For the trainings of single-network recovery made outside this study, "Training" names the study that made them.

| Condition | Seed | Training | Revision | Tracking error (m), mean (range) | Commanded completed of 34 | R | J | F | Excluded completed of 6 | Excluded E (m) | Ends |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tracking-only | 0 | this study | 8ad67d9d324c | 0.02744 (0.02743–0.02746) | 34, 34, 34 | 95, 91, 92 | 94, 89, 90 | 0, 0, 0 | 6, 6, 6 | 0.0403 | 0 in 273 / 0 in 51 / 0 in 129 |
| tracking-only | 1 | this study | 8ad67d9d324c | 0.02732 (0.02709–0.02745) | 34, 33, 33 | 64, 69, 65 | 60, 64, 65 | 0, 0, 0 | 6, 6, 6 | 0.0392 | 0 in 273 / 0 in 51 / 0 in 127 |
| tracking-only | 2 | this study | 806f97fbea35 | 0.02716 (0.02713–0.02719) | 34, 34, 34 | 79, 76, 81 | 75, 73, 78 | 0, 0, 0 | 6, 6, 5 | 0.0390 | 1 in 270 / 0 in 51 / 0 in 129 |
| tracking-only | 3 | this study | 806f97fbea35 | 0.02763 (0.02762–0.02765) | 34, 34, 34 | 92, 94, 95 | 91, 94, 94 | 0, 0, 1 | 6, 5, 6 | 0.0460 | 1 in 258 / 0 in 51 / 0 in 129 |
| tracking-only | 4 | this study | 806f97fbea35 | 0.02713 (0.02710–0.02718) | 34, 34, 34 | 82, 85, 88 | 81, 83, 82 | 1, 0, 0 | 6, 6, 6 | 0.0392 | 0 in 273 / 0 in 51 / 0 in 129 |
| single-network recovery | 0 | this study | 8ad67d9d324c | 0.03418 (0.03408–0.03424) | 34, 34, 34 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 5, 5, 5 | 0.0543 | 3 in 213 / 0 in 51 / 0 in 129 |
| single-network recovery | 0 | pair study's | f29424a6666c | 0.03430 (0.03429–0.03431) | 34, 34, 34 | 99, 99, 99 | 97, 97, 97 | 1, 1, 1 | 4, 4, 4 | 0.0568 | 4 in 241 / 2 in 49 / 0 in 129 |
| single-network recovery | 0 | world-model study's | e692ba9baac8 | 0.03424 (0.03422–0.03426) | 34, 34, 34 | 99, 99, 99 | 99, 99, 99 | 0, 0, 0 | 5, 5, 5 | 0.0536 | 3 in 252 / 0 in 51 / 0 in 129 |
| single-network recovery | 1 | this study | 8ad67d9d324c | 0.03576 (0.03569–0.03589) | 32, 34, 34 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 6, 6, 5 | 0.0577 | 1 in 253 / 0 in 51 / 0 in 129 |
| single-network recovery | 1 | pair study's | f29424a6666c | 0.03576 (0.03570–0.03586) | 34, 34, 34 | 97, 97, 97 | 97, 97, 97 | 0, 0, 0 | 5, 5, 6 | 0.0551 | 2 in 251 / 0 in 51 / 0 in 129 |
| single-network recovery | 1 | world-model study's | e692ba9baac8 | 0.03265 (0.03264–0.03267) | 33, 33, 33 | 99, 99, 99 | 99, 99, 99 | 0, 0, 0 | 5, 6, 6 | 0.0541 | 1 in 271 / 0 in 51 / 0 in 126 |
| single-network recovery | 2 | this study | 806f97fbea35 | 0.03288 (0.03282–0.03299) | 34, 34, 34 | 99, 99, 99 | 99, 99, 99 | 0, 0, 0 | 5, 6, 5 | 0.0538 | 0 in 273 / 2 in 25 / 0 in 129 |
| single-network recovery | 3 | this study | 806f97fbea35 | 0.03529 (0.03489–0.03556) | 33, 32, 34 | 98, 98, 98 | 98, 98, 98 | 0, 1, 0 | 6, 6, 6 | 0.0561 | 0 in 273 / 0 in 51 / 0 in 125 |
| single-network recovery | 4 | this study | 806f97fbea35 | 0.03216 (0.03212–0.03221) | 34, 33, 34 | 99, 99, 99 | 99, 99, 99 | 0, 0, 0 | 5, 5, 5 | 0.0542 | 0 in 219 / 0 in 51 / 0 in 129 |
| MoE recovery | 0 | first | 8ad67d9d324c | 0.03434 (0.03430–0.03436) | 34, 34, 34 | 97, 97, 97 | 97, 97, 97 | 0, 0, 0 | 6, 6, 6 | 0.0565 | 0 in 273 / 0 in 51 / 0 in 129 |
| MoE recovery | 0 | second | 55e1cddc68e2 | 0.03586 (0.03578–0.03592) | 33, 34, 33 | 98, 98, 98 | 98, 98, 98 | 0, 0, 1 | 5, 5, 5 | 0.0562 | 0 in 237 / 0 in 51 / 1 in 129 |
| MoE recovery | 1 | first | 8ad67d9d324c | 0.03467 (0.03465–0.03471) | 34, 34, 34 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 5, 6, 6 | 0.0570 | 1 in 258 / 0 in 51 / 0 in 129 |
| MoE recovery | 1 | second | 55e1cddc68e2 | 0.03234 (0.03225–0.03248) | 33, 33, 33 | 97, 97, 97 | 96, 96, 96 | 0, 0, 0 | 6, 5, 6 | 0.0558 | 1 in 265 / 0 in 51 / 0 in 120 |
| MoE recovery | 2 | one | 806f97fbea35 | 0.03693 (0.03683–0.03711) | 33, 32, 32 | 98, 97, 97 | 97, 97, 97 | 0, 0, 0 | 5, 5, 5 | 0.0615 | 3 in 225 / 0 in 51 / 3 in 115 |
| MoE recovery with full-width experts | 0 | one | 55e1cddc68e2 | 0.03302 (0.03296–0.03314) | 34, 34, 34 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 6, 6, 6 | 0.0514 | 0 in 273 / 0 in 51 / 0 in 129 |
| MoE recovery with full-width experts | 1 | one | 55e1cddc68e2 | 0.03128 (0.03128–0.03129) | 34, 34, 34 | 98, 98, 98 | 98, 98, 98 | 0, 1, 1 | 6, 6, 6 | 0.0509 | 0 in 273 / 0 in 51 / 0 in 129 |
| MoE recovery with full-width experts | 2 | one | 806f97fbea35 | 0.03248 (0.03238–0.03254) | 34, 33, 33 | 98, 98, 98 | 97, 97, 98 | 0, 0, 0 | 5, 5, 5 | 0.0509 | 3 in 228 / 0 in 51 / 0 in 127 |
| MoE recovery with full-width experts | 3 | one | 806f97fbea35 | 0.03160 (0.03157–0.03163) | 33, 33, 33 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 6, 6, 6 | 0.0490 | 0 in 273 / 0 in 51 / 0 in 129 |
| MoE recovery with full-width experts | 4 | one | 806f97fbea35 | 0.03319 (0.03316–0.03322) | 34, 34, 34 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 6, 6, 6 | 0.0509 | 0 in 273 / 0 in 51 / 0 in 129 |
| single-network recovery with a wider decoder | 0 | one | 806f97fbea35 | 0.03169 (0.03168–0.03171) | 33, 33, 34 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 6, 6, 6 | 0.0513 | 0 in 273 / 0 in 51 / 1 in 129 |
| single-network recovery with a wider decoder | 1 | one | 806f97fbea35 | 0.03168 (0.03159–0.03184) | 34, 34, 33 | 97, 97, 97 | 96, 96, 96 | 0, 0, 0 | 6, 6, 6 | 0.0503 | 0 in 273 / 0 in 51 / 1 in 129 |
| single-network recovery with a wider decoder | 2 | one | 806f97fbea35 | 0.03162 (0.03160–0.03164) | 34, 34, 34 | 95, 95, 95 | 95, 95, 95 | 0, 0, 0 | 6, 6, 6 | 0.0500 | 0 in 273 / 0 in 51 / 0 in 129 |
| single-network recovery with a wider decoder | 3 | one | 806f97fbea35 | 0.03333 (0.03331–0.03335) | 33, 34, 34 | 97, 97, 97 | 97, 97, 97 | 0, 0, 0 | 6, 6, 6 | 0.0523 | 0 in 273 / 0 in 51 / 0 in 129 |
| single-network recovery with a wider decoder | 4 | one | 806f97fbea35 | 0.03331 (0.03331–0.03332) | 33, 34, 34 | 96, 96, 96 | 95, 95, 95 | 0, 0, 0 | 6, 6, 6 | 0.0511 | 0 in 273 / 0 in 51 / 0 in 128 |
| tracking-only at twice the training length | 0 | one | 8ad67d9d324c | 0.02524 (0.02520–0.02529) | 34, 34, 33 | 63, 73, 62 | 62, 70, 59 | 0, 0, 0 | 6, 6, 6 | 0.0344 | 0 in 273 / 0 in 51 / 0 in 129 |
| tracking-only at twice the training length | 1 | one | 8ad67d9d324c | 0.02550 (0.02546–0.02554) | 32, 33, 33 | 77, 83, 88 | 74, 78, 85 | 0, 0, 1 | 6, 6, 6 | 0.0355 | 0 in 273 / 0 in 51 / 0 in 129 |
| single-network recovery at twice the training length | 0 | one | 8ad67d9d324c | 0.03190 (0.03189–0.03192) | 33, 33, 33 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 5, 5, 5 | 0.0479 | 3 in 243 / 0 in 51 / 0 in 126 |
| single-network recovery at twice the training length | 1 | one | 8ad67d9d324c | 0.03210 (0.03204–0.03213) | 34, 34, 33 | 98, 98, 98 | 98, 98, 98 | 0, 0, 0 | 6, 6, 6 | 0.0490 | 0 in 273 / 0 in 51 / 0 in 129 |

Table 2 reads the declared MoE rule. Each seed cell gives cost(single-network recovery) / cost(MoE condition) in
metres and, in brackets, their difference; "R" gives, per seed, R of the MoE condition / R of single-network recovery
(mean of three sets). The sensitivity reading was added after results and uses the pair study's and the world-model
study's trainings of single-network recovery. Outcomes are the rule's, at two seeds (0 and 1) and at three seeds (0
to 2); "Keeps getting up" is given at two / three seeds.

| Comparison | Reading | Seed 0 | Seed 1 | Seed 2 | R at seeds 0; 1; 2 | Outcome, two seeds | Outcome, three seeds | Keeps getting up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MoE recovery, first trainings | primary | 0.00674 / 0.00689 (-0.00016) | 0.00844 / 0.00735 (+0.00109) | 0.00572 / 0.00977 (-0.00405) | 97.0 / 98.0; 98.0 / 98.0; 97.3 / 99.0 | undecided | undecided | yes / yes |
| MoE recovery, first trainings | sensitivity | 0.00680 / 0.00765 (-0.00086) | 0.00741 / 0.00619 (+0.00122) | 0.00572 / 0.00977 (-0.00405) | 97.0 / 98.0; 98.0 / 98.0; 97.3 / 99.0 | undecided | undecided | yes / yes |
| MoE recovery, second trainings | primary | 0.00674 / 0.00842 (-0.00168) | 0.00844 / 0.00502 (+0.00342) | 0.00572 / 0.00977 (-0.00405) | 98.0 / 98.0; 97.0 / 98.0; 97.3 / 99.0 | undecided | undecided | yes / yes |
| MoE recovery, second trainings | sensitivity | 0.00680 / 0.00765 (-0.00086) | 0.00741 / 0.00619 (+0.00122) | 0.00572 / 0.00977 (-0.00405) | 98.0 / 98.0; 97.0 / 98.0; 97.3 / 99.0 | undecided | undecided | yes / yes |
| MoE recovery with full-width experts | primary | 0.00674 / 0.00558 (+0.00116) | 0.00844 / 0.00396 (+0.00448) | 0.00572 / 0.00532 (+0.00040) | 98.0 / 98.0; 98.0 / 98.0; 98.0 / 99.0 | undecided | undecided | yes / yes |
| MoE recovery with full-width experts | sensitivity | 0.00680 / 0.00558 (+0.00122) | 0.00741 / 0.00396 (+0.00344) | 0.00572 / 0.00532 (+0.00040) | 98.0 / 98.0; 98.0 / 98.0; 98.0 / 99.0 | undecided | undecided | yes / yes |

Table 3 reads the five-seed reading (exploratory) on the last checkpoint; its sensitivity reading uses the pair study's
and the world-model study's trainings of single-network recovery. Each seed cell gives cost(first) /
cost(second) in metres with their difference in brackets; the mean difference is in metres.

| First | Second | Reading | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean difference | Seeds with difference > 0 | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MoE recovery with full-width experts | single-network recovery | primary | 0.00558 / 0.00674 (+0.00116) | 0.00396 / 0.00844 (+0.00448) | 0.00532 / 0.00572 (+0.00040) | 0.00397 / 0.00766 (+0.00369) | 0.00606 / 0.00503 (-0.00103) | +0.00174 | 4 | undecided |
| MoE recovery with full-width experts | single-network recovery | sensitivity | 0.00558 / 0.00680 (+0.00122) | 0.00396 / 0.00741 (+0.00344) | 0.00532 / 0.00572 (+0.00040) | 0.00397 / 0.00766 (+0.00369) | 0.00606 / 0.00503 (-0.00103) | +0.00154 | 4 | undecided |
| MoE recovery with full-width experts | single-network recovery with a wider decoder | primary | 0.00558 / 0.00425 (-0.00133) | 0.00396 / 0.00436 (+0.00040) | 0.00532 / 0.00446 (-0.00086) | 0.00397 / 0.00570 (+0.00173) | 0.00606 / 0.00618 (+0.00012) | +0.00001 | 3 | does not lower it |
| MoE recovery with full-width experts | single-network recovery with a wider decoder | sensitivity | 0.00558 / 0.00425 (-0.00133) | 0.00396 / 0.00436 (+0.00040) | 0.00532 / 0.00446 (-0.00086) | 0.00397 / 0.00570 (+0.00173) | 0.00606 / 0.00618 (+0.00012) | +0.00001 | 3 | does not lower it |
| single-network recovery with a wider decoder | single-network recovery | primary | 0.00425 / 0.00674 (+0.00249) | 0.00436 / 0.00844 (+0.00408) | 0.00446 / 0.00572 (+0.00126) | 0.00570 / 0.00766 (+0.00196) | 0.00618 / 0.00503 (-0.00115) | +0.00173 | 4 | undecided |
| single-network recovery with a wider decoder | single-network recovery | sensitivity | 0.00425 / 0.00680 (+0.00255) | 0.00436 / 0.00741 (+0.00305) | 0.00446 / 0.00572 (+0.00126) | 0.00570 / 0.00766 (+0.00196) | 0.00618 / 0.00503 (-0.00115) | +0.00153 | 4 | undecided |

Table 4 reports each condition without a rule: the number of completed trainings, and the mean and standard deviation
(divisor n - 1, n the number of trainings) of their tracking errors over the 34 commanded clips in metres, including
the pair study's and the world-model study's trainings of single-network recovery. The last two columns read this
study's primary trainings over their seeds: the tracking error and the tracking cost, with [minimum, maximum]; for MoE
recovery they are the first trainings at seeds 0 and 1 and the one training at seed 2.

| Condition | Trainings | Mean (m) | Standard deviation (m) | Primary trainings: seeds | Tracking error, mean [min, max] (m) | Tracking cost, mean [min, max] (m) |
| --- | --- | --- | --- | --- | --- | --- |
| tracking-only | 5 | 0.02734 | 0.00021 | 0, 1, 2, 3, 4 | 0.02734 [0.02713, 0.02763] | – |
| single-network recovery | 9 | 0.03414 | 0.00134 | 0, 1, 2, 3, 4 | 0.03405 [0.03216, 0.03576] | 0.00672 [0.00503, 0.00844] |
| MoE recovery | 5 | 0.03483 | 0.00173 | 0, 1, 2 | 0.03531 [0.03434, 0.03693] | 0.00801 [0.00689, 0.00977] |
| MoE recovery with full-width experts | 5 | 0.03232 | 0.00085 | 0, 1, 2, 3, 4 | 0.03232 [0.03128, 0.03319] | 0.00498 [0.00396, 0.00606] |
| single-network recovery with a wider decoder | 5 | 0.03233 | 0.00091 | 0, 1, 2, 3, 4 | 0.03233 [0.03162, 0.03333] | 0.00499 [0.00425, 0.00618] |
| tracking-only at twice the training length | 2 | 0.02537 | 0.00018 | 0, 1 | 0.02537 [0.02524, 0.02550] | – |
| single-network recovery at twice the training length | 2 | 0.03200 | 0.00014 | 0, 1 | 0.03200 [0.03190, 0.03210] | 0.00663 [0.00660, 0.00666] |

Table 5 reads how far trainings repeated at one seed differ: the tracking error over the 34 commanded clips at the
last checkpoint, in metres, per training, and the largest minus the smallest.

| Condition | Seed | Trainings: tracking error (m) | Largest minus smallest (m) |
| --- | --- | --- | --- |
| single-network recovery | 0 | this study 0.03418; pair study's 0.03430; world-model study's 0.03424 | 0.00012 |
| single-network recovery | 1 | this study 0.03576; pair study's 0.03576; world-model study's 0.03265 | 0.00311 |
| MoE recovery | 0 | first 0.03434; second 0.03586 | 0.00153 |
| MoE recovery | 1 | first 0.03467; second 0.03234 | 0.00233 |

Table 6 reads the training-length rule: the tracking cost of single-network recovery at the declared length (7,000
iterations) and at twice the training length (14,000 iterations), in metres, each against tracking-only of the same
length, and their ratio. The sensitivity reading was added after results and uses the same trainings.

| Reading | Seed | Cost at 7,000 | Cost at 14,000 | Ratio | Outcome |
| --- | --- | --- | --- | --- | --- |
| primary | 0 | 0.00674 | 0.00666 | 0.989 | |
| primary | 1 | 0.00844 | 0.00660 | 0.782 | |
| primary | both seeds | | | | undecided |
| sensitivity | 0 | 0.00680 | 0.00666 | 0.981 | |
| sensitivity | 1 | 0.00741 | 0.00660 | 0.891 | |
| sensitivity | both seeds | | | | is not |

Table 7 gives, per actor form, the actor's parameters, the steady seconds per iteration and the peak card memory in
MiB, per training in the order of the seeds listed; MoE recovery lists seed 0 first, seed 0 second, seed 1 first, seed
1 second and seed 2.

| Condition (actor) | Parameters | Steady seconds per iteration | Peak card memory (MiB) |
| --- | --- | --- | --- |
| tracking-only (single network) | 9,116,477 | seeds 0 to 4: 0.1173, 0.1177, 0.1163, 0.1164, 0.1189 | 12250, 12250, 12322, 12322, 12318 |
| single-network recovery (single network) | 9,116,477 | seeds 0 to 4: 0.1224, 0.1236, 0.1240, 0.1246, 0.1264 | 22848, 12670, 12670, 12670, 12668 |
| single-network recovery with a wider decoder | 14,267,645 | seeds 0 to 4: 0.1473, 0.1447, 0.1473, 0.1506, 0.1485 | 13878, 13878, 13878, 13878, 13878 |
| MoE recovery (half-width experts) | 8,011,740 | 0.1451, 0.1428, 0.1406, 0.1419, 0.1452 | 14252, 14252, 14254, 14250, 14254 |
| MoE recovery with full-width experts | 14,244,828 | seeds 0 to 4: 0.1763, 0.1757, 0.1721, 0.1791, 0.1763 | 15936, 15932, 15936, 15934, 15932 |
| tracking-only at twice the training length | 9,116,477 | seeds 0 and 1: 0.1212, 0.1209 | 12322, 12318 |
| single-network recovery at twice the training length | 9,116,477 | seeds 0 and 1: 0.1227, 0.1226 | 12670, 12668 |

Table 8 gives the same checkpoint, the first training of MoE recovery at seed 0 at 7,000 iterations, rolled out on an
RTX 4090 with 24 GB and on an RTX 3090 with 24 GB: the tracking error per set in metres, its mean, and R per set. The
difference of the means (RTX 4090 minus RTX 3090) is -0.000005 m.

| Rollout card | Tracking error per set (m) | Mean (m) | R per set |
| --- | --- | --- | --- |
| RTX 4090, 24 GB | 0.03434, 0.03431, 0.03434 | 0.03433 | 97, 97, 97 |
| RTX 3090, 24 GB | 0.03435, 0.03430, 0.03436 | 0.03434 | 97, 97, 97 |

The spread between the three sets of one checkpoint, over 33 checkpoints, is at most 0.00067 m (single-network
recovery, seed 3, this study) with a median of 0.00008 m.

Table 9 reads the mixing weights during training: the mean recovery-expert mixing weight on tracking / recovery /
blend controls, then the mixing cross-entropy, as the mean over the 100 iterations before the iteration named, for
every MoE training.

| Condition | Seed | Training | 1,000 | 3,000 | 5,000 | 7,000 |
| --- | --- | --- | --- | --- | --- | --- |
| MoE recovery | 0 | first | 0.043 / 0.938 / 0.024; 0.130 | 0.122 / 0.701 / 0.243; 0.433 | 0.104 / 0.737 / 0.252; 0.401 | 0.098 / 0.758 / 0.244; 0.388 |
| MoE recovery | 0 | second | 0.052 / 0.928 / 0.028; 0.160 | 0.109 / 0.709 / 0.256; 0.409 | 0.107 / 0.747 / 0.252; 0.407 | 0.101 / 0.757 / 0.250; 0.392 |
| MoE recovery | 1 | first | 0.044 / 0.937 / 0.020; 0.131 | 0.120 / 0.699 / 0.242; 0.433 | 0.102 / 0.750 / 0.246; 0.396 | 0.092 / 0.758 / 0.250; 0.378 |
| MoE recovery | 1 | second | 0.047 / 0.932 / 0.024; 0.147 | 0.113 / 0.688 / 0.247; 0.417 | 0.099 / 0.736 / 0.250; 0.387 | 0.099 / 0.758 / 0.243; 0.395 |
| MoE recovery | 2 | one | 0.045 / 0.915 / 0.021; 0.135 | 0.121 / 0.705 / 0.247; 0.429 | 0.108 / 0.756 / 0.261; 0.416 | 0.092 / 0.771 / 0.250; 0.377 |
| MoE recovery with full-width experts | 0 | one | 0.059 / 0.893 / 0.047; 0.189 | 0.117 / 0.675 / 0.245; 0.422 | 0.105 / 0.713 / 0.239; 0.402 | 0.096 / 0.734 / 0.247; 0.388 |
| MoE recovery with full-width experts | 1 | one | 0.059 / 0.796 / 0.274; 0.280 | 0.129 / 0.690 / 0.240; 0.443 | 0.110 / 0.723 / 0.249; 0.407 | 0.104 / 0.721 / 0.246; 0.403 |
| MoE recovery with full-width experts | 2 | one | 0.059 / 0.866 / 0.151; 0.200 | 0.119 / 0.688 / 0.240; 0.426 | 0.107 / 0.703 / 0.252; 0.410 | 0.106 / 0.721 / 0.241; 0.403 |
| MoE recovery with full-width experts | 3 | one | 0.053 / 0.888 / 0.044; 0.172 | 0.118 / 0.681 / 0.244; 0.427 | 0.106 / 0.709 / 0.242; 0.400 | 0.098 / 0.718 / 0.243; 0.392 |
| MoE recovery with full-width experts | 4 | one | 0.055 / 0.919 / 0.031; 0.149 | 0.128 / 0.703 / 0.241; 0.435 | 0.109 / 0.718 / 0.249; 0.409 | 0.101 / 0.714 / 0.243; 0.399 |

At iteration 7,000 the recovery-expert mixing weight is 0.092 to 0.106 on tracking controls, 0.714 to 0.771 on
recovery controls and 0.241 to 0.250 on blend controls over the ten trainings.

**Outcome: the MoE actor.** On the primary reading (this study's own trainings of tracking-only and single-network
recovery), MoE recovery (first trainings), MoE recovery (second trainings) and MoE recovery with full-width experts each
read "undecided", at two seeds and at three. Each keeps getting up: R falls short of single-network recovery's by at
most 1.7 (97.3 against 99.0, seed 2), within the 5 the rule allows. For none of the three comparisons is the cost
difference (single-network recovery minus the MoE condition) 0.003 m or more at every seed, and for none is it below
0.001 m at every seed (Table 2). The first and the second trainings both read "undecided", so the two readings do not
differ and together read "undecided". The declared rule answers the question on the MoE actor "yes" if MoE recovery
lowers the tracking cost and keeps getting up, and "no" if it does not lower it. With the outcome "undecided", the rule
gives neither answer.

**Outcome: the training length.** On the primary reading, the tracking cost at twice the training length is 0.989
times the cost at the declared length at seed 0 (above 0.8) and 0.782 times at seed 1 (above 0.5, at most 0.8). The rule
needs at most 0.5 at both seeds for "is a matter of training length" and above 0.8 at both for "is not", so it reads
"undecided".

**Exploratory (added after results).** The sensitivity reading replaces each condition's tracking error at a seed by the
mean over its completed trainings with identical arguments: three for single-network recovery at seeds 0 and 1, two for
MoE recovery at seeds 0 and 1, one otherwise. R stays that of the primary reading; replacing it by its mean changes no
outcome, the largest shortfall being 1.7. This makes the cost differences of the first-training and second-training
readings of MoE recovery coincide. On the sensitivity reading, the MoE rule reads "undecided" and every MoE condition
keeps getting up, at two and at three seeds (Table 2). The training-length rule reads "is not", with ratios 0.981 and
0.891 (Table 6).

**Exploratory (added after results).** The five-seed reading (Table 3) was declared before the trainings of seeds 2 to
4 and of the wider decoder, but after the trainings at seeds 0 and 1 of the other conditions had been read. For each
pair, the mean difference is the mean over seeds 0 to 4 of the second condition's cost minus the first's. MoE recovery
with full-width experts against single-network recovery with a wider decoder: +0.00001 m, positive at 3 seeds, "does
not lower it", on the primary and the sensitivity reading. MoE recovery with full-width experts against
single-network recovery reads +0.00174 m (primary) and +0.00154 m (sensitivity), positive at 4 seeds, "undecided";
single-network recovery with a wider decoder against single-network recovery reads +0.00173 m (primary) and +0.00153 m
(sensitivity), positive at 4 seeds, "undecided". The difference is negative at seed 4 for both comparisons with
single-network recovery (-0.00103 and -0.00115 m).

**Exploratory (added after results).** The supplementary reading. Table 10 gives per training the tracking error over
the 34 commanded clips in metres at the last checkpoint (mean of three sets), at the checkpoint 500 iterations before
it and at the one 1,000 iterations before it (one set each), and their mean, the supplementary value. The three values
of one training span 0.00018 m at least, 0.00069 m as the median and 0.00224 m at most over the 33 trainings (largest:
MoE recovery, seed 0, first training).

| Condition | Seed | Training | Tracking error at the last checkpoint (m) | at the checkpoint 500 iterations before it (m) | at the checkpoint 1,000 iterations before it (m) | Supplementary value (m) |
| --- | --- | --- | --- | --- | --- | --- |
| tracking-only | 0 | this study | 0.02744 | 0.02742 | 0.02822 | 0.02769 |
| tracking-only | 1 | this study | 0.02732 | 0.02770 | 0.02831 | 0.02778 |
| tracking-only | 2 | this study | 0.02716 | 0.02789 | 0.02852 | 0.02786 |
| tracking-only | 3 | this study | 0.02763 | 0.02789 | 0.02858 | 0.02803 |
| tracking-only | 4 | this study | 0.02713 | 0.02720 | 0.02749 | 0.02727 |
| single-network recovery | 0 | this study | 0.03418 | 0.03398 | 0.03376 | 0.03397 |
| single-network recovery | 0 | pair study's | 0.03430 | 0.03452 | 0.03460 | 0.03447 |
| single-network recovery | 0 | world-model study's | 0.03424 | 0.03400 | 0.03461 | 0.03428 |
| single-network recovery | 1 | this study | 0.03576 | 0.03528 | 0.03656 | 0.03587 |
| single-network recovery | 1 | pair study's | 0.03576 | 0.03501 | 0.03580 | 0.03553 |
| single-network recovery | 1 | world-model study's | 0.03265 | 0.03375 | 0.03337 | 0.03326 |
| single-network recovery | 2 | this study | 0.03288 | 0.03339 | 0.03358 | 0.03328 |
| single-network recovery | 3 | this study | 0.03529 | 0.03481 | 0.03534 | 0.03515 |
| single-network recovery | 4 | this study | 0.03216 | 0.03251 | 0.03258 | 0.03242 |
| MoE recovery | 0 | first | 0.03434 | 0.03657 | 0.03649 | 0.03580 |
| MoE recovery | 0 | second | 0.03586 | 0.03639 | 0.03630 | 0.03618 |
| MoE recovery | 1 | first | 0.03467 | 0.03512 | 0.03528 | 0.03502 |
| MoE recovery | 1 | second | 0.03234 | 0.03386 | 0.03373 | 0.03331 |
| MoE recovery | 2 | one | 0.03693 | 0.03736 | 0.03895 | 0.03775 |
| MoE recovery with full-width experts | 0 | one | 0.03302 | 0.03343 | 0.03385 | 0.03343 |
| MoE recovery with full-width experts | 1 | one | 0.03128 | 0.03131 | 0.03177 | 0.03145 |
| MoE recovery with full-width experts | 2 | one | 0.03248 | 0.03252 | 0.03266 | 0.03256 |
| MoE recovery with full-width experts | 3 | one | 0.03160 | 0.03198 | 0.03235 | 0.03198 |
| MoE recovery with full-width experts | 4 | one | 0.03319 | 0.03323 | 0.03297 | 0.03313 |
| single-network recovery with a wider decoder | 0 | one | 0.03169 | 0.03278 | 0.03251 | 0.03233 |
| single-network recovery with a wider decoder | 1 | one | 0.03168 | 0.03218 | 0.03237 | 0.03208 |
| single-network recovery with a wider decoder | 2 | one | 0.03162 | 0.03222 | 0.03280 | 0.03221 |
| single-network recovery with a wider decoder | 3 | one | 0.03333 | 0.03291 | 0.03364 | 0.03330 |
| single-network recovery with a wider decoder | 4 | one | 0.03331 | 0.03438 | 0.03508 | 0.03426 |
| tracking-only at twice the training length | 0 | one | 0.02524 | 0.02532 | 0.02563 | 0.02540 |
| tracking-only at twice the training length | 1 | one | 0.02550 | 0.02549 | 0.02590 | 0.02563 |
| single-network recovery at twice the training length | 0 | one | 0.03190 | 0.03171 | 0.03189 | 0.03184 |
| single-network recovery at twice the training length | 1 | one | 0.03210 | 0.03151 | 0.03215 | 0.03192 |

Table 11 reads the declared MoE rule on the supplementary value at three seeds, in the layout of Table 2.

| Comparison | Reading | Seed 0 | Seed 1 | Seed 2 | R at seeds 0; 1; 2 | Outcome, three seeds | Keeps getting up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MoE recovery, first trainings | primary | 0.00628 / 0.00811 (-0.00183) | 0.00809 / 0.00725 (+0.00084) | 0.00542 / 0.00989 (-0.00447) | 97.0 / 98.0; 98.0 / 98.0; 97.3 / 99.0 | does not lower it | yes |
| MoE recovery, first trainings | sensitivity | 0.00655 / 0.00830 (-0.00175) | 0.00711 / 0.00639 (+0.00072) | 0.00542 / 0.00989 (-0.00447) | 97.0 / 98.0; 98.0 / 98.0; 97.3 / 99.0 | does not lower it | yes |
| MoE recovery, second trainings | primary | 0.00628 / 0.00849 (-0.00221) | 0.00809 / 0.00554 (+0.00256) | 0.00542 / 0.00989 (-0.00447) | 98.0 / 98.0; 97.0 / 98.0; 97.3 / 99.0 | undecided | yes |
| MoE recovery, second trainings | sensitivity | 0.00655 / 0.00830 (-0.00175) | 0.00711 / 0.00639 (+0.00072) | 0.00542 / 0.00989 (-0.00447) | 98.0 / 98.0; 97.0 / 98.0; 97.3 / 99.0 | does not lower it | yes |
| MoE recovery with full-width experts | primary | 0.00628 / 0.00574 (+0.00054) | 0.00809 / 0.00368 (+0.00441) | 0.00542 / 0.00470 (+0.00073) | 98.0 / 98.0; 98.0 / 98.0; 98.0 / 99.0 | undecided | yes |
| MoE recovery with full-width experts | sensitivity | 0.00655 / 0.00574 (+0.00081) | 0.00711 / 0.00368 (+0.00343) | 0.00542 / 0.00470 (+0.00073) | 98.0 / 98.0; 98.0 / 98.0; 98.0 / 99.0 | undecided | yes |

On the supplementary value MoE recovery (first trainings) reads "does not lower it" on the primary and on the
sensitivity reading, MoE recovery (second trainings) reads "undecided" on the primary and "does not lower it" on the
sensitivity reading, and MoE recovery with full-width experts reads "undecided" on both. The first and second
trainings together read "undecided" on the primary reading and "does not lower it" on the sensitivity reading, and
every condition keeps getting up.

Table 12 reads the five-seed reading on the supplementary value, in the layout of Table 3.

| First | Second | Reading | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean difference | Seeds with difference > 0 | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MoE recovery with full-width experts | single-network recovery | primary | 0.00574 / 0.00628 (+0.00054) | 0.00368 / 0.00809 (+0.00441) | 0.00470 / 0.00542 (+0.00073) | 0.00394 / 0.00711 (+0.00317) | 0.00586 / 0.00515 (-0.00071) | +0.00163 | 4 | undecided |
| MoE recovery with full-width experts | single-network recovery | sensitivity | 0.00574 / 0.00655 (+0.00081) | 0.00368 / 0.00711 (+0.00343) | 0.00470 / 0.00542 (+0.00073) | 0.00394 / 0.00711 (+0.00317) | 0.00586 / 0.00515 (-0.00071) | +0.00148 | 4 | undecided |
| MoE recovery with full-width experts | single-network recovery with a wider decoder | primary | 0.00574 / 0.00463 (-0.00110) | 0.00368 / 0.00430 (+0.00062) | 0.00470 / 0.00435 (-0.00034) | 0.00394 / 0.00526 (+0.00132) | 0.00586 / 0.00698 (+0.00112) | +0.00032 | 3 | does not lower it |
| MoE recovery with full-width experts | single-network recovery with a wider decoder | sensitivity | 0.00574 / 0.00463 (-0.00110) | 0.00368 / 0.00430 (+0.00062) | 0.00470 / 0.00435 (-0.00034) | 0.00394 / 0.00526 (+0.00132) | 0.00586 / 0.00698 (+0.00112) | +0.00032 | 3 | does not lower it |
| single-network recovery with a wider decoder | single-network recovery | primary | 0.00463 / 0.00628 (+0.00165) | 0.00430 / 0.00809 (+0.00379) | 0.00435 / 0.00542 (+0.00107) | 0.00526 / 0.00711 (+0.00185) | 0.00698 / 0.00515 (-0.00184) | +0.00130 | 4 | undecided |
| single-network recovery with a wider decoder | single-network recovery | sensitivity | 0.00463 / 0.00655 (+0.00192) | 0.00430 / 0.00711 (+0.00281) | 0.00435 / 0.00542 (+0.00107) | 0.00526 / 0.00711 (+0.00185) | 0.00698 / 0.00515 (-0.00184) | +0.00116 | 4 | undecided |

Table 13 reads the training-length rule on the supplementary value, in the layout of Table 6.

| Reading | Seed | Cost at 7,000 | Cost at 14,000 | Ratio | Outcome |
| --- | --- | --- | --- | --- | --- |
| primary | 0 | 0.00628 | 0.00644 | 1.026 | |
| primary | 1 | 0.00809 | 0.00629 | 0.778 | |
| primary | both seeds | | | | undecided |
| sensitivity | 0 | 0.00655 | 0.00644 | 0.984 | |
| sensitivity | 1 | 0.00711 | 0.00629 | 0.885 | |
| sensitivity | both seeds | | | | is not |

Table 14 reads each condition on the supplementary value: the number of trainings and the mean and standard deviation
(n - 1) of the supplementary values, in metres.

| Condition | Trainings | Mean (m) | Standard deviation (m) |
| --- | --- | --- | --- |
| tracking-only | 5 | 0.02773 | 0.00028 |
| single-network recovery | 9 | 0.03425 | 0.00114 |
| MoE recovery | 5 | 0.03561 | 0.00162 |
| MoE recovery with full-width experts | 5 | 0.03251 | 0.00081 |
| single-network recovery with a wider decoder | 5 | 0.03283 | 0.00093 |
| tracking-only at twice the training length | 2 | 0.02551 | 0.00016 |
| single-network recovery at twice the training length | 2 | 0.03188 | 0.00006 |

**What the results do not show.**

- **Seeds 2 to 4.** The rule outcomes at seeds 2 to 4 rest on one training per condition and seed; at seeds 0 and 1
  two or three trainings exist only for single-network recovery and MoE recovery (Table 5), and trainings with
  identical arguments at one seed differ by up to the largest value of Table 5.
- **Checkpoints.** Between checkpoints 500 iterations apart the tracking error of one training moves (Table 10).
- **Learning rate.** At the last checkpoint the learning rate is still close to its start: it is a cosine over
  `--lr_decay_iters 250000`, so 0.998 of its start at iteration 7,000 and 0.993 at 14,000; the policy average has
  decay 0.995 (`--pi_ema`).
- **Normalizers.** The critic's input normalizer stops updating near iteration 2,960 in every training of either
  length; the policy and world-model normalizers do not stop within 14,000 iterations.
- **Mixing term.** The weighted mixing term of the actor loss is near 0.04 (0.1 times a mixing cross-entropy near 0.4
  at iteration 7,000, Table 9), beside an actor loss of -8.975 to -8.602 in the MoE trainings.
- **Simulation.** The rollouts are kinematic readings of simulated trials in MuJoCo Warp; no hardware was used.

