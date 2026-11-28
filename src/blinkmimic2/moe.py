"""The mixture-of-experts (MoE) actor of StableMimic (arXiv 2608.02385v1, Section III, Eq. 3, and the
implementation-details table), which the paper calls a proprioception-gated dual-expert MoE, as train.py's --moe option
(RECOVERY.md, Mixture-of-experts actor): the mixing targets of imagined controls and the three mixing losses. The
mixing network is StableMimic's "gate". The networks are in networks.Policy (``moe_prop_dim``).

The two experts map the decoder's input (the motion latent and the normalized observation history) to an action
each; the mixing network maps the proprioceptive part of the normalized history to two mixing weights,
softmax(logits / TAU), and the action is w_m mu_m + w_r mu_r. The regime labels of imagined controls enter the losses
only.
"""

from __future__ import annotations

import torch

from .recovery import BLEND, RECOVERY, TRACKING, regime

TAU = 1.0  # softmax temperature; the paper gives none
MIXING_HIDDEN = (256, 128)
CE_WEIGHT = 0.1  # mixing cross-entropy: the mixing weights against the mixing target
BLEND_CE_WEIGHT = 4.0  # StableMimic's "transition 4.0" (its transition is this repository's blend), read as the
# weight of blend controls inside the mixing cross-entropy
CONSISTENCY_WEIGHT = 0.01  # mixing consistency between consecutive controls of one regime
ALIGNMENT_WEIGHT = 0.01  # expert alignment: the two experts' actions on blend controls
RESET_HEIGHT_RATIO = 0.8  # a recovery start placed at least this share of its command row's pelvis height keeps the
# tracking target for its episode
FLAG = "keeps_tracking_target"  # the per-episode flag of that rule, kept by the simulator and stored with every
# transition


def tracking_target_at_reset(recovering: torch.Tensor, body_height: torch.Tensor, command_height: torch.Tensor):
    """The flag of episodes that start at a reset with these regimes: a recovery start whose placed pelvis height is at
    least RESET_HEIGHT_RATIO times its command row's pelvis height."""
    return recovering & (body_height >= RESET_HEIGHT_RATIO * command_height)


def mixing_targets(recovering: torch.Tensor, blend_left: torch.Tensor, blend_controls: int,
                   tracking_episode: torch.Tensor):  # fmt: skip
    """The label and the mixing target of controls that start from states with these regime flags and blend counts, in
    episodes with the flag ``tracking_episode`` (broadcast). The label is the control's regime (recovery.regime), or
    TRACKING in a flagged episode. The target weight of the recovery expert is 1 for RECOVERY, 0 for TRACKING and
    (left - 1) / BLEND for a BLEND control with ``left`` blend controls still to run: one minus the blend control's
    reward weight (recovery.advance), 74/75 at the first blend control and 0 at the last. Returns (label, y_r)."""
    label = torch.where(tracking_episode, TRACKING, regime(recovering, blend_left))
    y_blend = (blend_left - 1).clamp(min=0).float() / blend_controls
    y_r = torch.where(label == RECOVERY, 1.0, torch.where(label == BLEND, y_blend, 0.0))
    return label, y_r


def mixing_losses(log_w: torch.Tensor, y_r: torch.Tensor, label: torch.Tensor, mu_m: torch.Tensor, mu_r: torch.Tensor,
                  survival: torch.Tensor):  # fmt: skip
    """The three mixing losses over an imagined rollout of H controls and B rollouts (dim 0 is time): ``log_w``
    (H, B, 2) the log mixing weights (tracking expert first), ``y_r`` and ``label`` (H, B) from ``mixing_targets``,
    ``mu_m`` and ``mu_r``
    (H, B, A) the experts' actions, ``survival`` (H, B) the survival weight of each control (no gradient), which
    weights every term as termination ends a rollout of the simulator. Branch-free (captured in CUDA graphs).

    - cross-entropy: -(y_m log w_m + y_r log w_r), times BLEND_CE_WEIGHT on BLEND-labelled controls, survival-weighted
      mean over all controls;
    - consistency: (w_r(t+1) - w_r(t))^2 over consecutive controls of one rollout with equal labels, survival-weighted
      (the later control's) mean over those pairs;
    - alignment: the mean over action dimensions of (mu_m - mu_r)^2 on BLEND-labelled controls, survival-weighted mean
      over them (0 without one).
    Returns (cross-entropy, consistency, alignment), each 0-dim."""
    s = survival
    y = torch.stack((1 - y_r, y_r), -1)
    omega = torch.where(label == BLEND, BLEND_CE_WEIGHT, 1.0)
    ce = (s * omega * -(y * log_w).sum(-1)).sum() / s.sum().clamp(min=1e-8)
    w_r = log_w[..., 1].exp()
    pair = (label[1:] == label[:-1]).to(s.dtype) * s[1:]
    consistency = (pair * (w_r[1:] - w_r[:-1]) ** 2).sum() / pair.sum().clamp(min=1e-8)
    blend = (label == BLEND).to(s.dtype) * s
    alignment = (blend * ((mu_m - mu_r) ** 2).mean(-1)).sum() / blend.sum().clamp(min=1e-8)
    return ce, consistency, alignment


def rollout_mixing_losses(pi, Z: torch.Tensor, O: torch.Tensor, label: torch.Tensor, y_r: torch.Tensor,
                          survival: torch.Tensor):  # fmt: skip
    """The mixing losses (``mixing_losses``) of an imagined rollout of the policy ``pi`` (networks.Policy with the
    experts) from its latents Z (H, B, z) and normalized histories O (H, B, hist, obs). The histories hold imagined
    states: the mixing network and the experts are evaluated again on them without gradient to the rollout, so that the
    losses train the mixing network and the experts and not the actions that led to those states; the latents depend
    on the reference rows only (their gradient reaches the encoder through the alignment term). Returns the three
    losses and the recovery expert's mixing weight
    (H, B) without gradient."""
    O = O.detach()
    log_w = pi.mixing_log_weights(O)
    mu_m, mu_r = pi.experts(Z, O)
    ce, consistency, alignment = mixing_losses(log_w, y_r, label, mu_m, mu_r, survival)
    return ce, consistency, alignment, log_w[..., 1].detach().exp()


STATS = 11  # length of the ``stats`` vector


def stats(w_r: torch.Tensor, regimes: torch.Tensor, tracking_episode: torch.Tensor, ce: torch.Tensor,
          consistency: torch.Tensor, alignment: torch.Tensor) -> torch.Tensor:
    """The summands of train.py's --moe log columns of one update, a vector of STATS: per regime of the controls
    (TRACKING, RECOVERY, BLEND, recovery.regime, flag or not) the summed recovery-expert weight ``w_r`` (H, B) and the
    number of controls, interleaved; the three losses; 1 (the update); the recovery controls whose episode keeps the
    tracking target (``tracking_episode`` broadcast to ``regimes``). No gradient."""
    with torch.no_grad():
        onehot = (regimes.unsqueeze(-1) == torch.arange(3, device=regimes.device)).to(w_r.dtype)
        per = torch.stack(((w_r.unsqueeze(-1) * onehot).sum((0, 1)), onehot.sum((0, 1))), -1).flatten()
        flagged = ((regimes == RECOVERY) & tracking_episode).sum().to(w_r.dtype)
        rest = torch.stack((ce.detach(), consistency.detach(), alignment.detach(), torch.ones_like(flagged), flagged))
        return torch.cat((per, rest.to(w_r.dtype)))


def total(ce: torch.Tensor, consistency: torch.Tensor, alignment: torch.Tensor) -> torch.Tensor:
    """The mixing losses' term of the actor's loss."""
    return CE_WEIGHT * ce + CONSISTENCY_WEIGHT * consistency + ALIGNMENT_WEIGHT * alignment
