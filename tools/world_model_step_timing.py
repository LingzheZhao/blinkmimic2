"""World-model step timing: the cost of one world-model step (one application of a world model, one control) in
BlinkMimic 2's updates on one card, for the trainer's world model (class ``WorldModel`` in networks.py) and for
DynaMimic's physics-informed world model (arXiv 2611.32944), whose state transition is a force predictor (contact
wrenches at candidate support links and DynaMimic's residual generalized force) followed by a rigid-body integrator.

Two workloads per world model, forward and backward, with train.py's precision settings (TF32 matmuls, bf16 autocast
for MLP products only, elementwise 3x3 rotation products), each a CUDA-graph replay of one whole update as train.py's
``Graphed`` runs them, after torch.compile:

  policy           32 world-model steps of an imagined rollout at the policy batch (2,048): each control's action is a
                   leaf plus a small state feedback (so the backward pass runs through every state), the loss the sum
                   of the final states; gradients with respect to the action leaves; world-model parameters not updated
  world_model_fit  8 world-model steps under recorded actions at the world model's batch (2,048) with gradients with
                   respect to the world model's parameters, as one of train.py's ``wm_update`` (5 per iteration)

World models (rows):

  world_model    WorldModel (structured, hidden 1024 x 3)
  physics_informed_{N}_substeps[_tree][_ltdl|_columns|_lu][_without_tf32]
                 a force predictor (an MLP with hidden 1024 x 3, the world model's width; input: the world model's
                 inputs plus height, root-frame position and world-frame velocity of each candidate link; output
                 4 x L x 6 contact wrenches + 35 for DynaMimic's residual generalized force), the decoder of Eq. 4, and
                 N integration timesteps of the rigid-body integrator rigid_body.RigidBody (N = 2: DynaMimic's
                 2 x 10 ms; N = 4: 4 x 5 ms, one per physics substep of the simulator)

"substeps" in the row names and "physics_substep" in the report key
``accuracy_one_10ms_physics_substep_against_cpu_float64`` mean integration timesteps of the rigid-body integrator,
not the simulator's 5 ms physics substeps.

Options per physics-informed row: the dense path (Jacobians of every body, M = sum J^T I J) or the tree path (_tree:
composite rigid-body mass matrix, Newton-Euler bias with the link wrenches folded in, no Jacobian tensors); the linear
solve (batched Cholesky, cholesky_ex without error raising; _lu: LU solve; _ltdl: MuJoCo's tree-sparse L^T D L,
unrolled; _columns: Cholesky column by column in elementwise operations); float32 with and without TF32 in the
rigid-body products (_without_tf32). Accuracy: one 10 ms integration timestep on the card (dense path with batched
Cholesky, tree path with L^T D L and with the column-by-column solve) against the same integration timestep in
float64 on the CPU, from random states.

Projection: seconds per iteration = baseline + (physics-informed - world model) for the policy update
+ 5 x (physics-informed - world model) for the world-model updates (the data collection, critic
and policy MLP are unchanged). One process per row: a failed CUDA-graph capture leaves its process's CUDA context
unusable; ``--aggregate`` merges the per-row reports.

  python tools/world_model_step_timing.py --rows world_model --links L1 ... --out row.json
  python tools/world_model_step_timing.py --aggregate row1.json row2.json ... --out timing.json \
      [--baseline_seconds_per_iteration 0.12]
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import torch
import torch.nn as nn

import blinkmimic2.networks as networks
import blinkmimic2.rotations as rotations
from blinkmimic2.dr import XiLayout
from blinkmimic2.mdp import inv_dim, invariant_features
from blinkmimic2.networks import RunningNorm, WorldModel, mlp, run_mlp
from blinkmimic2.rigid_body import RigidBody, from_trainer, to_trainer, trainer_model
from blinkmimic2.robot import get_robot

LINKS = None  # set in main


class PhysicsInformedWorldModel(nn.Module):
    """Force predictor (an MLP) + Eq. 4 decoder + rigid-body integrator: the physics-informed world model, untrained."""

    def __init__(self, rb: RigidBody, robot, layout, links, n_sub, hidden=(1024, 1024, 1024), solve="cholesky", tree=False):
        super().__init__()
        self.rb, self.n_sub, self.solve, self.tree = rb, n_sub, solve, tree
        # a device tensor: indexing with a Python list would copy host-to-device, which a CUDA-graph capture refuses
        self.register_buffer("links", torch.tensor(links, device=rb.device))
        J, L = robot.num_joints, len(links)
        self.L = L
        self.in_dim = inv_dim(J) + J + layout.dim + 7 * L
        self.out_dim = 4 * L * 6 + rb.nv
        self.net = mlp(self.in_dim, list(hidden), self.out_dim, act=nn.SiLU, layer_norm=True)
        self.in_norm = RunningNorm(self.in_dim)
        dev = rb.device
        self.register_buffer("default_q", torch.tensor(robot.default_joint_pos_list(), device=dev))
        self.register_buffer("alpha", torch.tensor(robot.action_scale_list(), device=dev))
        self.register_buffer("ell", torch.full((L, 3), 0.1, device=dev))
        self.mg = rb.total_mass * 9.81
        self.sl_off, self.sl_fric, self.sl_com = layout.slices["joint_offset"], layout.slices["friction"], layout.slices["com"]

    def decode(self, u, mu):
        """Eq. 4: u (B, 4, L, 6) -> world wrench (force, moment about the link origin)."""
        uz, uxy, un = u[..., 2:3], u[..., 0:2], u[..., 3:6]
        Fz = self.mg * nn.functional.softplus(uz - 2)
        rho = torch.clamp(torch.linalg.vector_norm(uxy, dim=-1, keepdim=True), min=1e-6)
        Fxy = mu * Fz * torch.tanh(rho) / rho * uxy
        ell = self.ell * torch.cat((torch.ones_like(mu), torch.ones_like(mu), mu), -1)  # (B, 1, L, 3): (l_x, l_y, mu l_z)
        n = Fz * ell * torch.tanh(un)
        return torch.cat((Fxy, Fz, n), -1)

    def compile_parts(self, fullgraph: bool):
        """--compile_unit physics_substep: the force predictor with the decoder, and one integration timestep, each
        compiled on its own (graphs half the size of a control's and reused by every integration timestep)."""
        self._predict = torch.compile(self._predict, dynamic=False, fullgraph=fullgraph)
        self._substep = torch.compile(self._substep, dynamic=False, fullgraph=fullgraph)

    def _substep(self, p, R, q, nu, target, dt, tau_res, Wi, com):
        fn = self.rb.substep_tree if self.tree else self.rb.substep
        return fn(p, R, q, nu, target, dt, tau_ext=tau_res, links=self.links, wrench=Wi, com_offset=com, solve=self.solve)

    def forward(self, S, a, xi, xi_n):
        p, R, q, nu, W, tau_res, target, com = self._predict(S, a, xi, xi_n)
        per = 4 // self.n_sub
        dt = 0.02 / self.n_sub
        for i in range(self.n_sub):
            p, R, q, nu = self._substep(p, R, q, nu, target, dt, tau_res, W[:, i * per : (i + 1) * per].mean(1), com)
        return to_trainer(p, R, q, nu)

    def _predict(self, S, a, xi, xi_n):
        """Force predictor and Eq. 4 decoder: the contact wrenches of the four 5 ms intervals, DynaMimic's residual
        generalized force, the joint targets and the state in MuJoCo's convention."""
        rb = self.rb
        p, R, q, nu = from_trainer(S)
        com = xi[..., self.sl_com]
        kin = rb.kinematics(p, R, q, com)
        x = kin["x"][..., self.links, :]
        if self.tree:
            vl = rb.velocities(kin, nu)[1][:, self.links]
        else:
            Jp, _ = rb.jacobians(kin)
            vl = torch.einsum("bldi,bd->bli", Jp[:, self.links], nu)
        rel = ((x - p.unsqueeze(-2)).unsqueeze(-1) * R.unsqueeze(-3)).sum(-2)  # R^T (x - p)
        feats = torch.cat((x[..., 2], rel.flatten(-2), vl.flatten(-2)), -1)
        y = run_mlp(self.net, self.in_norm(torch.cat((invariant_features(S), a, xi_n, feats), -1)))
        B = y.shape[0]
        W = self.decode(y[:, : 4 * self.L * 6].view(B, 4, self.L, 6), xi[..., self.sl_fric].view(B, 1, 1, 1))
        tau_res = y[:, 4 * self.L * 6 :]
        target = self.default_q + xi[..., self.sl_off] + self.alpha * a
        return p, R, q, nu, W, tau_res, target, com


def random_states(B, dev, J, rb, gen):
    p = torch.zeros(B, 3, device=dev)
    p[:, 2] = 0.3 + 0.5 * torch.rand(B, device=dev, generator=gen)
    v = torch.randn(B, 3, device=dev, generator=gen) * 0.5
    rv = torch.randn(B, 3, device=dev, generator=gen) * 0.8
    R = rotations.so3_exp(rv)
    w = torch.randn(B, 3, device=dev, generator=gen)
    q = torch.randn(B, J, device=dev, generator=gen) * 0.3
    qd = torch.randn(B, J, device=dev, generator=gen) * 2
    return (p, R, v, w, q, qd)


def time_graph(fn, reps=20, warm=3):
    """Capture fn (one update) in a CUDA graph after warm-up calls on a side stream; mean replay seconds."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def time_eager(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline_seconds_per_iteration", type=float, default=0.12)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--fit_window", type=int, default=8)
    ap.add_argument("--fit_updates", type=int, default=5)
    ap.add_argument("--links", nargs="+", default=None)
    ap.add_argument("--aggregate", nargs="+", default=None, help="per-row reports to merge into --out (no timing)")
    ap.add_argument("--rows", nargs="+", default=None)
    ap.add_argument("--fullgraph", action="store_true", help="compile each unit as one graph; a graph break raises")
    ap.add_argument("--compile_unit", choices=("control", "physics_substep"), default="control",
                    help="physics-informed rows: compile one control, or the force predictor and one integration "
                    "timestep of the rigid-body integrator apart ('physics_substep' names that integration timestep)")
    ap.add_argument("--device", default="cuda:0", help="cpu: a short test on the CPU (eager, no graphs, no compile)")
    a = ap.parse_args()
    if a.aggregate:
        return aggregate(a.aggregate, a.out, a.baseline_seconds_per_iteration, a.fit_updates)
    dev = a.device
    cpu_short_test = dev == "cpu"
    if cpu_short_test:
        torch.cuda.synchronize = lambda: None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16
    networks.RunningNorm.no_sync = True
    rotations.ELEMENTWISE = True
    robot = get_robot("mjlab_g1")
    J = robot.num_joints
    layout = XiLayout(robot, dev)
    m = trainer_model(robot)
    rb = RigidBody(m, robot.joint_names, device=dev, dtype=torch.float32)
    links = rb.index(a.links)
    gen = torch.Generator(device=dev).manual_seed(0)
    B, H, Wn = a.batch, a.horizon, a.fit_window
    S0 = random_states(B, dev, J, rb, gen)
    xi = torch.zeros(B, layout.dim, device=dev)
    xi[:, layout.slices["friction"]] = 0.8
    xi_n = layout.normalize(xi)
    report = {"card": torch.cuda.get_device_name(0) if not cpu_short_test else "cpu", "torch": torch.__version__, "batch": B, "horizon": H,
              "fit_window": Wn, "fit_updates": a.fit_updates, "links": a.links, "rows": {}}  # fmt: skip

    # ------------------------------------------------------------------ accuracy of the card's float32 step
    acc = {}
    rb64 = RigidBody(m, robot.joint_names, device="cpu", dtype=torch.float64)
    na = min(256, B)
    Sa = tuple(x[:na] for x in S0)
    tgt = torch.randn(na, J, device=dev, generator=gen) * 0.3
    tau = torch.randn(na, rb.nv, device=dev, generator=gen) * 5
    ref = rb64.substep(*from_trainer(tuple(x.double().cpu() for x in Sa)), tgt.double().cpu(), 0.01, tau_ext=tau.double().cpu())
    dnu_ref = ref[3] - from_trainer(tuple(x.double().cpu() for x in Sa))[3]
    for path, fn, method in (("dense_cholesky", rb.substep, "cholesky"), ("tree_ltdl", rb.substep_tree, "ltdl"),
                             ("tree_columns", rb.substep_tree, "columns")):
        for tf32 in (True, False):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            out = fn(*from_trainer(Sa), tgt, 0.01, tau_ext=tau, solve=method)
            e = (out[3].double().cpu() - ref[3]).abs().amax(-1) / dnu_ref.abs().amax(-1).clamp(min=1e-9)
            acc[f"{path}_tf32_{tf32}"] = {"velocity_change_rel_err_max": float(e.max()), "median": float(e.median())}
    torch.backends.cuda.matmul.allow_tf32 = True
    report["accuracy_one_10ms_physics_substep_against_cpu_float64"] = acc
    print(json.dumps(acc), flush=True)

    # ------------------------------------------------------------------ world models
    def make(name):
        if name == "world_model":
            return WorldModel(J, layout.dim, use_prev_action=False, hidden=(1024, 1024, 1024), structured=True, dt=0.02).to(dev)
        n_sub = int(re.search(r"_(\d+)_substeps", name).group(1))
        solve = next((m for m in ("ltdl", "columns", "lu") if f"_{m}" in name), "cholesky")
        return PhysicsInformedWorldModel(rb, robot, layout, links, n_sub, solve=solve, tree="tree" in name).to(dev)

    def call(model, S, a_, xi_):
        if isinstance(model, WorldModel):
            return model.step(S, a_, a_, xi_n[: S[0].shape[0]])
        return model(S, a_, xi_, xi_n[: S[0].shape[0]])

    rows = a.rows or ["world_model"] + [f"physics_informed_{n}_substeps{o}" for n, o in (
        (2, ""), (4, ""), (2, "_lu"), (2, "_without_tf32"), (2, "_tree"), (2, "_tree_ltdl"), (4, "_tree"), (4, "_tree_ltdl"))]
    for name in rows:
        model = make(name.replace("_without_tf32", ""))
        tf32 = "without_tf32" not in name
        step = (lambda S, a_, xi_, model=model: call(model, S, a_, xi_))
        if not cpu_short_test:
            if a.compile_unit == "physics_substep" and not isinstance(model, WorldModel):
                model.compile_parts(a.fullgraph)
            else:
                step = torch.compile(step, dynamic=False, fullgraph=a.fullgraph)
        K = nn.Linear(inv_dim(J), J, bias=False).to(dev)
        nn.init.normal_(K.weight, std=1e-3)
        A_leaf = torch.zeros(H, B, J, device=dev, requires_grad=True)
        S_init = tuple(x.clone() for x in S0)
        model.requires_grad_(False)
        K.requires_grad_(False)

        def policy_fn():
            A_leaf.grad = None
            S = S_init
            for t in range(H):
                act = A_leaf[t] + torch.tanh(K(invariant_features(S)))
                S = step(S, act, xi)
            loss = sum(x.float().sum() for x in S if x.dim() == 2) + S[1].sum()
            loss.backward()

        params = [p_ for p_ in model.parameters()]
        A_rec = torch.randn(Wn, B, J, device=dev, generator=gen) * 0.3

        def fit_fn():
            for p_ in params:
                p_.grad = None
            S = S_init
            loss = 0
            for t in range(Wn):
                S = step(S, A_rec[t], xi)
                loss = loss + sum((x.float() ** 2).mean() for x in S)
            loss.backward()

        torch.backends.cuda.matmul.allow_tf32 = tf32
        res = {}
        from torch._dynamo.utils import counters

        report["rows"][name] = res
        report["compile_unit"] = a.compile_unit

        def attempt(key, fn):
            try:
                res[key] = fn()
            except Exception as e:  # noqa: BLE001
                res[key + "_error"] = repr(e)[:1500]
            json.dump(report, open(a.out, "w"), indent=1)  # after every measurement: a timeout keeps what was taken
            print(name, key, res.get(key, str(res.get(key + "_error"))[:300]), flush=True)

        def with_grad(flag, fn):
            def run():
                model.requires_grad_(flag)
                return fn()
            return run

        # the policy workload is captured before the fit compiles (a failed capture leaves the CUDA context unusable)
        attempt("policy_eager_compiled_s", with_grad(False, lambda: time_eager(policy_fn)))
        res["peak_memory_gib_policy_eager"] = torch.cuda.max_memory_allocated() / 2**30 if not cpu_short_test else None
        if not cpu_short_test:
            attempt("policy_graph_s", with_grad(False, lambda: time_graph(policy_fn)))
        if "policy_graph_s_error" not in res:
            attempt("world_model_fit_eager_compiled_s", with_grad(True, lambda: time_eager(fit_fn)))
        res["dynamo_unique_graphs"] = int(counters["stats"]["unique_graphs"])
        if not cpu_short_test and "policy_graph_s" in res:
            attempt("world_model_fit_graph_s", with_grad(True, lambda: time_graph(fit_fn)))
        if cpu_short_test:
            res["policy_graph_s"], res["world_model_fit_graph_s"] = res.get("policy_eager_compiled_s"), res.get("world_model_fit_eager_compiled_s")
        if any(k.endswith("_error") for k in res):
            res["error"] = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if not cpu_short_test:
            torch.cuda.reset_peak_memory_stats()
        report["rows"][name] = res
        print(name, json.dumps(res), flush=True)
        del model, step
        if not cpu_short_test:
            torch.cuda.empty_cache()

    json.dump(report, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def aggregate(files, out, baseline, fit_updates):
    """Merge per-row reports (one process per row) and project the seconds per training iteration."""
    reps = [json.load(open(f)) for f in files]
    report = {k: v for k, v in reps[0].items() if k != "rows"}
    report["rows"] = {}
    for r in reps:
        report["rows"].update(r["rows"])
    report["accuracy_one_10ms_physics_substep_against_cpu_float64"] = {f: r["accuracy_one_10ms_physics_substep_against_cpu_float64"] for f, r in zip(files, reps)}
    base = report["rows"].get("world_model", {})
    proj = {}
    for name, r in report["rows"].items():
        if name == "world_model":
            continue
        for mode in ("graph", "eager_compiled"):
            if any(f"{w}_{mode}_s" not in d for w in ("policy", "world_model_fit") for d in (r, base)):
                continue
            dp = r[f"policy_{mode}_s"] - base[f"policy_{mode}_s"]
            dw = r[f"world_model_fit_{mode}_s"] - base[f"world_model_fit_{mode}_s"]
            s = baseline + dp + fit_updates * dw
            proj[f"{name}/{mode}"] = {"seconds_per_iteration": s, "ratio_to_baseline": s / baseline,
                                      "policy_only_seconds": baseline + dp, "policy_only_ratio": (baseline + dp) / baseline}  # fmt: skip
    report["projection"] = proj
    report["baseline_seconds_per_iteration"] = baseline
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(proj, indent=1))


if __name__ == "__main__":
    main()
