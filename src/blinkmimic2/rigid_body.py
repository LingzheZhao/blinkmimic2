"""Rigid-body dynamics of the trainer's robot in batched, differentiable torch: the rigid-body integrator of DynaMimic's
physics-informed world model (arXiv 2611.32944), in which a force predictor predicts contact wrenches at candidate
support links and DynaMimic's residual generalized force, and the rigid-body integrator computes the next state from them
(forward dynamics).

Every constant is read from the ``MjModel`` the training simulator steps (``trainer_model``: the scene of
``env.make_scene_cfg`` compiled by mjlab, with ``env.make_sim``'s options applied), so the quantities follow the
model MuJoCo integrates: the mass matrix with the rotor armature, the bias forces (gravity, Coriolis and centrifugal),
body-origin Jacobians, the built-in position actuators (force = kp (target - q) - kd qd, clamped to the force range)
and the passive joint damping (zero on the G1, kept for generality).

Generalized coordinates follow MuJoCo: ``nu = (v, w_local, qd)`` with ``v`` the world linear velocity of the root
origin and ``w_local`` the root angular velocity in the root frame; the configuration is ``(p, R, q)``. The trainer's
state tuple (mdp.py) holds the world angular velocity ``w = R w_local``; ``from_trainer`` / ``to_trainer`` convert.

One integration timestep follows MuJoCo's ``implicitfast`` integrator: with D the velocity derivative of the actuator
and passive forces (actuators clamped by their force range contribute nothing),

    (M - dt D) (nu1 - nu) = dt (tau_act + tau_pass - h + tau_ext),   q1 = q (+) dt nu1,

which is DynaMimic's ``(M + dt B)`` with ``B = -D``. ``tau_ext`` is the constraint force: in DynaMimic the link wrenches
mapped through the link Jacobians plus DynaMimic's residual generalized force.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .rotations import so3_exp

FREE = mujoco.mjtJoint.mjJNT_FREE
HINGE = mujoco.mjtJoint.mjJNT_HINGE


def trainer_model(robot) -> mujoco.MjModel:
    """The MjModel of the training simulator (one environment, compiled on the CPU)."""
    from mjlab.scene import Scene

    from .env import make_scene_cfg, make_sim

    m = Scene(make_scene_cfg(robot, 1), device="cpu").compile()
    make_sim(robot).mujoco.apply(m)
    return m


def _cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


def _mv(A, v):
    """A @ v for (..., 3, 3) and (..., 3), elementwise (no TF32 on the card)."""
    return (A * v.unsqueeze(-2)).sum(-1)


def _mm(A, B):
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def _hinge_mat(axis, angle):
    """Rotation by ``angle`` (...) about the unit ``axis`` (3,)."""
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    return torch.stack((c + x * x * C, x * y * C - z * s, x * z * C + y * s,
                        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
                        z * x * C - y * s, z * y * C + x * s, c + z * z * C), -1).reshape(angle.shape + (3, 3))  # fmt: skip


def _quat_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


class RigidBody:
    """The articulated robot of a MuJoCo model: one free root and one hinge at the origin of every other body.

    ``joint_names`` (the trainer's joint order) must be the model's dof order after the root (the constructor refuses
    otherwise), so ``q`` and
    ``qd`` are the trainer's joint vectors and ``nu[..., 6:] = qd``. ``com_body`` names the body whose centre of mass
    the trainer randomizes (an offset per environment, ``xi`` slice ``com``); ``com_offset`` arguments add to its ``ipos``."""

    def __init__(self, m: mujoco.MjModel, joint_names, device="cpu", dtype=torch.float64, prefix: str = "robot/",
                 com_body: str | None = "torso_link"):  # fmt: skip
        self.device, self.dtype = device, dtype
        self.joint_names = tuple(joint_names)
        bodies = [b for b in range(m.nbody) if m.body_jntnum[b] > 0]
        root = bodies[0]
        if m.body_jntnum[root] != 1 or m.jnt_type[m.body_jntadr[root]] != FREE or m.body_parentid[root] != 0:
            raise ValueError("the first moving body must carry one free joint and hang from the world")
        if m.nv != 6 + len(bodies) - 1 or m.nv != 6 + len(joint_names):
            raise ValueError(f"nv {m.nv}: expected 6 + one hinge per body ({len(bodies) - 1}) = 6 + {len(joint_names)} joints")
        local = {b: i for i, b in enumerate(bodies)}
        self.names = [m.body(b).name.removeprefix(prefix) for b in bodies]
        self.mj_body = bodies
        nb = len(bodies)
        self.nb, self.nv, self.nj = nb, m.nv, m.nv - 6
        parent, pos, Roff, axis, dof_of_body = [-1], [np.zeros(3)], [np.eye(3)], [np.zeros(3)], [-1]
        for b in bodies[1:]:
            j = m.body_jntadr[b]
            if m.body_jntnum[b] != 1 or m.jnt_type[j] != HINGE:
                raise ValueError(f"body {m.body(b).name}: expected one hinge")
            if np.abs(m.jnt_pos[j]).max() > 0:
                raise ValueError(f"joint {m.joint(j).name}: anchor off the body origin")
            if m.qpos0[m.jnt_qposadr[j]] != 0:
                raise ValueError(f"joint {m.joint(j).name}: qpos0 != 0")
            name = m.joint(j).name.removeprefix(prefix)
            d = m.jnt_dofadr[j]
            if d != 6 + list(joint_names).index(name):
                raise ValueError(f"joint {name}: dof {d} is not 6 + its index in joint_names")
            if m.body_parentid[b] not in local:
                raise ValueError(f"body {m.body(b).name}: parent is not a moving body")
            parent.append(int(local[m.body_parentid[b]]))
            pos.append(m.body_pos[b].copy())
            Roff.append(_quat_mat(m.body_quat[b]))
            axis.append(m.jnt_axis[j] / np.linalg.norm(m.jnt_axis[j]))
            dof_of_body.append(int(d))  # a Python int: a numpy index would break the compiled graph
        self.parent = parent
        self.dof_of_body = dof_of_body  # dof of each body's hinge (-1 for the root)
        t = lambda x: torch.tensor(np.asarray(x), dtype=dtype, device=device)  # noqa: E731
        self.pos, self.Roff, self.axis = t(np.array(pos)), t(np.array(Roff)), t(np.array(axis))
        self.mass = t(m.body_mass[bodies])
        self.ipos = t(m.body_ipos[bodies])
        I = np.stack([_quat_mat(m.body_iquat[b]) @ np.diag(m.body_inertia[b]) @ _quat_mat(m.body_iquat[b]).T for b in bodies])
        self.inertia = t(I)  # body frame, about the centre of mass
        self.armature = t(m.dof_armature)
        self.damping = t(m.dof_damping)
        if np.abs(m.dof_frictionloss).max() > 0 or np.abs(m.jnt_stiffness).max() > 0:
            raise ValueError("frictionloss or joint stiffness: not modelled")
        self.gravity = t(m.opt.gravity)
        # ancestors: dof d moves body b
        mask = np.zeros((nb, m.nv))
        for i in range(nb):
            mask[i, :6] = 1
            k = i
            while k > 0:
                mask[i, dof_of_body[k]] = 1
                k = parent[k]
        self.mask = t(mask)
        # body carrying each rotational dof (3..nv-1): the root for 3..5
        body_of_dof = np.zeros(m.nv - 3, dtype=np.int64)
        for i in range(1, nb):
            body_of_dof[dof_of_body[i] - 3] = i
        self.body_of_dof = torch.tensor(body_of_dof, device=device)
        self.com_index = self.names.index(com_body) if com_body is not None else None
        # actuators, in joint order: force = kp (ctrl - q) - kd qd clamped to [lo, hi]
        kp, kd, lo, hi = (np.full(self.nj, np.nan) for _ in range(4))
        for a in range(m.nu):
            if (m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
                    or m.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE
                    or m.actuator_gear[a][0] != 1 or np.abs(m.actuator_gear[a][1:]).max() > 0 or m.actuator_ctrllimited[a]):  # fmt: skip
                raise ValueError(f"actuator {m.actuator(a).name}: not an unlimited-ctrl joint position actuator")
            g, bp = m.actuator_gainprm[a], m.actuator_biasprm[a]
            if bp[0] != 0 or bp[1] != -g[0]:
                raise ValueError(f"actuator {m.actuator(a).name}: bias is not -kp q - kd qd")
            k = m.jnt_dofadr[m.actuator_trnid[a, 0]] - 6
            if not np.isnan(kp[k]):
                raise ValueError("two actuators on one joint")
            kp[k], kd[k] = g[0], -bp[2]
            lo[k], hi[k] = (m.actuator_forcerange[a] if m.actuator_forcelimited[a] else (-np.inf, np.inf))
        if np.isnan(kp).any():
            raise ValueError("a joint without an actuator")
        self.kp, self.kd, self.f_lo, self.f_hi = t(kp), t(kd), t(lo), t(hi)
        self.total_mass = float(m.body_mass[bodies].sum())
        # tree structure for the Jacobian-free path: subtree indicator T[i, j] = body j is in the subtree of body i,
        # the body each dof moves with (root dofs: body 0), and MuJoCo's dof parents (the root's six dofs a chain)
        sub = np.zeros((nb, nb))
        for j in range(nb):
            k = j
            while k >= 0:
                sub[k, j] = 1
                k = parent[k]
        self.subtree = t(sub)
        dof_body = np.zeros(m.nv, dtype=np.int64)
        dof_body[6:] = [self.dof_of_body.index(d) for d in range(6, m.nv)]
        self.dof_body = torch.tensor(dof_body, device=device)
        self.dof_parent = [int(x) for x in m.dof_parentid]
        anc = np.zeros((m.nv, m.nv), dtype=bool)  # anc[k, j]: dof k is dof j or one of its ancestors
        for j in range(m.nv):
            k = j
            while k >= 0:
                anc[k, j] = True
                k = self.dof_parent[k]
        self.dof_anc = torch.tensor(anc, device=device)

    def index(self, names) -> list[int]:
        return [self.names.index(n) for n in names]

    # ------------------------------------------------------------------ kinematics
    def kinematics(self, p, R, q, com_offset=None):
        """Body origins x (B, nb, 3), rotations Rb (B, nb, 3, 3), world hinge axes ax (B, nb, 3) (zero for the root),
        centres of mass c (B, nb, 3) and world inertias Iw (B, nb, 3, 3)."""
        X, Rs, A = [p], [R], [torch.zeros_like(p)]
        for i in range(1, self.nb):
            par = self.parent[i]
            Rp = Rs[par]
            Ro = _mm(Rp, self.Roff[i])
            X.append(X[par] + _mv(Rp, self.pos[i]))
            Rs.append(_mm(Ro, _hinge_mat(self.axis[i], q[..., self.dof_of_body[i] - 6])))
            A.append(_mv(Ro, self.axis[i]))
        x, Rb, ax = torch.stack(X, -2), torch.stack(Rs, -3), torch.stack(A, -2)
        ipos = self.ipos.expand(x.shape)
        if com_offset is not None:
            ipos = ipos.clone()
            ipos[..., self.com_index, :] = ipos[..., self.com_index, :] + com_offset
        c = x + _mv(Rb, ipos)
        Iw = _mm(_mm(Rb, self.inertia), Rb.transpose(-1, -2))
        return {"x": x, "R": Rb, "ax": ax, "c": c, "Iw": Iw}

    def jacobians(self, kin):
        """Translational (at the body origins) and rotational Jacobians, each (B, nb, nv, 3)."""
        x, Rb, ax = kin["x"], kin["R"], kin["ax"]
        B = x.shape[:-2]
        # rotational dofs 3..nv-1: the root's local axes (columns of R) at the root origin, then the hinges
        A = torch.cat((Rb[..., 0, :, :].transpose(-1, -2), ax[..., 1:, :]), -2)  # (B, nv-3, 3): axis of dof 3+k
        O = x[..., self.body_of_dof, :]  # anchors
        lin = _cross(A.unsqueeze(-3), x.unsqueeze(-2) - O.unsqueeze(-3))  # (B, nb, nv-3, 3)
        eye = torch.eye(3, dtype=x.dtype, device=x.device).expand(*B, self.nb, 3, 3)
        m = self.mask[:, 3:, None]
        Jp = torch.cat((eye, lin * m), -2)
        Jr = torch.cat((torch.zeros_like(eye), A.unsqueeze(-3).expand(*B, self.nb, self.nv - 3, 3) * m), -2)
        return Jp, Jr

    @staticmethod
    def point_jacobian(Jp, Jr, r):
        """Translational Jacobian of the points at offsets r (B, nb, 3) from the body origins: Jp - [r]x Jr."""
        return Jp - _cross(r.unsqueeze(-2), Jr)

    def mass_matrix(self, kin, Jp, Jr):
        Jc = self.point_jacobian(Jp, Jr, kin["c"] - kin["x"])
        M = torch.einsum("...kdi,...kei->...de", Jc * self.mass[:, None, None], Jc)
        IJ = torch.einsum("...kij,...kej->...kei", kin["Iw"], Jr)
        M = M + torch.einsum("...kdi,...kei->...de", Jr, IJ)
        return M + torch.diag_embed(self.armature.expand(M.shape[:-1]))

    def bias(self, kin, Jp, Jr, nu, q_dot=None):
        """qfrc_bias: gravity plus Coriolis and centrifugal forces (Newton-Euler bias accelerations at qacc = 0)."""
        x, R, ax, c, Iw = kin["x"], kin["R"], kin["ax"], kin["c"], kin["Iw"]
        v0, wl, qd = nu[..., :3], nu[..., 3:6], nu[..., 6:]
        W, AL, AC = [_mv(R[..., 0, :, :], wl)], [torch.zeros_like(v0)], [(-self.gravity).expand(v0.shape)]
        for i in range(1, self.nb):
            par = self.parent[i]
            r = x[..., i, :] - x[..., par, :]
            wa = ax[..., i, :] * qd[..., self.dof_of_body[i] - 6 : self.dof_of_body[i] - 5]
            wp, alp = W[par], AL[par]
            W.append(wp + wa)
            AL.append(alp + _cross(wp, wa))
            AC.append(AC[par] + _cross(alp, r) + _cross(wp, _cross(wp, r)))
        w, al, acc = torch.stack(W, -2), torch.stack(AL, -2), torch.stack(AC, -2)
        rc = c - x
        acc_c = acc + _cross(al, rc) + _cross(w, _cross(w, rc))
        f = acc_c * self.mass[:, None]
        n = _mv(Iw, al) + _cross(w, _mv(Iw, w))
        Jc = self.point_jacobian(Jp, Jr, rc)
        return torch.einsum("...kdi,...ki->...d", Jc, f) + torch.einsum("...kdi,...ki->...d", Jr, n)

    def actuator(self, q, qd, target):
        """Actuator generalized force on the joints (B, nj) and its velocity derivative's magnitude kd where the force is
        not clamped (MuJoCo's implicit integrators drop clamped actuators from the derivative)."""
        f = self.kp * (target - q) - self.kd * qd
        fc = torch.minimum(torch.maximum(f, self.f_lo), self.f_hi)
        free = (f > self.f_lo) & (f < self.f_hi)
        return fc, torch.where(free, self.kd, torch.zeros_like(self.kd))

    def wrench_force(self, Jp, Jr, links, wrench):
        """Generalized force of world-frame wrenches (B, L, 6) = (force, moment about the link origin) on ``links``."""
        return (torch.einsum("...ldi,...li->...d", Jp[..., links, :, :], wrench[..., :3])
                + torch.einsum("...ldi,...li->...d", Jr[..., links, :, :], wrench[..., 3:]))  # fmt: skip

    # ------------------------------------------------------------------ Jacobian-free (tree) path
    def motion_axes(self, kin):
        """Each dof's spatial motion axis about the root origin p, world axes: angular part (B, nv, 3) and the linear
        velocity of the point at p (B, nv, 3)."""
        x, Rb, ax = kin["x"], kin["R"], kin["ax"]
        p = x[..., 0, :]
        R = Rb[..., 0, :, :]
        B = p.shape[:-1]
        eye = torch.eye(3, dtype=p.dtype, device=p.device).expand(*B, 3, 3)
        ang = torch.cat((torch.zeros_like(eye), R.transpose(-1, -2), ax[..., self.dof_body[6:], :]), -2)
        o = x[..., self.dof_body[6:], :]
        lin = torch.cat((eye, torch.zeros_like(eye), _cross(ax[..., self.dof_body[6:], :], p.unsqueeze(-2) - o)), -2)
        return ang, lin

    def velocities(self, kin, nu):
        """World angular velocities and body-origin linear velocities of every body (B, nb, 3) each, by recursion."""
        x, R, ax = kin["x"], kin["R"], kin["ax"]
        qd = nu[..., 6:]
        W, V = [_mv(R[..., 0, :, :], nu[..., 3:6])], [nu[..., :3]]
        for i in range(1, self.nb):
            par = self.parent[i]
            V.append(V[par] + _cross(W[par], x[..., i, :] - x[..., par, :]))
            W.append(W[par] + ax[..., i, :] * qd[..., self.dof_of_body[i] - 6 : self.dof_of_body[i] - 5])
        return torch.stack(W, -2), torch.stack(V, -2)

    def mass_matrix_tree(self, kin, S=None):
        """Composite-rigid-body mass matrix: subtree inertias about the root origin, M[k, j] = S_k . (I_sub(j) S_j) for
        k an ancestor of j (or j), copied to the transposed entries, plus the armature."""
        x, c, Iw = kin["x"], kin["c"], kin["Iw"]
        p = x[..., 0, :]
        r = c - p.unsqueeze(-2)
        m = self.mass[:, None]
        s = m * r  # first moments about p
        rr = (r * r).sum(-1)[..., None, None] * torch.eye(3, dtype=r.dtype, device=r.device) - r.unsqueeze(-1) * r.unsqueeze(-2)
        Ip = Iw + self.mass[:, None, None] * rr
        T = self.subtree
        mc = T @ self.mass  # (nb,)
        sc = torch.einsum("ij,...jk->...ik", T, s)
        Ic = torch.einsum("ij,...jkl->...ikl", T, Ip)
        ang, lin = S if S is not None else self.motion_axes(kin)
        b = self.dof_body
        md, sd, Id = mc[b], sc[..., b, :], Ic[..., b, :, :]
        L = _mv(Id, ang) + _cross(sd, lin)  # angular momentum about p of unit motion of dof j's subtree
        P = md[:, None] * lin - _cross(sd, ang)
        Mfull = torch.einsum("...ki,...ji->...kj", ang, L) + torch.einsum("...ki,...ji->...kj", lin, P)
        U = torch.where(self.dof_anc, Mfull, torch.zeros_like(Mfull))
        M = U + U.transpose(-1, -2) - torch.diag_embed(torch.diagonal(U, dim1=-2, dim2=-1))
        return M + torch.diag_embed(self.armature.expand(M.shape[:-1]))

    def bias_tree(self, kin, nu, links=None, wrench=None, S=None):
        """qfrc_bias - J^T wrench without Jacobians: the Newton-Euler body wrenches (as ``bias``) minus the external link
        wrenches, summed over subtrees about the root origin and projected on each dof's motion axis."""
        x, R, ax, c, Iw = kin["x"], kin["R"], kin["ax"], kin["c"], kin["Iw"]
        p = x[..., 0, :]
        v0, wl, qd = nu[..., :3], nu[..., 3:6], nu[..., 6:]
        W, AL, AC = [_mv(R[..., 0, :, :], wl)], [torch.zeros_like(v0)], [(-self.gravity).expand(v0.shape)]
        for i in range(1, self.nb):
            par = self.parent[i]
            r = x[..., i, :] - x[..., par, :]
            wa = ax[..., i, :] * qd[..., self.dof_of_body[i] - 6 : self.dof_of_body[i] - 5]
            wp, alp = W[par], AL[par]
            W.append(wp + wa)
            AL.append(alp + _cross(wp, wa))
            AC.append(AC[par] + _cross(alp, r) + _cross(wp, _cross(wp, r)))
        w, al, acc = torch.stack(W, -2), torch.stack(AL, -2), torch.stack(AC, -2)
        rc = c - x
        f = (acc + _cross(al, rc) + _cross(w, _cross(w, rc))) * self.mass[:, None]
        n = _mv(Iw, al) + _cross(w, _mv(Iw, w)) + _cross(c - p.unsqueeze(-2), f)  # about p
        if wrench is not None:
            Fe = torch.zeros_like(f).index_add(-2, torch.as_tensor(links, device=f.device), wrench[..., :3])
            Ne = torch.zeros_like(f).index_add(-2, torch.as_tensor(links, device=f.device),
                                               wrench[..., 3:] + _cross(x[..., links, :] - p.unsqueeze(-2), wrench[..., :3]))  # fmt: skip
            f, n = f - Fe, n - Ne
        Fs = torch.einsum("ij,...jk->...ik", self.subtree, f)[..., self.dof_body, :]
        Ns = torch.einsum("ij,...jk->...ik", self.subtree, n)[..., self.dof_body, :]
        ang, lin = S if S is not None else self.motion_axes(kin)
        return (ang * Ns).sum(-1) + (lin * Fs).sum(-1)

    def ltdl_solve(self, A, b):
        """Solve A x = b for A with the tree's sparsity (MuJoCo's mj_factorM / mj_solveM, no fill-in), unrolled over
        the dofs: per-sample scalar operations only, which torch.compile fuses."""
        nv, par = self.nv, self.dof_parent
        E = {}
        for j in range(nv):
            k = j
            while k >= 0:
                E[(j, k)] = A[..., j, k]
                k = par[k]
        for k in range(nv - 1, -1, -1):
            i = par[k]
            while i >= 0:
                tmp = E[(k, i)] / E[(k, k)]
                j = i
                while j >= 0:
                    E[(i, j)] = E[(i, j)] - tmp * E[(k, j)]
                    j = par[j]
                E[(k, i)] = tmp
                i = par[i]
        x = [b[..., k] for k in range(nv)]
        for k in range(nv - 1, -1, -1):  # x <- L^-T ... (MuJoCo's order: L^T D L with L unit lower in this storage)
            i = par[k]
            while i >= 0:
                x[i] = x[i] - E[(k, i)] * x[k]
                i = par[i]
        x = [x[k] / E[(k, k)] for k in range(nv)]
        for k in range(nv):
            i = par[k]
            while i >= 0:
                x[k] = x[k] - E[(k, i)] * x[i]
                i = par[i]
        return torch.stack(x, -1)

    def solve(self, A, b, method):
        """x with A x = b for symmetric positive definite A (B, nv, nv): "cholesky" (cuSOLVER / MAGMA batched),
        "lu" (torch.linalg.solve), "ltdl" (tree-sparse, unrolled per entry) or "columns" (dense Cholesky as a loop over
        the columns in elementwise tensor operations: no TF32, a few hundred graph nodes)."""
        if method == "cholesky":
            L, _ = torch.linalg.cholesky_ex(A)
            return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        if method == "lu":
            return torch.linalg.solve(A, b)
        if method == "ltdl":
            return self.ltdl_solve(A, b)
        if method == "columns":
            return cholesky_columns_solve(A, b)
        raise ValueError(method)

    def substep_tree(self, p, R, q, nu, target, dt, tau_ext=None, links=None, wrench=None, com_offset=None,
                     solve="cholesky"):  # fmt: skip
        """``substep`` on the Jacobian-free path (composite-rigid-body mass matrix, Newton-Euler bias with the link
        wrenches folded in); ``solve``: "cholesky" (batched dense) or "ltdl" (tree-sparse, unrolled)."""
        kin = self.kinematics(p, R, q, com_offset)
        S = self.motion_axes(kin)
        M = self.mass_matrix_tree(kin, S)
        hb = self.bias_tree(kin, nu, links, wrench, S)
        qd = nu[..., 6:]
        fa, kd = self.actuator(q, qd, target)
        rhs = torch.cat((torch.zeros_like(nu[..., :6]), fa - self.damping[6:] * qd), -1) - hb
        if tau_ext is not None:
            rhs = rhs + tau_ext
        diag = torch.cat((self.damping[:6].expand(kd.shape[:-1] + (6,)), kd + self.damping[6:]), -1)
        A = M + dt * torch.diag_embed(diag)
        dnu = self.solve(A, dt * rhs, solve)
        nu1 = nu + dnu
        return self.integrate(p, R, q, nu1, dt) + (nu1,)

    # ------------------------------------------------------------------ integration timestep
    def substep(self, p, R, q, nu, target, dt, tau_ext=None, links=None, wrench=None, com_offset=None, solve="cholesky"):
        """One implicitfast integration timestep of ``dt`` from (p, R, q, nu) with the joint targets ``target`` held,
        plus the generalized force ``tau_ext`` (B, nv) and the link ``wrench`` (B, L, 6) on ``links`` mapped through the
        Jacobians of the state the integration timestep starts from. Returns (p1, R1, q1, nu1).

        "substep" in the names ``substep`` and ``substep_tree`` means one integration timestep of the integrator, of
        any ``dt``; with the simulator's 5 ms it is one physics substep of the simulator."""
        kin = self.kinematics(p, R, q, com_offset)
        Jp, Jr = self.jacobians(kin)
        M = self.mass_matrix(kin, Jp, Jr)
        h = self.bias(kin, Jp, Jr, nu)
        qd = nu[..., 6:]
        fa, kd = self.actuator(q, qd, target)
        rhs = torch.cat((torch.zeros_like(nu[..., :6]), fa - self.damping[6:] * qd), -1) - h
        if tau_ext is not None:
            rhs = rhs + tau_ext
        if wrench is not None:
            rhs = rhs + self.wrench_force(Jp, Jr, links, wrench)
        diag = torch.cat((self.damping[:6].expand(kd.shape[:-1] + (6,)), kd + self.damping[6:]), -1)
        A = M + dt * torch.diag_embed(diag)
        dnu = self.solve(A, dt * rhs, solve)
        nu1 = nu + dnu
        return self.integrate(p, R, q, nu1, dt) + (nu1,)

    @staticmethod
    def integrate(p, R, q, nu1, dt):
        """MuJoCo's mj_integratePos with the velocity at the end of the integration timestep: root position in the
        world frame, rotation by the local angular velocity, joints."""
        return p + dt * nu1[..., :3], _mm(R, so3_exp(dt * nu1[..., 3:6])), q + dt * nu1[..., 6:]


def cholesky_columns_solve(A, b):
    """Solve A x = b (A symmetric positive definite, (..., n, n)) with a Cholesky factor built column by column from
    elementwise products and sums, then forward and backward substitution, also by columns."""
    n = A.shape[-1]
    rows = torch.arange(n, device=A.device)
    cols = []  # columns of L, (..., n) each, zero above the diagonal
    for j in range(n):
        v = A[..., :, j]
        if j:
            Lp = torch.stack(cols, -1)  # (..., n, j)
            v = v - (Lp * Lp[..., j : j + 1, :]).sum(-1)
        d = torch.sqrt(v[..., j : j + 1])
        cols.append(torch.where(rows >= j, v / d, torch.zeros_like(v)))
    L = torch.stack(cols, -1)  # (..., n, n), lower triangular
    y = []
    for j in range(n):  # L y = b
        r = b[..., j]
        if j:
            r = r - (L[..., j, :j] * torch.stack(y, -1)).sum(-1)
        y.append(r / L[..., j, j])
    x = [None] * n
    for j in range(n - 1, -1, -1):  # L^T x = y
        r = y[j]
        if j < n - 1:
            r = r - (L[..., j + 1 :, j] * torch.stack(x[j + 1 :], -1)).sum(-1)
        x[j] = r / L[..., j, j]
    return torch.stack(x, -1)


# ---------------------------------------------------------------------- conversions
def from_trainer(S):
    """Trainer state tuple (p, R, v, w_world, q, qd) -> (p, R, q, nu) in MuJoCo's convention."""
    p, R, v, w, q, qd = S
    return p, R, q, torch.cat((v, _mv(R.transpose(-1, -2), w), qd), -1)


def to_trainer(p, R, q, nu):
    return p, R, nu[..., :3], _mv(R, nu[..., 3:6]), q, nu[..., 6:]


def from_mujoco(qpos, qvel):
    """MuJoCo qpos (B, 7 + nj) and qvel (B, 6 + nj) of a model whose only free joint is first -> (p, R, q, nu)."""
    from .rotations import quat_to_mat

    return qpos[..., :3], quat_to_mat(qpos[..., 3:7]), qpos[..., 7:], qvel
