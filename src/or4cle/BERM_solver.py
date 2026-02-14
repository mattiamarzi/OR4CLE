#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# modules/solvers/BERM_solver.py

"""
-----------------------------------------------------------------------------
Bayesian Erdős–Rényi Model (BERM) solvers and rolling evaluators.

Model
-----
For an undirected simple graph with N nodes, define M = N(N-1)/2 potential links.
Conditional on a global density parameter p in (0, 1), links are independent Bernoulli:

    a_ij | p  ~  Bernoulli(p),   for i < j.

Given a snapshot at time t, the sufficient statistic is the (possibly fractional)
upper-triangle link count

    L_t = sum_{i<j} a_ij^t,

where fractional counts correspond to a "soft adjacency" matrix.

Posterior-predictive link probabilities for time t+1 are constant across pairs:

    q_ij^{t+1} = E[p | L_t, N],  for all i < j.

Supported priors on p
---------------------
1) Beta prior (default family in rolling calibration):
       p ~ Beta(alpha, beta) with alpha > 0, beta > 0.
   Posterior is exact (also for fractional counts):
       p | L_t ~ Beta(alpha + L_t, beta + (M - L_t)).

2) Normal prior on the logit scale:
       u = logit(p),   u ~ Normal(mu0, sigma0^2),   sigma0 > 0.
   The posterior on u is log-concave but non-conjugate.

Inference modes (BFM-consistent)
--------------------------------
- mode="mc":
    * Beta prior: uses the exact posterior mean (deterministic) and reports the exact variance.
    * Normal prior: slice samples u from the exact log posterior, then p = sigmoid(u).
      Predictive mean is approximated by Monte Carlo.

- mode="gh" (default):
    * Beta prior: uses the exact posterior mean (deterministic).
    * Normal prior: Laplace approximation around MAP u_hat, then Gauss–Hermite quadrature
      to approximate E[sigmoid(U)] with U ~ Normal(u_hat, sigma2).

Rolling pipelines
-----------------
This module also provides a package-friendly rolling runner which requires:
  (i) a seed/calibration period used to build the initial prior sample of p-hats,
 (ii) an analysis period defining the t -> t+1 loop,
(iii) a snapshot loader returning (A, node_ids) for each label.

Two update strategies are supported:
- update="observed" (data-driven):
    rolling prior is fitted from observed p-hats over the historical window.

- update="counterfactual" (self-sustained):
    after a configurable seed_end, hard A_t is forbidden; training uses only the
    previous step's predictive probability projected onto the current active set.
    The rolling prior sample is updated with p_MAP at each step.

All rolling computations follow the manuscript convention: the posterior at time t is
computed on the node set observed at time t (V_t), while predictions are evaluated
on the node set observed at time t+1 (V_{t+1}).

Public API
----------
- Snapshot: minimal container for an adjacency and node ids
- Prior: container for supported prior families and empirical-Bayes fit constructors
- berm_predictive_probabilities(...) -> (p_pred, L_pred, info) or (Q, L_pred, info)
- solve_berm(A_t, prior, ...) -> (p_pred or Q, L_pred, info)
- BERMStepModel: single-snapshot posterior predictive
- BERMRollingModel: end-to-end rolling runner driven by a snapshot loader
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray


# =============================================================================
# Snapshot container
# =============================================================================
@dataclass(frozen=True)
class Snapshot:
    """
    Representation of a network snapshot.

    Parameters
    ----------
    A:
        Square adjacency matrix. Entries may be binary or "soft" in [0, 1].
        Only the strict upper triangle is used in likelihood computations.
    node_ids:
        One-dimensional array of node identifiers matching the ordering of A.
        If node_ids is None, indices 0..N-1 are used as identifiers.
    """
    A: Array
    node_ids: Optional[Array] = None

    def __post_init__(self) -> None:
        A = np.asarray(self.A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Snapshot.A must be a square matrix.")
        if self.node_ids is None:
            object.__setattr__(self, "node_ids", np.arange(A.shape[0], dtype=np.int64))
        else:
            ids = np.asarray(self.node_ids)
            if ids.ndim != 1 or ids.shape[0] != A.shape[0]:
                raise ValueError("Snapshot.node_ids must be 1D with length equal to A.shape[0].")
            object.__setattr__(self, "node_ids", ids.astype(np.int64, copy=False))


# =============================================================================
# Cached upper-triangle indices
# =============================================================================
@lru_cache(maxsize=128)
def _triu_indices_cached(n: int) -> Tuple[Array, Array]:
    """Return cached indices for the strict upper triangle of an n x n matrix."""
    iu = np.triu_indices(int(n), k=1)
    return iu[0].astype(np.int64), iu[1].astype(np.int64)


# =============================================================================
# Core numerical utilities
# =============================================================================
def links_from_adjacency(A: Array) -> float:
    """
    Compute L = sum_{i<j} A_ij for an undirected adjacency matrix.

    This function accepts binary or soft adjacencies; the diagonal is ignored.
    """
    M = np.asarray(A, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = int(M.shape[0])
    i_idx, j_idx = _triu_indices_cached(n)
    return float(np.sum(M[i_idx, j_idx]))


def _upper_triangle_vec(A: Array) -> Array:
    """Return the strict upper-triangle entries of a square matrix as a 1D vector."""
    M = np.asarray(A, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = int(M.shape[0])
    i_idx, j_idx = _triu_indices_cached(n)
    return np.asarray(M[i_idx, j_idx], dtype=np.float64).reshape(-1)


def _top_L_prediction(q_vec: Array, L: int) -> Array:
    """Return a binary prediction vector selecting the L largest probabilities."""
    q = np.asarray(q_vec, dtype=np.float64).reshape(-1)
    m = int(q.size)
    if m == 0:
        return np.zeros(0, dtype=np.float64)
    L = int(L)
    if L <= 0:
        return np.zeros(m, dtype=np.float64)
    if L >= m:
        return np.ones(m, dtype=np.float64)

    # Argpartition provides an O(m) selection; ties are broken implicitly by the
    # underlying algorithm and remain deterministic for a fixed input array.
    idx = np.argpartition(-q, L - 1)[:L]
    pred = np.zeros(m, dtype=np.float64)
    pred[idx] = 1.0
    return pred


def _M_pairs(n: int) -> float:
    """Return M = N(N-1)/2 as float."""
    n = int(n)
    return 0.5 * n * (n - 1)


def _validate_n_nodes(n: int) -> int:
    """Validate a non-negative integer number of nodes."""
    n_int = int(n)
    if n_int < 0:
        raise ValueError("N must be non-negative.")
    return n_int


def _validate_finite(x: float, name: str) -> float:
    """Validate a finite scalar."""
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite.")
    return float(x)


def _sigmoid(u: float) -> float:
    """Numerically stable sigmoid."""
    if u >= 0.0:
        z = math.exp(-u)
        return 1.0 / (1.0 + z)
    z = math.exp(u)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-12) -> float:
    """Numerically stable logit with clipping."""
    pp = float(np.clip(p, eps, 1.0 - eps))
    return math.log(pp) - math.log1p(-pp)


def _unique_int_ids(ids: Array) -> Array:
    """Return a sorted array of unique int64 identifiers."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return ids
    return np.unique(ids)


def _intersection_size(a: Array, b: Array) -> int:
    """Compute |a ∩ b| for int64 arrays."""
    a = _unique_int_ids(a)
    b = _unique_int_ids(b)
    if a.size == 0 or b.size == 0:
        return 0
    return int(np.intersect1d(a, b, assume_unique=True).size)


def _submatrix_on_ids_sum_upper(
    snap: Snapshot,
    target_ids: Array,
) -> float:
    """
    Compute the upper-triangle sum of snap.A restricted to the node set target_ids.

    Nodes absent in snap are treated as isolated (contributing zero).
    """
    A = np.asarray(snap.A, dtype=np.float64)
    ids_src = np.asarray(snap.node_ids, dtype=np.int64)
    target_ids = _unique_int_ids(np.asarray(target_ids, dtype=np.int64))

    if target_ids.size < 2:
        return 0.0

    # Map source ids to indices
    pos = {int(b): i for i, b in enumerate(ids_src)}
    idx = [pos[int(b)] for b in target_ids if int(b) in pos]
    if len(idx) < 2:
        return 0.0

    idx_arr = np.asarray(idx, dtype=np.int64)
    A_sub = A[np.ix_(idx_arr, idx_arr)]
    return links_from_adjacency(A_sub)


# =============================================================================
# Prior container and empirical-Bayes fitting
# =============================================================================
@dataclass(frozen=True)
class Prior:
    """
    Prior on the ER density parameter p.

    family:
        "beta" or "normal"

    params:
        - beta:   {"alpha": float, "beta": float}
        - normal: {"mu0": float, "sigma0": float}   (on u = logit(p))
    """
    family: Literal["beta", "normal"]
    params: Dict[str, float] = field(default_factory=dict)

    # ------------------------- constructors -------------------------
    @staticmethod
    def beta(alpha: float, beta: float) -> "Prior":
        """Construct a Beta(alpha, beta) prior."""
        return Prior(family="beta", params={"alpha": float(alpha), "beta": float(beta)})

    @staticmethod
    def normal(mu0: float, sigma0: float) -> "Prior":
        """Construct a Normal(mu0, sigma0^2) prior on u = logit(p)."""
        return Prior(family="normal", params={"mu0": float(mu0), "sigma0": float(sigma0)})

    # ------------------------- validation -------------------------
    def validate(self) -> None:
        """Validate prior parameters."""
        if self.family == "beta":
            a = float(self.params.get("alpha", np.nan))
            b = float(self.params.get("beta", np.nan))
            if (not np.isfinite(a)) or a <= 0.0:
                raise ValueError("Beta prior requires alpha > 0 and finite.")
            if (not np.isfinite(b)) or b <= 0.0:
                raise ValueError("Beta prior requires beta > 0 and finite.")
        elif self.family == "normal":
            mu0 = float(self.params.get("mu0", np.nan))
            sigma0 = float(self.params.get("sigma0", np.nan))
            if not np.isfinite(mu0):
                raise ValueError("Normal prior requires finite mu0 (logit scale).")
            if (not np.isfinite(sigma0)) or sigma0 <= 0.0:
                raise ValueError("Normal prior requires sigma0 > 0 and finite (logit scale).")
        else:
            raise ValueError(f"Unsupported prior family: {self.family}")

    # ------------------------- EB fits -------------------------
    @staticmethod
    def beta_from_samples(
        p_samples: Array,
        *,
        eps: float = 1e-9,
        var_floor: float = 1e-12,
        fallback: Optional["Prior"] = None,
    ) -> "Prior":
        """
        Construct a Beta prior by method-of-moments from a sample of probabilities.

        The sample is clipped to (eps, 1-eps). If the sample is too small or
        degenerate, the fallback prior is returned.
        """
        if fallback is None:
            fallback = Prior.beta(1.0, 1.0)

        x = np.asarray(p_samples, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return fallback

        x = np.clip(x, eps, 1.0 - eps)
        m = float(np.mean(x))
        v = float(np.var(x, ddof=0))
        v = max(v, var_floor)
        # ensure feasibility: v <= m(1-m)
        v = min(v, 0.9999 * m * (1.0 - m))

        tpar = (m * (1.0 - m) / v) - 1.0
        alpha = max(1e-12, m * tpar)
        beta = max(1e-12, (1.0 - m) * tpar)
        return Prior.beta(alpha, beta)

    @staticmethod
    def normal_from_samples(
        p_samples: Array,
        *,
        eps: float = 1e-9,
        sigma_floor: float = 1e-3,
        fallback: Optional["Prior"] = None,
    ) -> "Prior":
        """
        Construct a Normal prior on u = logit(p) from a sample of probabilities.

        The sample is clipped to (eps, 1-eps). If the sample is too small or
        degenerate, the fallback prior is returned.
        """
        if fallback is None:
            fallback = Prior.normal(0.0, 2.0)

        x = np.asarray(p_samples, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return fallback

        x = np.clip(x, eps, 1.0 - eps)
        u = np.log(x) - np.log1p(-x)
        mu0 = float(np.mean(u))
        sigma0 = float(np.std(u, ddof=0))
        sigma0 = max(sigma0, sigma_floor)
        return Prior.normal(mu0, sigma0)


# =============================================================================
# Posterior on u = logit(p) for Normal prior
# =============================================================================
@dataclass
class _PosteriorLogitNormal:
    """
    Log posterior for u = logit(p) under a Binomial likelihood with count L and M trials.

    Up to an additive constant:
        g(u) = L*u - M*log(1+exp(u)) - (u - mu0)^2 / (2*sigma0^2)
    """
    L: float
    M: float
    mu0: float
    sigma0: float

    def g(self, u: float) -> float:
        """Log posterior up to a constant."""
        if u > 0.0:
            softplus = u + math.log1p(math.exp(-u))
        else:
            softplus = math.log1p(math.exp(u))
        quad = 0.5 * ((u - self.mu0) / self.sigma0) ** 2
        return (self.L * u) - (self.M * softplus) - quad

    def d1d2(self, u: float) -> Tuple[float, float]:
        """First and second derivatives of g(u)."""
        p = _sigmoid(u)
        invv = 1.0 / (self.sigma0 * self.sigma0)
        d1 = (self.L - self.M * p) - (u - self.mu0) * invv
        d2 = -(self.M * p * (1.0 - p)) - invv
        return float(d1), float(d2)


def _laplace_map_u(
    post: _PosteriorLogitNormal,
    *,
    u0: float,
    max_iter: int,
    tol: float,
    hard_bounds_u: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Compute MAP u_hat and sigma2 = -1/g''(u_hat) using damped Newton ascent.
    """
    u_lo, u_hi = float(hard_bounds_u[0]), float(hard_bounds_u[1])
    if not (u_lo < u_hi):
        raise ValueError("Invalid hard bounds for u.")

    u = float(np.clip(u0, u_lo + 1e-12, u_hi - 1e-12))
    g_u = post.g(u)

    for _ in range(int(max_iter)):
        d1, d2 = post.d1d2(u)
        if (not np.isfinite(d1)) or (not np.isfinite(d2)) or d2 >= 0.0:
            d2 = -1e-8

        step = -d1 / d2

        alpha = 1.0
        accepted = False
        for _ls in range(60):
            u_try = float(np.clip(u + alpha * step, u_lo + 1e-12, u_hi - 1e-12))
            g_try = post.g(u_try)
            if np.isfinite(g_try) and (g_try >= g_u):
                u = u_try
                g_u = g_try
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            u = float(np.clip(u + 1e-3 * step, u_lo + 1e-12, u_hi - 1e-12))
            g_u = post.g(u)

        if abs(alpha * step) < float(tol):
            break

    _, d2 = post.d1d2(u)
    sigma2 = (-1.0 / d2) if (np.isfinite(d2) and d2 < 0.0) else 1e-6
    return float(u), float(sigma2)


def _logit_normal_mean_via_gh(u_mu: float, u_sigma: float, nodes: int = 40) -> float:
    """
    Compute E[sigmoid(U)] for U ~ Normal(u_mu, u_sigma^2) using Gauss–Hermite quadrature.
    """
    from numpy.polynomial.hermite import hermgauss

    xk, wk = hermgauss(int(nodes))
    xk = np.asarray(xk, dtype=np.float64)
    wk = np.asarray(wk, dtype=np.float64)

    wk_norm = wk / float(np.sum(wk))
    sqrt2 = math.sqrt(2.0)

    acc = 0.0
    for x, w in zip(xk, wk_norm):
        u = float(u_mu + sqrt2 * u_sigma * float(x))
        acc += float(w) * _sigmoid(u)
    return float(acc)


def _slice_sample_u(
    post: _PosteriorLogitNormal,
    u_init: float,
    n_samples: int,
    *,
    burnin: int = 400,
    width: float = 1.0,
    hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """
    Univariate slice sampler for a log-concave posterior in u.

    This sampler targets the exact posterior g(u) and enforces hard bounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    u_lo, u_hi = float(hard_bounds_u[0]), float(hard_bounds_u[1])
    if not (u_lo < u_hi):
        raise ValueError("Invalid hard bounds for u.")

    def g(u: float) -> float:
        if (u < u_lo) or (u > u_hi):
            return -np.inf
        return post.g(u)

    u = float(np.clip(u_init, u_lo + 1e-12, u_hi - 1e-12))
    gu = g(u)

    out = np.empty(int(n_samples), dtype=np.float64)
    kept = 0

    for it in range(int(n_samples) + int(burnin)):
        y = gu - rng.exponential(1.0)

        L = u - width * rng.random()
        R = L + width

        while (g(L) > y) and (L > u_lo):
            L -= width
        while (g(R) > y) and (R < u_hi):
            R += width

        while True:
            u_prop = float(rng.uniform(max(L, u_lo), min(R, u_hi)))
            gu_prop = g(u_prop)
            if np.isfinite(gu_prop) and (gu_prop > y):
                u = u_prop
                gu = gu_prop
                break
            if u_prop < u:
                L = u_prop
            else:
                R = u_prop

        if it >= int(burnin):
            out[kept] = u
            kept += 1

    return out


# =============================================================================
# Single-step predictive computation
# =============================================================================
def berm_predictive_probabilities(
    N: int,
    L_t: float,
    prior: Prior,
    *,
    mode: Literal["mc", "gh"] = "gh",
    # MC controls (Normal prior)
    mc_samples: int = 3000,
    mc_burnin: int = 600,
    mc_width: float = 1.0,
    mc_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
    rng: Optional[np.random.Generator] = None,
    # GH / Laplace controls (Normal prior)
    gh_nodes: int = 40,
    gh_max_iter: int = 80,
    gh_tol: float = 1e-12,
    gh_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
    # Output control
    return_Q: bool = True,
    # Numerics
    clip: float = 1e-9,
) -> Tuple[Union[Array, float], float, Dict[str, float]]:
    """
    Compute BERM posterior-predictive probabilities given N and L_t.

    Parameters
    ----------
    N:
        Number of nodes.
    L_t:
        Upper-triangle link count. Fractional values are allowed.
    prior:
        Prior on p (Beta or Normal on logit scale).
    mode:
        "mc" or "gh" (default).
    return_Q:
        If True, returns the full N x N predictive matrix with constant off-diagonal.
        If False, returns the scalar p_pred instead of Q.
    """
    prior.validate()
    N = _validate_n_nodes(N)
    L_t = _validate_finite(float(L_t), "L_t")

    M = float(_M_pairs(N))
    if M <= 0.0:
        p_pred = 0.0
        info = {"N": float(N), "M": float(M), "L_t": float(L_t), "p_pred": float(p_pred), "L_pred": 0.0}
        if return_Q:
            Q0 = np.zeros((N, N), dtype=np.float64)
            return Q0, 0.0, info
        return p_pred, 0.0, info

    if (L_t < 0.0) or (L_t > M + 1e-9):
        raise ValueError("L_t must satisfy 0 <= L_t <= M for the given N.")

    if rng is None:
        rng = np.random.default_rng()

    info: Dict[str, float] = {
        "N": float(N),
        "M": float(M),
        "L_t": float(L_t),
        "mode": 1.0 if mode == "mc" else 0.0,
    }

    # -------------------------------------------------------------------------
    # Beta prior (conjugate)
    # -------------------------------------------------------------------------
    if prior.family == "beta":
        alpha = float(prior.params["alpha"])
        beta = float(prior.params["beta"])
        alpha_post = alpha + L_t
        beta_post = beta + (M - L_t)
        if alpha_post <= 0.0 or beta_post <= 0.0:
            raise ValueError("Posterior Beta parameters must be positive.")

        # Exact posterior mean and variance
        denom = alpha_post + beta_post
        p_mean = float(alpha_post / denom)
        p_var = float((alpha_post * beta_post) / (denom * denom * (denom + 1.0)))

        # Posterior mode when defined; otherwise the mean is used as a stable proxy
        if (alpha_post > 1.0) and (beta_post > 1.0):
            p_map = float((alpha_post - 1.0) / (alpha_post + beta_post - 2.0))
        else:
            p_map = p_mean

        p_pred = p_mean
        info.update({
            "alpha_post": float(alpha_post),
            "beta_post": float(beta_post),
            "p_mean": float(p_mean),
            "p_var": float(p_var),
            "p_map": float(p_map),
            "conjugate": 1.0,
        })

    # -------------------------------------------------------------------------
    # Normal prior on logit(p) (non-conjugate)
    # -------------------------------------------------------------------------
    elif prior.family == "normal":
        mu0 = float(prior.params["mu0"])
        sigma0 = float(prior.params["sigma0"])
        post = _PosteriorLogitNormal(L=float(L_t), M=float(M), mu0=mu0, sigma0=sigma0)

        if mode == "mc":
            u_samps = _slice_sample_u(
                post,
                u_init=float(mu0),
                n_samples=int(mc_samples),
                burnin=int(mc_burnin),
                width=float(mc_width),
                hard_bounds_u=gh_hard_bounds_u,
                rng=rng,
            )
            p_samps = np.empty_like(u_samps)
            for i in range(u_samps.size):
                p_samps[i] = _sigmoid(float(u_samps[i]))

            p_pred = float(np.mean(p_samps))
            u_mean = float(np.mean(u_samps))
            u_std = float(np.std(u_samps, ddof=1)) if u_samps.size > 1 else 0.0

            # MAP estimate from Laplace Newton (cheap in 1D), used for counterfactual updates
            u_map, _sigma2 = _laplace_map_u(
                post, u0=float(u_mean), max_iter=int(gh_max_iter), tol=float(gh_tol), hard_bounds_u=gh_hard_bounds_u
            )
            p_map = float(_sigmoid(u_map))

            info.update({
                "u_mean": float(u_mean),
                "u_std": float(u_std),
                "p_mc_std": float(np.std(p_samps, ddof=1)) if p_samps.size > 1 else 0.0,
                "mc_samples": float(u_samps.size),
                "mc_burnin": float(mc_burnin),
                "mc_width": float(mc_width),
                "mc_u_lo": float(mc_hard_bounds_u[0]),
                "mc_u_hi": float(mc_hard_bounds_u[1]),
                "u_map": float(u_map),
                "p_map": float(p_map),
                "conjugate": 0.0,
            })

        elif mode == "gh":
            u_map, sigma2 = _laplace_map_u(
                post,
                u0=float(mu0),
                max_iter=int(gh_max_iter),
                tol=float(gh_tol),
                hard_bounds_u=gh_hard_bounds_u,
            )
            u_sigma = float(math.sqrt(max(sigma2, 1e-16)))
            p_pred = _logit_normal_mean_via_gh(u_mu=u_map, u_sigma=u_sigma, nodes=int(gh_nodes))
            p_map = float(_sigmoid(u_map))
            info.update({
                "u_map": float(u_map),
                "sigma2_laplace": float(sigma2),
                "u_sigma": float(u_sigma),
                "gh_nodes": float(gh_nodes),
                "gh_u_lo": float(gh_hard_bounds_u[0]),
                "gh_u_hi": float(gh_hard_bounds_u[1]),
                "p_map": float(p_map),
                "conjugate": 0.0,
            })

        else:
            raise ValueError("mode must be 'mc' or 'gh'.")

    else:
        raise ValueError(f"Unsupported prior family: {prior.family}")

    if float(clip) > 0.0:
        p_pred = float(np.clip(p_pred, float(clip), 1.0 - float(clip)))

    L_pred = float(p_pred * M)
    info["p_pred"] = float(p_pred)
    info["L_pred"] = float(L_pred)

    if not return_Q:
        return float(p_pred), float(L_pred), info

    Q = np.full((N, N), p_pred, dtype=np.float64)
    np.fill_diagonal(Q, 0.0)
    return Q, float(L_pred), info


def solve_berm(
    A_t: Array,
    prior: Prior,
    *,
    mode: Literal["mc", "gh"] = "gh",
    return_Q: bool = True,
    **kwargs: Any,
) -> Tuple[Union[Array, float], float, Dict[str, float]]:
    """
    Convenience wrapper: compute (N, L_t) from A_t and return the BERM prediction.
    """
    A = np.asarray(A_t, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A_t must be a square matrix.")
    N = int(A.shape[0])
    L_t = links_from_adjacency(A)
    return berm_predictive_probabilities(N=N, L_t=L_t, prior=prior, mode=mode, return_Q=return_Q, **kwargs)


# =============================================================================
# Step model container
# =============================================================================
class BERMStepModel:
    """
    Single-step BERM posterior predictive model.

    This container stores N, L_t, and the prior, and computes p_pred (and Q if requested).
    """

    def __init__(self, N: int, L_t: float, prior: Prior) -> None:
        self.N = _validate_n_nodes(N)
        self.M = float(_M_pairs(self.N))
        self.L_t = float(_validate_finite(float(L_t), "L_t"))
        if self.M > 0.0 and ((self.L_t < 0.0) or (self.L_t > self.M + 1e-9)):
            raise ValueError("L_t must satisfy 0 <= L_t <= M for the given N.")
        prior.validate()
        self.prior = prior
        self._info: Optional[Dict[str, float]] = None
        self._p_pred: Optional[float] = None

    def fit(
        self,
        *,
        mode: Literal["mc", "gh"] = "gh",
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute the posterior predictive mean p_pred and store diagnostics."""
        p_pred, L_pred, info = berm_predictive_probabilities(
            N=self.N, L_t=self.L_t, prior=self.prior, mode=mode, return_Q=False, rng=rng, **kwargs
        )
        self._p_pred = float(p_pred)
        self._info = dict(info)
        _ = L_pred
        return dict(info)

    def predict(self, *, return_Q: bool = True, clip: float = 1e-9) -> Tuple[Union[Array, float], float, Dict[str, float]]:
        """Return the predictive quantity (Q or p_pred) and corresponding expected links L_pred."""
        if self._p_pred is None or self._info is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        p_pred = float(self._p_pred)
        if float(clip) > 0.0:
            p_pred = float(np.clip(p_pred, float(clip), 1.0 - float(clip)))
        L_pred = float(p_pred * self.M)

        info = dict(self._info)
        info["p_pred"] = float(p_pred)
        info["L_pred"] = float(L_pred)

        if not return_Q:
            return float(p_pred), float(L_pred), info

        Q = np.full((self.N, self.N), p_pred, dtype=np.float64)
        np.fill_diagonal(Q, 0.0)
        return Q, float(L_pred), info


# =============================================================================
# Label parsing and generation (package-friendly)
# =============================================================================
Agg = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]
PeriodLike = Union[
    Tuple[Any, Any],
    List[str],
    Tuple[List[str],],
]


def _parse_week_label(x: Any) -> Tuple[int, int]:
    """Parse a weekly label into (iso_year, iso_week)."""
    if isinstance(x, tuple) and len(x) == 2:
        y, w = int(x[0]), int(x[1])
        return y, w
    if isinstance(x, (datetime, date)):
        iso = x.isocalendar()
        return int(iso[0]), int(iso[1])
    if isinstance(x, str):
        s = x.strip()
        # accepted: "YYYY-Www", "YYYYWww", "YYYY-Www-1", "YYYY-Www-D"
        if "-W" in s:
            left, right = s.split("-W", 1)
            y = int(left)
            w = int(right[:2])
            return y, w
        if "W" in s:
            left, right = s.split("W", 1)
            y = int(left[:4])
            w = int(right[:2])
            return y, w
    raise ValueError(f"Cannot parse weekly label from: {x!r}")


def _week_to_monday(y: int, w: int) -> date:
    """Return the Monday date of ISO year-week (y, w)."""
    return date.fromisocalendar(int(y), int(w), 1)


def _iso_weeks_in_year(y: int) -> int:
    """Return the number of ISO weeks in year *y*.

    The ISO week calendar admits either 52 or 53 weeks depending on the weekday
    pattern of the year. This helper is used to robustly construct rolling
    calibration windows by clamping week numbers when shifting years.
    """
    y = int(y)
    try:
        # If week 53 exists, this call succeeds.
        _ = date.fromisocalendar(y, 53, 1)
        return 53
    except Exception:
        return 52


def _format_week_label(d: date) -> str:
    """Format a date into an ISO week label 'YYYY-Www'."""
    iso = d.isocalendar()
    return f"{int(iso[0])}-W{int(iso[1]):02d}"


def _parse_day_label(x: Any) -> date:
    """Parse a daily label into a date."""
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        return datetime.strptime(x.strip(), "%Y-%m-%d").date()
    raise ValueError(f"Cannot parse daily label from: {x!r}")


def _parse_month_label(x: Any) -> Tuple[int, int]:
    """Parse a monthly label into (year, month)."""
    if isinstance(x, tuple) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, (datetime, date)):
        return int(x.year), int(x.month)
    if isinstance(x, str):
        s = x.strip()
        y, m = s.split("-", 1)
        return int(y), int(m)
    raise ValueError(f"Cannot parse monthly label from: {x!r}")


def _parse_quarter_label(x: Any) -> Tuple[int, int]:
    """Parse a quarterly label into (year, quarter)."""
    if isinstance(x, tuple) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, (datetime, date)):
        q = (int(x.month) - 1) // 3 + 1
        return int(x.year), int(q)
    if isinstance(x, str):
        s = x.strip()
        y, q = s.split("-Q", 1)
        return int(y), int(q)
    raise ValueError(f"Cannot parse quarterly label from: {x!r}")


def _parse_year_label(x: Any) -> int:
    """Parse a yearly label into an integer year."""
    if isinstance(x, int):
        return int(x)
    if isinstance(x, (datetime, date)):
        return int(x.year)
    if isinstance(x, str):
        return int(x.strip())
    raise ValueError(f"Cannot parse yearly label from: {x!r}")


def labels_in_period(agg: Agg, start: Any, end: Any) -> List[str]:
    """
    Generate canonical labels from start to end (inclusive) for the given aggregation.

    The start/end inputs may be strings in canonical form, tuples (year, ..),
    or datetime/date objects.
    """
    if agg == "daily":
        d0 = _parse_day_label(start)
        d1 = _parse_day_label(end)
        if d0 > d1:
            raise ValueError("start must be <= end.")
        out = []
        d = d0
        while d <= d1:
            out.append(d.isoformat())
            d = d + timedelta(days=1)
        return out

    if agg == "weekly":
        y0, w0 = _parse_week_label(start)
        y1, w1 = _parse_week_label(end)
        d0 = _week_to_monday(y0, w0)
        d1 = _week_to_monday(y1, w1)
        if d0 > d1:
            raise ValueError("start must be <= end.")
        out = []
        d = d0
        while d <= d1:
            out.append(_format_week_label(d))
            d = d + timedelta(days=7)
        return out

    if agg == "monthly":
        y0, m0 = _parse_month_label(start)
        y1, m1 = _parse_month_label(end)
        if (y0, m0) > (y1, m1):
            raise ValueError("start must be <= end.")
        out = []
        y, m = int(y0), int(m0)
        while (y, m) <= (int(y1), int(m1)):
            out.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                y += 1
                m = 1
        return out

    if agg == "quarterly":
        y0, q0 = _parse_quarter_label(start)
        y1, q1 = _parse_quarter_label(end)
        if (y0, q0) > (y1, q1):
            raise ValueError("start must be <= end.")
        out = []
        y, q = int(y0), int(q0)
        while (y, q) <= (int(y1), int(q1)):
            out.append(f"{y:04d}-Q{q}")
            q += 1
            if q == 5:
                y += 1
                q = 1
        return out

    if agg == "yearly":
        y0 = _parse_year_label(start)
        y1 = _parse_year_label(end)
        if y0 > y1:
            raise ValueError("start must be <= end.")
        return [f"{y:04d}" for y in range(int(y0), int(y1) + 1)]

    raise ValueError(f"Unsupported agg: {agg!r}")


def _labels_cal_window(agg: Agg, end_label: str, global_start: str, years: Optional[int]) -> List[str]:
    """
    Build a calibration label set from max(global_start, end_label - years) to end_label (inclusive).
    """
    if years is None:
        return labels_in_period(agg, global_start, end_label)

    if agg == "weekly":
        y, w = _parse_week_label(end_label)
        y0 = int(y) - int(years)
        w0 = int(w)
        # If the shifted year does not admit ISO week 53, clamp to week 52.
        if w0 > _iso_weeks_in_year(y0):
            w0 = _iso_weeks_in_year(y0)
        start_bound = f"{y0}-W{w0:02d}"
    elif agg == "monthly":
        y, m = _parse_month_label(end_label)
        start_bound = f"{y - int(years)}-{m:02d}"
    elif agg == "quarterly":
        y, q = _parse_quarter_label(end_label)
        start_bound = f"{y - int(years)}-Q{q}"
    elif agg == "yearly":
        y = _parse_year_label(end_label)
        start_bound = f"{y - int(years)}"
    else:
        # for daily, use a conservative approximation (365*years)
        d1 = _parse_day_label(end_label)
        d0 = d1 - timedelta(days=365 * int(years))
        start_bound = d0.isoformat()

    # choose the later start in chronological order; canonical labels compare lexicographically for these formats
    start_eff = max(str(global_start), str(start_bound))
    return labels_in_period(agg, start_eff, end_label)


# =============================================================================
# Rolling results container
# =============================================================================
@dataclass
class BERMRollingResults:
    """
    Rolling output container.

    The 'steps' field stores per-step diagnostics. Each element corresponds to
    one transition t -> t+1.
    """
    steps: List[Dict[str, Any]]
    prior_family: str
    mode: str
    update: str
    agg: str
    seed_period: Tuple[str, str]
    analysis_period: Tuple[str, str]


# =============================================================================
# Rolling runner
# =============================================================================
class BERMRollingModel:
    """
    End-to-end rolling BERM runner driven by a snapshot loader.

    Required inputs:
    - seed_period: (start, end) used to build the initial prior sample of p-hats
    - analysis_period: (start, end) used for the t -> t+1 evaluation loop
    - loader(label) -> Snapshot

    Default configuration:
    - prior_family="beta"
    - mode="gh"
    - update="observed"
    """

    def __init__(
        self,
        *,
        agg: Agg,
        seed_period: Tuple[Any, Any],
        analysis_period: Tuple[Any, Any],
        loader: Callable[[str], Snapshot],
        prior_family: Literal["beta", "normal"] = "beta",
        mode: Literal["mc", "gh"] = "gh",
        update: Literal["observed", "counterfactual"] = "observed",
        cal_window_years: Optional[int] = None,
        seed_end: Optional[Any] = None,
        # Prior fitting controls
        eps: float = 1e-9,
        fallback_prior: Optional[Prior] = None,
        # Likelihood controls
        clip: float = 1e-9,
        # Inference controls (Normal prior)
        mc_samples: int = 3000,
        mc_burnin: int = 600,
        mc_width: float = 1.0,
        mc_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
        gh_nodes: int = 40,
        gh_max_iter: int = 80,
        gh_tol: float = 1e-12,
        gh_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
        rng: Optional[np.random.Generator] = None,
        # Optional p-hat estimator (jackknife etc.)
        p_estimator: Optional[Callable[[Snapshot], Sequence[float]]] = None,
    ) -> None:
        self.agg: Agg = agg
        self.seed_period_raw = seed_period
        self.analysis_period_raw = analysis_period
        self.loader = loader
        self.prior_family = prior_family
        self.mode = mode
        self.update = update
        self.cal_window_years = cal_window_years
        self.eps = float(eps)
        self.clip = float(clip)

        self.mc_samples = int(mc_samples)
        self.mc_burnin = int(mc_burnin)
        self.mc_width = float(mc_width)
        self.mc_hard_bounds_u = (float(mc_hard_bounds_u[0]), float(mc_hard_bounds_u[1]))
        self.gh_nodes = int(gh_nodes)
        self.gh_max_iter = int(gh_max_iter)
        self.gh_tol = float(gh_tol)
        self.gh_hard_bounds_u = (float(gh_hard_bounds_u[0]), float(gh_hard_bounds_u[1]))
        
        self.rng = np.random.default_rng() if rng is None else rng

        if fallback_prior is None:
            fallback_prior = Prior.beta(1.0, 1.0) if prior_family == "beta" else Prior.normal(0.0, 2.0)
        fallback_prior.validate()
        self.fallback_prior = fallback_prior

        if p_estimator is None:
            self.p_estimator = self._default_p_estimator
        else:
            self.p_estimator = p_estimator

        # Canonicalize periods to labels
        seed_labels = labels_in_period(self.agg, seed_period[0], seed_period[1])
        if len(seed_labels) == 0:
            raise ValueError("seed_period produced no labels.")
        analysis_labels = labels_in_period(self.agg, analysis_period[0], analysis_period[1])
        if len(analysis_labels) < 2:
            raise ValueError("analysis_period must contain at least two labels for a t->t+1 loop.")

        self.seed_start = seed_labels[0]
        self.seed_end = seed_labels[-1] if seed_end is None else labels_in_period(self.agg, seed_end, seed_end)[0]
        self.analysis_start = analysis_labels[0]
        self.analysis_end = analysis_labels[-1]

        # Internal caches
        self._snap_cache: Dict[str, Snapshot] = {}
        self._p_pool_by_label: Dict[str, List[float]] = {}
        self._p_obs_by_label: Dict[str, List[float]] = {}

        # Build the initial prior pool from the seed period (observed data only)
        for lbl in seed_labels:
            ps = list(self._get_p_samples_observed(lbl))
            if ps:
                self._p_pool_by_label.setdefault(lbl, []).extend(ps)

        if not self._p_pool_by_label:
            raise RuntimeError("No valid p-hats collected in the seed period; check loader/p_estimator.")

    @staticmethod
    def _default_p_estimator(snap: Snapshot) -> Sequence[float]:
        """
        Default p-hat estimator: p_hat = L/M from the snapshot adjacency.
        """
        A = np.asarray(snap.A, dtype=np.float64)
        n = int(A.shape[0])
        M = float(_M_pairs(n))
        if M <= 0.0:
            return []
        L = links_from_adjacency(A)
        p_hat = float(L / M)
        if not np.isfinite(p_hat):
            return []
        if p_hat <= 0.0 or p_hat >= 1.0:
            return []
        return [p_hat]

    def _load(self, label: str) -> Snapshot:
        """Load and cache a snapshot for a label."""
        if label in self._snap_cache:
            return self._snap_cache[label]
        snap = self.loader(label)
        if not isinstance(snap, Snapshot):
            raise TypeError("loader(label) must return a Snapshot instance.")
        self._snap_cache[label] = snap
        return snap

    def _get_p_samples_observed(self, label: str) -> Sequence[float]:
        """
        Return observed p-hat samples for a given label.

        Observed samples are cached in a dedicated container (``_p_obs_by_label``) so
        that counterfactual pooled values (self-sustained mode) cannot overwrite the
        empirical prior window.
        """
        label = str(label)
        if label in self._p_obs_by_label:
            return self._p_obs_by_label[label]
        try:
            snap = self._load(label)
            p_samps = list(self.p_estimator(snap))
        except Exception:
            p_samps = []
        self._p_obs_by_label[label] = list(p_samps)
        return p_samps

    def _fit_prior_from_window(self, end_label: str) -> Tuple[Prior, Dict[str, Any]]:
        """
        Fit the rolling prior using all p samples in the calibration window ending at end_label.
        """
        labels_cal = _labels_cal_window(self.agg, end_label, self.seed_start, self.cal_window_years)

        # Collect prior samples. The seed period is always treated as observed, even when
        # running in counterfactual (self-sustained) mode.
        p_vals: List[float] = []
        end_lbl = str(end_label)
        seed_end = str(self.seed_end)

        for lbl in labels_cal:
            lbl_s = str(lbl)

            if end_lbl <= seed_end:
                p_vals.extend(list(self._get_p_samples_observed(lbl_s)))
                continue

            if self.update == "observed":
                p_vals.extend(list(self._get_p_samples_observed(lbl_s)))
                continue

            # Counterfactual regime: keep seed labels observed, post-seed labels counterfactual.
            if lbl_s <= seed_end:
                p_vals.extend(list(self._get_p_samples_observed(lbl_s)))
            else:
                vv = self._p_pool_by_label.get(lbl_s, [])
                if vv:
                    p_vals.extend(list(vv))
                else:
                    p_vals.extend(list(self._get_p_samples_observed(lbl_s)))

        x = np.asarray(p_vals, dtype=np.float64)
        x = x[np.isfinite(x)]
        x = x[(x > 0.0) & (x < 1.0)]

        if x.size < 2:
            # fallback: use the entire pool accumulated so far
            all_vals = [p for lst in self._p_pool_by_label.values() for p in lst]
            x = np.asarray(all_vals, dtype=np.float64)
            x = x[np.isfinite(x)]
            x = x[(x > 0.0) & (x < 1.0)]

        if self.prior_family == "beta":
            pr = Prior.beta_from_samples(x, eps=self.eps, fallback=self.fallback_prior)
        else:
            pr = Prior.normal_from_samples(x, eps=self.eps, fallback=self.fallback_prior)

        diag: Dict[str, Any] = {"chosen": pr.family, "n_samples": int(x.size)}
        if pr.family == "beta":
            diag.update({"alpha": float(pr.params["alpha"]), "beta": float(pr.params["beta"])})
        else:
            diag.update({"mu0": float(pr.params["mu0"]), "sigma0": float(pr.params["sigma0"])})
        return pr, diag

    def run(self, *, return_Q: bool = False) -> BERMRollingResults:
        """
        Execute the rolling pipeline and return per-step diagnostics.

        Parameters
        ----------
        return_Q:
            If True, includes the constant predictive matrix Q in each step dict.
            This is typically unnecessary for BERM and may be memory expensive.
        """
        labels_eval = labels_in_period(self.agg, self.analysis_start, self.analysis_end)

        steps: List[Dict[str, Any]] = []

        # State for counterfactual update:
        # store only the scalar predictive probability and the active id set.
        prev_p_pred: Optional[float] = None
        prev_ids: Optional[Array] = None

        used_hard_after_seed = 0
        used_hard_seed = 0
        used_soft = 0
        skipped_pairs = 0

        for t_lbl, t1_lbl in zip(labels_eval[:-1], labels_eval[1:]):
            # Rolling prior from history up to t_lbl
            prior, prior_diag = self._fit_prior_from_window(t_lbl)

            # Load snapshots t and t+1
            try:
                snap_t = self._load(t_lbl)
                snap_t1 = self._load(t1_lbl)
            except Exception:
                skipped_pairs += 1
                continue

            # Training and prediction domains
            #
            # In the manuscript formulation, the posterior is conditioned on the snapshot at time t
            # (node set V_t), while the prediction targets the snapshot at time t+1 (node set V_{t+1}).
            train_ids = np.asarray(snap_t.node_ids, dtype=np.int64)
            eval_ids = np.asarray(snap_t1.node_ids, dtype=np.int64)

            n_train = int(train_ids.size)
            M_train = float(_M_pairs(n_train))
            n_act = int(eval_ids.size)
            M_act = float(_M_pairs(n_act))

            if M_train <= 0.0 or M_act <= 0.0:
                skipped_pairs += 1
                continue

            # Training signal L_train on V_t
            if self.update == "observed":
                L_train = float(links_from_adjacency(np.asarray(snap_t.A, dtype=np.float64)))
                hard_used = True
            else:
                # counterfactual: allow hard A_t up to and including seed_end; afterwards use soft
                if t_lbl <= self.seed_end:
                    L_train = float(links_from_adjacency(np.asarray(snap_t.A, dtype=np.float64)))
                    hard_used = True
                    used_hard_seed += 1
                else:
                    hard_used = False
                    if prev_p_pred is None or prev_ids is None:
                        skipped_pairs += 1
                        continue
                    # In self-sustained mode, replace A_t by the predictive density from the previous step.
                    # Under the Erdős–Rényi model this corresponds to a constant probability p_pred.
                    if set(map(int, prev_ids)) != set(map(int, train_ids)):
                        skipped_pairs += 1
                        continue
                    L_train = float(prev_p_pred) * float(M_train)
                    used_soft += 1

            # Compute posterior predictive mean p_pred using the training snapshot (N_t, L_t).
            # The quantity p_pred is then used to construct predictions on V_{t+1}.
            _p_pred_scalar, _L_pred_train, info = berm_predictive_probabilities(
                N=n_train,
                L_t=float(L_train),
                prior=prior,
                mode=self.mode,
                rng=self.rng,
                mc_samples=self.mc_samples,
                mc_burnin=self.mc_burnin,
                mc_width=self.mc_width,
                gh_nodes=self.gh_nodes,
                gh_max_iter=self.gh_max_iter,
                gh_tol=self.gh_tol,
                mc_hard_bounds_u=self.mc_hard_bounds_u,
                gh_hard_bounds_u=self.gh_hard_bounds_u,
                return_Q=False,
                clip=self.clip,
            )

            # Extract scalar p_pred for state propagation and reporting.
            p_pred = float(info["p_pred"])
            p_map = float(info.get("p_map", p_pred))

            # Expected number of links at t+1 on the evaluation node set.
            L_pred = float(p_pred) * float(M_act)

            pred = None
            if return_Q:
                Q = np.full((n_act, n_act), float(p_pred), dtype=np.float64)
                np.fill_diagonal(Q, 0.0)
                pred = Q

            # Update state for counterfactual mode only after the seed period.
            if self.update == "counterfactual":
                seed_end = str(self.seed_end)
                t_lbl_s = str(t_lbl)
                t1_lbl_s = str(t1_lbl)

                # Start the counterfactual chain from the last observed seed snapshot.
                if t_lbl_s >= seed_end:
                    prev_p_pred = p_pred
                    prev_ids = eval_ids.copy()

                # Store counterfactual p estimates only for post-seed labels.
                if t1_lbl_s > seed_end:
                    self._p_pool_by_label.setdefault(t1_lbl_s, []).append(float(p_map))

            # Observed adjacency on the active set at t+1 (evaluation target).
            A_true = np.asarray(snap_t1.A, dtype=np.float64)
            L_true = links_from_adjacency(A_true)

            # Expected link count at t+1 under the posterior predictive density.
            L_hat = float(L_pred)
            RE_L = float(abs(L_hat - L_true) / max(1.0, L_true))

            # Upper-triangle vector representations used for link-level diagnostics.
            a_vec = _upper_triangle_vec(A_true)
            q_vec = np.full_like(a_vec, float(p_pred), dtype=np.float64)

            # Top-L predictor used for PI/JI.
            #
            # NOTE (legacy definition): in the original notebook the quantity labelled
            # “PI” was defined as TP / L_true (a recall-like index at the true link
            # count), while the Top-L predictor itself used L_hat to select the number
            # of predicted positives. This choice makes PI invariant to fluctuations of
            # L_hat in counterfactual/self-sustained mode and preserves comparability
            # with prior results.
            #
            # In contrast, the usual precision would be TP / L_hat (i.e., TP / (TP+FP)).
            # We store that quantity separately as `precision_topL`.
            L_hat_int = int(np.clip(int(round(L_hat)), 1, int(a_vec.size))) if a_vec.size else 0
            pred_vec = _top_L_prediction(q_vec, L_hat_int) if L_hat_int > 0 else np.zeros_like(q_vec)

            a_bool = a_vec > 0.5
            pred_bool = pred_vec > 0.5
            tp = int(np.sum(a_bool & pred_bool))
            fp = int(np.sum((~a_bool) & pred_bool))
            fn = int(np.sum(a_bool & (~pred_bool)))
            precision_topL = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
            PI = float(tp / L_true) if L_true > 0 else float("nan")
            JI = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float("nan")

            if a_vec.size:
                ACC_soft = float(np.mean(a_vec * q_vec + (1.0 - a_vec) * (1.0 - q_vec)))
            else:
                ACC_soft = float("nan")

            # Degree-level diagnostics (expected degrees are uniform under BERM).
            k_true = np.sum(A_true, axis=1).astype(np.float64)
            k_hat_scalar = float(p_pred) * max(0, int(n_act) - 1)
            k_hat = np.full(int(n_act), k_hat_scalar, dtype=np.float64)
            if k_true.size:
                rel_deg_err = np.abs(k_hat - k_true) / np.maximum(1.0, k_true)
                MRE_k = float(np.max(rel_deg_err))
            else:
                MRE_k = float("nan")

            step: Dict[str, Any] = {
                "label_t": t_lbl,
                "label_t1": t1_lbl,
                "N_active": n_act,
                "N_train": int(n_train),
                "M_active": float(M_act),
                "M_train": float(M_train),
                "L_train": float(L_train),
                "L_true": float(L_true),
                "L_hat": float(L_hat),
                "L_hat_sumq": float(L_hat),
                "L_hat_int": int(L_hat_int),
                "RE_L": float(RE_L),
                "k_true": k_true,
                "k_hat": k_hat,
                "MRE_k": float(MRE_k),
                "PI": float(PI),
                "PREC_topL": float(precision_topL),
                "precision_topL": float(precision_topL),
                "JI": float(JI),
                "ACC_soft": float(ACC_soft),
                "p_pred": float(p_pred),
                "L_pred": float(L_pred),
                "prior_family": prior.family,
                "mode": self.mode,
                "update": self.update,
                "hard_used": bool(hard_used),
                "prior_diag": dict(prior_diag),
                "p_map": float(p_map),
                "info": dict(info),
            }
            if return_Q:
                step["Q"] = pred  # type: ignore[assignment]
            steps.append(step)

        # Leakage audit metadata
        if self.update == "counterfactual":
            if used_hard_after_seed != 0:
                raise RuntimeError(f"Leakage detected: used hard A_t after seed in {used_hard_after_seed} step(s).")

        return BERMRollingResults(
            steps=steps,
            prior_family=self.prior_family,
            mode=self.mode,
            update=self.update,
            agg=self.agg,
            seed_period=(self.seed_start, self.seed_end),
            analysis_period=(self.analysis_start, self.analysis_end),
        )