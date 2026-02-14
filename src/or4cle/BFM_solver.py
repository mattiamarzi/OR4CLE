#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# modules/solvers/BFM_solver.py

"""
Bayesian Fitness Model (BFM) solver for posterior-predictive link probabilities.

Model
-----
For an undirected simple graph with fitness proxies s_i >= 0, define pair weights
w_ij = s_i s_j for i != j. Conditional on a global parameter z > 0, links are
independent Bernoulli with probabilities

    p_ij(z) = (z w_ij) / (1 + z w_ij).

Given a snapshot at time t, the BFM uses the sufficient statistic

    L_t = sum_{i<j} a_ij^t,

(number of links in the strict upper triangle, or a fractional count for soft
adjacency), and computes the out-of-sample posterior-predictive probabilities
for time t+1:

    q_ij^{t+1} = E_{z | L_t, s_t}[ p_ij^{t+1}(z) ],

where w_ij^{t+1} = s_i^{t+1} s_j^{t+1}.

Posterior in u = log z
----------------------
Up to an additive constant independent of u, the log-likelihood is

    ell(u) = L_t * u - sum_{i<j} log(1 + exp(u) w_ij^t).

The term sum_{i<j} a_ij log w_ij^t does not depend on u and is omitted.

Supported priors on z
---------------------
1) Gamma:      z ~ Gamma(k, theta) (shape k, scale theta)
2) LogNormal:  z ~ LogNormal(mu0, sigma0) where mu0, sigma0 are on u = log z
3) Uniform:    z ~ Uniform(a, b), with 0 < a < b

In u-space including the Jacobian dz/du = exp(u), the corresponding log prior
terms are implemented as log pi_u(u) = log pi(exp(u)) + u, with bounds enforced
for Uniform.

Computational modes
-------------------
Two numerical modes are provided:

- mode="mc":
  Univariate slice sampling of u, followed by Monte Carlo averaging of
  p_ij(exp(u)).

- mode="gh" (default):
  Laplace approximation around the MAP u_hat (Newton with monotone line search),
  followed by Gauss-Hermite quadrature under u ~ N(u_hat, sigma^2).
  For Uniform priors, quadrature nodes are truncated to [log a, log b] and
  weights are renormalized.

Performance notes
-----------------
- All expensive sums are over the strict upper triangle only.
- Core posterior evaluations and posterior-predictive averaging are optionally
  accelerated with Numba (njit), with parallelization over pairs where beneficial.
- The posterior depends on A_t only through L_t. This module therefore accepts
  L_t directly. Helper functions are provided to compute L_t from an adjacency.

Rolling runner
--------------
To mirror the structure used in BERM_solver.py, this module also provides an
end-to-end rolling runner (BFMRollingModel) driven by a snapshot loader. The
runner supports:

- update="observed":     train on observed L_t,
- update="counterfactual": self-sustained training using L_t derived from the
  previously predicted adjacency (no leakage after the seed period).

The rolling prior can be fitted from a calibration window of z-hat values, with
either a fixed family or an automatic selection based on KS (then BIC).

Public API
----------
- Prior: container for supported prior families on z
- links_from_adjacency(A) -> L
- dcgm_calibrate_z(s, L) -> z_hat  (density calibration for logistic kernel)
- bfm_predictive_probabilities(s_t, L_t, s_next, prior, mode, ...) -> (Q, L_pred, info)
- BFMModel: dcGM-style container (fit posterior, then predict)
- Snapshot, BFMRollingModel: rolling runner

All functions are deterministic given a fixed RNG in mode="mc".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray
Agg = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]

__all__ = [
    "Prior",
    "links_from_adjacency",
    "dcgm_calibrate_z",
    "bfm_predictive_probabilities",
    "BFMModel",
    "BFMPosterior",
    "BFMResult",
    "Snapshot",
    "BFMRollingModel",
    "BFMRollingResults",
    "labels_in_period",
    "label_to_datetime",
    "rolling_mean_sym",
]


# -----------------------------------------------------------------------------
# Optional accelerator (Numba)
# -----------------------------------------------------------------------------
try:  # pragma: no cover
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
    def njit(*_args, **_kwargs):  # type: ignore
        """Return an identity decorator when Numba is unavailable."""
        def _decorator(func):
            return func
        return _decorator

    def prange(*args):  # type: ignore
        """Fallback parallel range when Numba is unavailable."""
        return range(*args)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _M_pairs(n: int) -> float:
    """Return the number of unordered pairs M = n(n-1)/2."""
    n = int(n)
    return 0.5 * n * (n - 1)


@lru_cache(maxsize=64)
def _triu_indices_cached(n: int) -> Tuple[Array, Array]:
    """Return cached indices for the strict upper triangle of an n x n matrix."""
    iu = np.triu_indices(int(n), k=1)
    return iu[0].astype(np.int64), iu[1].astype(np.int64)


def _validate_strengths(s: Array, name: str) -> Array:
    """Validate and coerce strengths to a one-dimensional float64 array."""
    v = np.asarray(s, dtype=np.float64).reshape(-1)
    if v.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(~np.isfinite(v)):
        raise ValueError(f"{name} must contain finite values only.")
    if np.any(v < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return v


def _validate_scalar(x: float, name: str) -> float:
    """Validate a finite scalar."""
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite.")
    return float(x)


def rolling_mean_sym(arr: Array, x: int = 5) -> Array:
    """
    Compute a symmetric rolling mean with half-window x.

    The i-th output is the mean of arr[j] for j in [i-x, ..., i+x] intersected
    with the valid index range. NaN values are ignored.
    """
    v = np.asarray(arr, dtype=float).reshape(-1)
    n = int(v.size)
    x = int(max(0, x))
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - x)
        hi = min(n, i + x + 1)
        out[i] = float(np.nanmean(v[lo:hi]))
    return out


# -----------------------------------------------------------------------------
# Prior container
# -----------------------------------------------------------------------------
@dataclass
class Prior:
    """
    Prior on z > 0 for the BFM.

    family:
        "gamma", "lognormal", or "uniform"

    params:
        - gamma:     {"k": float, "theta": float}
        - lognormal: {"mu0": float, "sigma0": float}   (on u = log z)
        - uniform:   {"a": float, "b": float}          (support 0 < a < b)
    """
    family: Literal["gamma", "lognormal", "uniform"]
    params: Dict[str, float] = field(default_factory=dict)

    # ------------------------- constructors -------------------------
    @staticmethod
    def gamma(k: float, theta: float) -> "Prior":
        """Construct a Gamma(k, theta) prior (shape k, scale theta)."""
        return Prior(family="gamma", params={"k": float(k), "theta": float(theta)})

    @staticmethod
    def lognormal(mu0: float, sigma0: float) -> "Prior":
        """Construct a LogNormal prior where u = log z ~ N(mu0, sigma0^2)."""
        return Prior(family="lognormal", params={"mu0": float(mu0), "sigma0": float(sigma0)})

    @staticmethod
    def uniform(a: float, b: float) -> "Prior":
        """Construct a Uniform(a, b) prior on z with 0 < a < b."""
        return Prior(family="uniform", params={"a": float(a), "b": float(b)})

    # ------------------------- validation -------------------------
    def validate(self) -> None:
        """Validate prior parameters."""
        fam = self.family
        p = self.params

        if fam == "gamma":
            k = float(p.get("k", np.nan))
            theta = float(p.get("theta", np.nan))
            if (not np.isfinite(k)) or k <= 0.0:
                raise ValueError("Gamma prior requires k > 0 and finite.")
            if (not np.isfinite(theta)) or theta <= 0.0:
                raise ValueError("Gamma prior requires theta > 0 and finite.")
        elif fam == "lognormal":
            mu0 = float(p.get("mu0", np.nan))
            sigma0 = float(p.get("sigma0", np.nan))
            if not np.isfinite(mu0):
                raise ValueError("LogNormal prior requires finite mu0.")
            if (not np.isfinite(sigma0)) or sigma0 <= 0.0:
                raise ValueError("LogNormal prior requires sigma0 > 0 and finite.")
        elif fam == "uniform":
            a = float(p.get("a", np.nan))
            b = float(p.get("b", np.nan))
            if (not np.isfinite(a)) or (not np.isfinite(b)) or (not (0.0 < a < b)):
                raise ValueError("Uniform prior requires finite 0 < a < b.")
        else:
            raise ValueError(f"Unsupported prior family: {fam}")

    # ------------------------- EB fits -------------------------
    @staticmethod
    def gamma_from_samples(
        z_samples: Array,
        *,
        var_floor: float = 1e-12,
        fallback: Optional["Prior"] = None,
    ) -> "Prior":
        """
        Construct a Gamma prior by method-of-moments from a sample of z values.

        If the sample is too small or degenerate, the fallback prior is returned.
        """
        if fallback is None:
            fallback = Prior.gamma(1.0, 1.0)

        z = np.asarray(z_samples, dtype=np.float64)
        z = z[np.isfinite(z) & (z > 0.0)]
        if z.size < 2:
            return fallback

        m = float(np.mean(z))
        v = float(np.var(z, ddof=0))
        v = max(v, float(var_floor))
        k = max(1e-12, (m * m) / v)
        theta = max(1e-12, v / m) if m > 0.0 else 1.0
        return Prior.gamma(k, theta)

    @staticmethod
    def lognormal_from_samples(
        z_samples: Array,
        *,
        sigma_floor: float = 1e-6,
        fallback: Optional["Prior"] = None,
    ) -> "Prior":
        """
        Construct a LogNormal prior on u = log z from a sample of z values.

        If the sample is too small or degenerate, the fallback prior is returned.
        """
        if fallback is None:
            fallback = Prior.lognormal(0.0, 2.0)

        z = np.asarray(z_samples, dtype=np.float64)
        z = z[np.isfinite(z) & (z > 0.0)]
        if z.size < 2:
            return fallback

        u = np.log(z)
        mu0 = float(np.mean(u))
        sigma0 = float(np.std(u, ddof=0))
        sigma0 = max(sigma0, float(sigma_floor))
        return Prior.lognormal(mu0, sigma0)

    @staticmethod
    def uniform_from_samples(
        z_samples: Array,
        *,
        pad: float = 1e-3,
        fallback: Optional["Prior"] = None,
    ) -> "Prior":
        """
        Construct a Uniform prior from a sample of z values.

        The support is set to a padded min-max interval to avoid degeneracy.
        """
        if fallback is None:
            fallback = Prior.uniform(1e-6, 1.0)

        z = np.asarray(z_samples, dtype=np.float64)
        z = z[np.isfinite(z) & (z > 0.0)]
        if z.size < 2:
            return fallback

        lo = float(np.min(z))
        hi = float(np.max(z))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0:
            return fallback
        if hi <= lo:
            hi = lo * (1.0 + 1e-6)

        lo = max(1e-12, lo * (1.0 - float(pad)))
        hi = max(lo * (1.0 + 1e-12), hi * (1.0 + float(pad)))
        return Prior.uniform(lo, hi)

    # ------------------------- u-space helpers -------------------------
    def bounds_u(self) -> Tuple[bool, float, float]:
        """Return (has_bounds, u_lo, u_hi) for u = log z."""
        if self.family == "uniform":
            a = float(self.params["a"])
            b = float(self.params["b"])
            return True, float(math.log(a)), float(math.log(b))
        return False, -np.inf, np.inf

    def suggest_u0(self) -> float:
        """Return a robust initial guess for u = log z."""
        if self.family == "gamma":
            k = float(self.params["k"])
            theta = float(self.params["theta"])
            return float(math.log(max(1e-12, k * theta)))
        if self.family == "lognormal":
            return float(self.params["mu0"])
        if self.family == "uniform":
            _, u_lo, u_hi = self.bounds_u()
            return float(0.5 * (u_lo + u_hi))
        return 0.0


# -----------------------------------------------------------------------------
# Adjacency helpers
# -----------------------------------------------------------------------------
def links_from_adjacency(A: Array) -> float:
    """
    Return L = sum_{i<j} A_ij for an undirected adjacency matrix.

    The input may be binary or soft. The diagonal is ignored.
    """
    M = np.asarray(A, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = int(M.shape[0])
    if n < 2:
        return 0.0
    i_idx, j_idx = _triu_indices_cached(n)
    return float(np.sum(M[i_idx, j_idx]))


def _pair_products(s: Array) -> Array:
    """Return w_ij = s_i s_j over the strict upper triangle as a flat float64 array."""
    n = int(s.size)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    i_idx, j_idx = _triu_indices_cached(n)
    return (s[i_idx] * s[j_idx]).astype(np.float64)


# -----------------------------------------------------------------------------
# Density calibration (dcGM-like): solve sum p_ij(z) = L for z >= 0
# -----------------------------------------------------------------------------
def dcgm_calibrate_z(
    s: Array,
    L: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
    z_hi_init: float = 1.0,
) -> float:
    """
    Calibrate z for the logistic kernel p_ij(z) = z w_ij / (1 + z w_ij).

    The equation sum_{i<j} p_ij(z) = L is solved by bisection on z, exploiting
    monotonicity. The solution is returned as a non-negative float.

    If L is zero the returned value is 0. If L exceeds the saturation level
    (number of pairs with w_ij > 0), a ValueError is raised.
    """
    s = _validate_strengths(s, "s")
    L = _validate_scalar(float(L), "L")
    if L < 0.0:
        raise ValueError("L must be non-negative.")

    n = int(s.size)
    if n < 2:
        return 0.0

    w = _pair_products(s)
    pos = w > 0.0
    M_pos = float(np.sum(pos))
    if L == 0.0:
        return 0.0
    if M_pos <= 0.0:
        raise ValueError("All pair weights are zero; cannot calibrate z for L > 0.")
    if L > M_pos + 1e-9:
        raise ValueError(f"Infeasible L: L={L} exceeds saturation M_pos={M_pos}.")

    w = w[pos].astype(np.float64)

    def sum_p(z: float) -> float:
        x = z * w
        return float(np.sum(x / (1.0 + x)))

    z_lo = 0.0
    z_hi = float(max(z_hi_init, 1e-12))
    while sum_p(z_hi) < L:
        z_hi *= 2.0
        if z_hi > 1e40:
            break

    for _ in range(int(max_iter)):
        z_mid = 0.5 * (z_lo + z_hi)
        f_mid = sum_p(z_mid) - L
        if abs(f_mid) <= float(tol) * max(1.0, L):
            return float(z_mid)
        if f_mid < 0.0:
            z_lo = z_mid
        else:
            z_hi = z_mid

    return float(0.5 * (z_lo + z_hi))


# -----------------------------------------------------------------------------
# Numba posterior evaluations (family-specific, constants omitted)
# -----------------------------------------------------------------------------
@njit(cache=True)
def _sum_log1p_exp_u_w(u: float, w: Array) -> float:
    """Compute sum_i log(1 + exp(u) w_i) with exp(u) evaluated once."""
    z = math.exp(u)
    acc = 0.0
    for i in range(w.size):
        acc += math.log(1.0 + z * w[i])
    return acc


@njit(cache=True)
def _sum_ratio_and_curv(u: float, w: Array) -> Tuple[float, float]:
    """
    Compute sums needed for derivatives:
      ratio = sum x/(1+x), curv = sum x/(1+x)^2, where x = exp(u) w.
    """
    z = math.exp(u)
    ratio = 0.0
    curv = 0.0
    for i in range(w.size):
        x = z * w[i]
        den = 1.0 + x
        ratio += x / den
        curv += x / (den * den)
    return ratio, curv


@njit(cache=True)
def _g_gamma(u: float, w: Array, L: float, k: float, theta: float) -> float:
    """Unnormalized log posterior g(u) for Gamma prior, constants omitted."""
    return (L + k) * u - _sum_log1p_exp_u_w(u, w) - math.exp(u) / theta


@njit(cache=True)
def _d1d2_gamma(u: float, w: Array, L: float, k: float, theta: float) -> Tuple[float, float]:
    """First and second derivatives of g(u) for Gamma prior."""
    ratio, curv = _sum_ratio_and_curv(u, w)
    z = math.exp(u)
    d1 = (L + k) - ratio - z / theta
    d2 = -curv - z / theta
    return d1, d2


@njit(cache=True)
def _g_lognormal(u: float, w: Array, L: float, mu0: float, sigma0: float) -> float:
    """Unnormalized log posterior g(u) for LogNormal prior, constants omitted."""
    quad = 0.5 * ((u - mu0) * (u - mu0)) / (sigma0 * sigma0)
    return L * u - _sum_log1p_exp_u_w(u, w) - quad


@njit(cache=True)
def _d1d2_lognormal(u: float, w: Array, L: float, mu0: float, sigma0: float) -> Tuple[float, float]:
    """First and second derivatives of g(u) for LogNormal prior."""
    ratio, curv = _sum_ratio_and_curv(u, w)
    invv = 1.0 / (sigma0 * sigma0)
    d1 = L - ratio - (u - mu0) * invv
    d2 = -curv - invv
    return d1, d2


@njit(cache=True)
def _g_uniform(u: float, w: Array, L: float) -> float:
    """Unnormalized log posterior g(u) for Uniform prior in z, within bounds, constants omitted."""
    return (L + 1.0) * u - _sum_log1p_exp_u_w(u, w)


@njit(cache=True)
def _d1d2_uniform(u: float, w: Array, L: float) -> Tuple[float, float]:
    """First and second derivatives of g(u) for Uniform prior in z, within bounds."""
    ratio, curv = _sum_ratio_and_curv(u, w)
    d1 = (L + 1.0) - ratio
    d2 = -curv
    return d1, d2


def _g_dispatch(u: float, w: Array, L: float, prior: Prior) -> float:
    """Return the unnormalized log posterior g(u) for the selected prior family."""
    fam = prior.family
    p = prior.params
    if fam == "gamma":
        return float(_g_gamma(float(u), w, float(L), float(p["k"]), float(p["theta"])))
    if fam == "lognormal":
        return float(_g_lognormal(float(u), w, float(L), float(p["mu0"]), float(p["sigma0"])))
    if fam == "uniform":
        return float(_g_uniform(float(u), w, float(L)))
    raise ValueError("Unsupported prior family.")


def _map_from_u_samples(u_samps: Array, w_t: Array, L_t: float, prior: Prior) -> float:
    """
    Approximate u_MAP by selecting the posterior sample maximizing g(u).

    This estimator is deterministic given the sampled u values and avoids
    additional optimization steps.
    """
    u_samps = np.asarray(u_samps, dtype=np.float64).reshape(-1)
    if u_samps.size == 0:
        return float(prior.suggest_u0())
    vals = np.empty_like(u_samps, dtype=np.float64)
    for i in range(u_samps.size):
        vals[i] = float(_g_dispatch(float(u_samps[i]), w_t, float(L_t), prior))
    return float(u_samps[int(np.argmax(vals))])


# -----------------------------------------------------------------------------
# Laplace MAP in u with monotone line search
# -----------------------------------------------------------------------------
def _laplace_map_u(
    w_t: Array,
    L_t: float,
    prior: Prior,
    *,
    u0: float,
    max_iter: int,
    tol: float,
    hard_bounds: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Compute u_map and sigma2 = -1/g''(u_map) for the selected prior family.

    For Uniform priors, bounds from the prior are enforced, in addition to
    hard_bounds.
    """
    prior.validate()
    u_lo_h, u_hi_h = float(hard_bounds[0]), float(hard_bounds[1])

    has_b, u_lo_p, u_hi_p = prior.bounds_u()
    u_lo = max(u_lo_h, float(u_lo_p)) if has_b else u_lo_h
    u_hi = min(u_hi_h, float(u_hi_p)) if has_b else u_hi_h

    if not (u_lo < u_hi):
        raise ValueError("Infeasible bounds for u = log z.")

    u = float(np.clip(u0, u_lo + 1e-12, u_hi - 1e-12))

    fam = prior.family
    p = prior.params

    def g(u_: float) -> float:
        return float(_g_dispatch(u_, w_t, float(L_t), prior))

    def d1d2(u_: float) -> Tuple[float, float]:
        if fam == "gamma":
            return _d1d2_gamma(u_, w_t, float(L_t), float(p["k"]), float(p["theta"]))
        if fam == "lognormal":
            return _d1d2_lognormal(u_, w_t, float(L_t), float(p["mu0"]), float(p["sigma0"]))
        if fam == "uniform":
            return _d1d2_uniform(u_, w_t, float(L_t))
        raise ValueError("Unsupported prior family.")

    g_u = g(u)
    for _ in range(int(max_iter)):
        d1, d2 = d1d2(u)
        if (not np.isfinite(d1)) or (not np.isfinite(d2)) or d2 >= 0.0:
            d2 = -1e-8

        step = -float(d1) / float(d2)

        alpha = 1.0
        accepted = False
        for _ls in range(40):
            u_try = float(np.clip(u + alpha * step, u_lo + 1e-12, u_hi - 1e-12))
            g_try = g(u_try)
            if np.isfinite(g_try) and (g_try >= g_u):
                u = u_try
                g_u = g_try
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            u = float(np.clip(u + 1e-3 * step, u_lo + 1e-12, u_hi - 1e-12))
            g_u = g(u)

        if abs(alpha * step) < float(tol):
            break

    _, d2 = d1d2(u)
    sigma2 = float((-1.0 / d2) if (np.isfinite(d2) and d2 < 0.0) else 1e-6)
    return float(u), float(sigma2)


# -----------------------------------------------------------------------------
# Slice sampling on u for exact posterior (MC mode)
# -----------------------------------------------------------------------------
def _slice_sample_u(
    w_t: Array,
    L_t: float,
    prior: Prior,
    *,
    u_init: float,
    n_samples: int,
    burnin: int,
    width: float,
    hard_bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> Array:
    """
    Univariate slice sampler for u, using stepping-out and shrinkage.

    For Uniform priors, sampling is bounded to the prior support in u.
    For other priors, a hard bounded interval is enforced for numerical stability.
    """
    prior.validate()

    u_lo_h, u_hi_h = float(hard_bounds[0]), float(hard_bounds[1])

    has_b, u_lo_p, u_hi_p = prior.bounds_u()
    u_lo = max(u_lo_h, float(u_lo_p)) if has_b else u_lo_h
    u_hi = min(u_hi_h, float(u_hi_p)) if has_b else u_hi_h

    if not (u_lo < u_hi):
        raise ValueError("Infeasible bounds for u = log z.")

    def g(u_: float) -> float:
        if (u_ <= u_lo) or (u_ >= u_hi):
            return -np.inf
        return float(_g_dispatch(u_, w_t, float(L_t), prior))

    u = float(np.clip(u_init, u_lo + 1e-12, u_hi - 1e-12))
    g_u = g(u)

    out = np.empty(int(n_samples), dtype=np.float64)
    kept = 0
    total = int(n_samples) + int(burnin)

    for it in range(total):
        y = g_u - float(rng.exponential(1.0))

        L = u - float(width) * float(rng.random())
        R = L + float(width)

        while (L > u_lo) and (g(L) > y):
            L -= float(width)
        while (R < u_hi) and (g(R) > y):
            R += float(width)

        L = max(L, u_lo + 1e-12)
        R = min(R, u_hi - 1e-12)

        while True:
            u_prop = float(rng.uniform(L, R))
            g_prop = g(u_prop)
            if np.isfinite(g_prop) and (g_prop > y):
                u = u_prop
                g_u = g_prop
                break
            if u_prop < u:
                L = u_prop
            else:
                R = u_prop

        if it >= burnin:
            out[kept] = u
            kept += 1

    return out


# -----------------------------------------------------------------------------
# Posterior-predictive averaging kernels (Numba, parallel over pairs)
# -----------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _qvec_from_z_samples(z_samps: Array, w_next: Array) -> Array:
    """Compute q_i = mean_s [ (z_s w_i)/(1 + z_s w_i) ] for each pair weight w_i."""
    S = z_samps.size
    M = w_next.size
    out = np.empty(M, dtype=np.float64)
    for i in prange(M):
        wi = w_next[i]
        acc = 0.0
        for s in range(S):
            x = z_samps[s] * wi
            acc += x / (1.0 + x)
        out[i] = acc / float(S)
    return out


@njit(cache=True, parallel=True)
def _qvec_from_gh(u_mu: float, u_sigma: float, xk: Array, wk_norm: Array, w_next: Array) -> Array:
    """
    Gauss-Hermite approximation with normalized weights (sum wk_norm = 1).

    Nodes are u_k = u_mu + sqrt(2) u_sigma x_k, and the expectation is
    sum_k wk_norm[k] f(u_k).
    """
    M = w_next.size
    K = xk.size
    out = np.empty(M, dtype=np.float64)
    sqrt2 = math.sqrt(2.0)
    for i in prange(M):
        wi = w_next[i]
        acc = 0.0
        for k in range(K):
            u = u_mu + sqrt2 * u_sigma * xk[k]
            z = math.exp(u)
            x = z * wi
            acc += wk_norm[k] * (x / (1.0 + x))
        out[i] = acc
    return out


# -----------------------------------------------------------------------------
# Result containers (dcGM-style)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BFMPosterior:
    """Container for posterior representation over u = log z."""
    prior: Prior
    mode: Literal["mc", "gh"]
    L_t: float
    u_samples: Optional[Array] = None
    u_map: Optional[float] = None
    sigma2: Optional[float] = None


@dataclass(frozen=True)
class BFMResult:
    """Container for posterior-predictive outputs at t+1."""
    Q: Array
    L_pred: float
    posterior: BFMPosterior
    info: Dict[str, float]


# -----------------------------------------------------------------------------
# Core public function (stateless)
# -----------------------------------------------------------------------------
def bfm_predictive_probabilities(
    s_t: Array,
    L_t: float,
    s_next: Array,
    prior: Prior,
    *,
    mode: Literal["mc", "gh"] = "gh",
    # MC controls
    mc_samples: int = 3000,
    mc_burnin: int = 600,
    mc_width: float = 1.0,
    mc_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
    rng: Optional[np.random.Generator] = None,
    # GH controls
    gh_nodes: int = 40,
    gh_max_iter: int = 80,
    gh_tol: float = 1e-10,
    gh_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
    # Numerics
    clip: float = 1e-9,
) -> Tuple[Array, float, Dict[str, float]]:
    """
    Compute Q_{t+1} = E_{z|L_t,s_t}[p_{ij}^{t+1}(z)].

    Parameters
    ----------
    s_t:
        Fitness proxies at time t, shape (N_t,).
    L_t:
        Link count at time t, either an integer (binary A_t) or fractional (soft A_t).
    s_next:
        Fitness proxies at time t+1, shape (N_{t+1},).
    prior:
        Prior on z.
    mode:
        "mc" or "gh" (default).
    clip:
        Output probabilities are clipped to [clip, 1-clip] for numerical stability.

    Returns
    -------
    Q:
        Symmetric posterior-predictive probability matrix with zero diagonal.
    L_pred:
        Expected number of links at t+1, sum_{i<j} Q_ij.
    info:
        Diagnostic dictionary (posterior summaries and settings).
    """
    prior.validate()

    s_t = _validate_strengths(s_t, "s_t")
    s_next = _validate_strengths(s_next, "s_next")
    L_t = _validate_scalar(float(L_t), "L_t")
    if L_t < 0.0:
        raise ValueError("L_t must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    n_train = int(s_t.size)
    n = int(s_next.size)
    if n < 2:
        Q0 = np.zeros((n, n), dtype=np.float64)
        return Q0, 0.0, {"n_nodes": float(n), "mode": float(0.0 if mode == "gh" else 1.0)}

    w_t = _pair_products(s_t)
    w_next = _pair_products(s_next)
    i_idx, j_idx = _triu_indices_cached(n)

    info: Dict[str, float] = {
        "n_nodes": float(n),
        "n_train": float(n_train),
        "n_pairs": float(w_t.size),
        "L_t": float(L_t),
        "mode_mc": 1.0 if mode == "mc" else 0.0,
        "mode_gh": 1.0 if mode == "gh" else 0.0,
    }
    for k_, v_ in prior.params.items():
        info[f"prior_{k_}"] = float(v_)

    if mode == "mc":
        u_init = float(prior.suggest_u0())
        u_samps = _slice_sample_u(
            w_t=w_t,
            L_t=float(L_t),
            prior=prior,
            u_init=u_init,
            n_samples=int(mc_samples),
            burnin=int(mc_burnin),
            width=float(mc_width),
            hard_bounds=mc_hard_bounds_u,
            rng=rng,
        )
        z_samps = np.exp(u_samps).astype(np.float64)
        q_vec = _qvec_from_z_samples(z_samps, w_next)

        u_map = _map_from_u_samples(u_samps, w_t, float(L_t), prior)
        z_map = float(math.exp(u_map))

        info.update({
            "u_mean": float(np.mean(u_samps)) if u_samps.size else np.nan,
            "u_std": float(np.std(u_samps, ddof=1)) if u_samps.size > 1 else np.nan,
            "z_mean": float(np.mean(z_samps)) if z_samps.size else np.nan,
            "z_std": float(np.std(z_samps, ddof=1)) if z_samps.size > 1 else np.nan,
            "u_map": float(u_map),
            "z_map": float(z_map),
            "mc_samples": float(u_samps.size),
            "mc_burnin": float(mc_burnin),
            "mc_width": float(mc_width),
            "u_lo": float(mc_hard_bounds_u[0]),
            "u_hi": float(mc_hard_bounds_u[1]),
        })

    elif mode == "gh":
        u0 = float(prior.suggest_u0())
        u_map, sigma2 = _laplace_map_u(
            w_t=w_t,
            L_t=float(L_t),
            prior=prior,
            u0=u0,
            max_iter=int(gh_max_iter),
            tol=float(gh_tol),
            hard_bounds=gh_hard_bounds_u,
        )
        z_map = float(math.exp(u_map))
        u_sigma = float(math.sqrt(max(sigma2, 1e-16)))

        from numpy.polynomial.hermite import hermgauss

        xk, wk = hermgauss(int(gh_nodes))
        xk = np.asarray(xk, dtype=np.float64)
        wk = np.asarray(wk, dtype=np.float64)

        has_b, u_lo_p, u_hi_p = prior.bounds_u()
        if has_b:
            u_nodes = u_map + math.sqrt(2.0) * u_sigma * xk
            mask = (u_nodes >= float(u_lo_p)) & (u_nodes <= float(u_hi_p))
            if not np.any(mask):
                u_mid = 0.5 * (float(u_lo_p) + float(u_hi_p))
                z_mid = float(math.exp(u_mid))
                x = z_mid * w_next
                q_vec = (x / (1.0 + x)).astype(np.float64)
                info.update({"gh_truncated": 1.0, "gh_fallback_midpoint": 1.0})
            else:
                xk = xk[mask]
                wk = wk[mask]
                wk_norm = wk / float(np.sum(wk))
                q_vec = _qvec_from_gh(float(u_map), float(u_sigma), xk, wk_norm, w_next)
                info.update({"gh_truncated": 1.0, "gh_kept_nodes": float(xk.size)})
        else:
            wk_norm = wk / float(np.sum(wk))
            q_vec = _qvec_from_gh(float(u_map), float(u_sigma), xk, wk_norm, w_next)

        info.update({
            "u_map": float(u_map),
            "z_map": float(z_map),
            "sigma2_laplace": float(sigma2),
            "u_sigma": float(u_sigma),
            "gh_nodes": float(gh_nodes),
            "gh_max_iter": float(gh_max_iter),
            "gh_tol": float(gh_tol),
            "u_lo": float(gh_hard_bounds_u[0]),
            "u_hi": float(gh_hard_bounds_u[1]),
        })

    else:
        raise ValueError("mode must be 'mc' or 'gh'.")

    if clip is not None and float(clip) > 0.0:
        q_vec = np.clip(q_vec, float(clip), 1.0 - float(clip))

    Q = np.zeros((n, n), dtype=np.float64)
    Q[i_idx, j_idx] = q_vec
    Q[j_idx, i_idx] = q_vec
    np.fill_diagonal(Q, 0.0)

    L_pred = float(np.sum(q_vec))
    info["L_pred"] = float(L_pred)
    return Q, float(L_pred), info


# -----------------------------------------------------------------------------
# dcGM-style model container
# -----------------------------------------------------------------------------
class BFMModel:
    """
    Bayesian Fitness Model calibrated from (s_t, L_t) under a prior on z.

    The posterior is one-dimensional (u = log z). After calling fit(), the
    posterior representation is cached and can be used to predict Q_{t+1}.
    """

    def __init__(self, s_t: Array, L_t: float, prior: Prior) -> None:
        self.s_t = _validate_strengths(s_t, "s_t")
        self.L_t = float(_validate_scalar(float(L_t), "L_t"))
        if self.L_t < 0.0:
            raise ValueError("L_t must be non-negative.")
        prior.validate()
        self.prior = prior
        self._posterior: Optional[BFMPosterior] = None

    def fit(
        self,
        *,
        mode: Literal["mc", "gh"] = "gh",
        rng: Optional[np.random.Generator] = None,
        # MC controls
        mc_samples: int = 3000,
        mc_burnin: int = 600,
        mc_width: float = 1.0,
        mc_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
        # GH controls
        gh_nodes: int = 40,
        gh_max_iter: int = 80,
        gh_tol: float = 1e-10,
        gh_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
    ) -> BFMPosterior:
        """
        Cache a posterior representation over u = log z.

        In mode="mc", u_samples and u_map are stored.
        In mode="gh", (u_map, sigma2) are stored.
        """
        if rng is None:
            rng = np.random.default_rng()

        w_t = _pair_products(self.s_t)
        if mode == "mc":
            u_init = float(self.prior.suggest_u0())
            u_samps = _slice_sample_u(
                w_t=w_t,
                L_t=float(self.L_t),
                prior=self.prior,
                u_init=u_init,
                n_samples=int(mc_samples),
                burnin=int(mc_burnin),
                width=float(mc_width),
                hard_bounds=mc_hard_bounds_u,
                rng=rng,
            )
            u_map = _map_from_u_samples(u_samps, w_t, float(self.L_t), self.prior)
            post = BFMPosterior(prior=self.prior, mode="mc", L_t=float(self.L_t),
                                u_samples=u_samps, u_map=float(u_map))
        elif mode == "gh":
            u0 = float(self.prior.suggest_u0())
            u_map, sigma2 = _laplace_map_u(
                w_t=w_t,
                L_t=float(self.L_t),
                prior=self.prior,
                u0=u0,
                max_iter=int(gh_max_iter),
                tol=float(gh_tol),
                hard_bounds=gh_hard_bounds_u,
            )
            post = BFMPosterior(prior=self.prior, mode="gh", L_t=float(self.L_t),
                                u_map=float(u_map), sigma2=float(sigma2))
        else:
            raise ValueError("mode must be 'mc' or 'gh'.")

        self._posterior = post
        return post

    def predict(
        self,
        s_next: Array,
        *,
        rng: Optional[np.random.Generator] = None,
        gh_nodes: int = 40,
        clip: float = 1e-9,
    ) -> BFMResult:
        """
        Compute posterior-predictive Q_{t+1} using the cached posterior.

        If the model is not fitted, fit() must be called first.
        """
        if self._posterior is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if rng is None:
            rng = np.random.default_rng()

        s_next = _validate_strengths(s_next, "s_next")

        n_train = int(self.s_t.size)
        n = int(s_next.size)
        i_idx, j_idx = _triu_indices_cached(n)
        w_next = _pair_products(s_next)

        info: Dict[str, float] = {"n_nodes": float(n), "n_train": float(n_train), "L_t": float(self.L_t)}
        for k_, v_ in self.prior.params.items():
            info[f"prior_{k_}"] = float(v_)

        if self._posterior.mode == "mc":
            u_samps = np.asarray(self._posterior.u_samples, dtype=np.float64)
            z_samps = np.exp(u_samps).astype(np.float64)
            q_vec = _qvec_from_z_samples(z_samps, w_next)

            u_map = float(self._posterior.u_map) if self._posterior.u_map is not None else _map_from_u_samples(
                u_samps, _pair_products(self.s_t), float(self.L_t), self.prior
            )
            z_map = float(math.exp(u_map))

            info.update({
                "mode_mc": 1.0,
                "u_mean": float(np.mean(u_samps)) if u_samps.size else np.nan,
                "u_std": float(np.std(u_samps, ddof=1)) if u_samps.size > 1 else np.nan,
                "u_map": float(u_map),
                "z_map": float(z_map),
                "n_samples": float(u_samps.size),
            })

        else:
            u_map = float(self._posterior.u_map)
            sigma2 = float(self._posterior.sigma2)
            u_sigma = float(math.sqrt(max(sigma2, 1e-16)))
            z_map = float(math.exp(u_map))

            from numpy.polynomial.hermite import hermgauss

            xk, wk = hermgauss(int(gh_nodes))
            xk = np.asarray(xk, dtype=np.float64)
            wk = np.asarray(wk, dtype=np.float64)

            has_b, u_lo_p, u_hi_p = self.prior.bounds_u()
            if has_b:
                u_nodes = u_map + math.sqrt(2.0) * u_sigma * xk
                mask = (u_nodes >= float(u_lo_p)) & (u_nodes <= float(u_hi_p))
                if not np.any(mask):
                    u_mid = 0.5 * (float(u_lo_p) + float(u_hi_p))
                    z_mid = float(math.exp(u_mid))
                    x = z_mid * w_next
                    q_vec = (x / (1.0 + x)).astype(np.float64)
                    info.update({"gh_truncated": 1.0, "gh_fallback_midpoint": 1.0})
                else:
                    xk = xk[mask]
                    wk = wk[mask]
                    wk_norm = wk / float(np.sum(wk))
                    q_vec = _qvec_from_gh(float(u_map), float(u_sigma), xk, wk_norm, w_next)
                    info.update({"gh_truncated": 1.0, "gh_kept_nodes": float(xk.size)})
            else:
                wk_norm = wk / float(np.sum(wk))
                q_vec = _qvec_from_gh(float(u_map), float(u_sigma), xk, wk_norm, w_next)

            info.update({
                "mode_gh": 1.0,
                "u_map": float(u_map),
                "z_map": float(z_map),
                "sigma2_laplace": float(sigma2),
                "u_sigma": float(u_sigma),
                "gh_nodes": float(gh_nodes),
            })

        if clip is not None and float(clip) > 0.0:
            q_vec = np.clip(q_vec, float(clip), 1.0 - float(clip))

        Q = np.zeros((n, n), dtype=np.float64)
        Q[i_idx, j_idx] = q_vec
        Q[j_idx, i_idx] = q_vec
        np.fill_diagonal(Q, 0.0)

        L_pred = float(np.sum(q_vec))
        info["L_pred"] = float(L_pred)
        return BFMResult(Q=Q, L_pred=float(L_pred), posterior=self._posterior, info=info)


# =============================================================================
# Label utilities (shared with the notebook plotting code)
# =============================================================================
def _parse_week_label(x: Any) -> Tuple[int, int]:
    """Parse a weekly label into (iso_year, iso_week)."""
    if isinstance(x, tuple) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, (datetime, date)):
        iso = x.isocalendar()
        return int(iso[0]), int(iso[1])
    if isinstance(x, str):
        s = x.strip()
        if "-W" in s:
            y, w = s.split("-W", 1)
            return int(y), int(w[:2])
        if "W" in s:
            y, w = s.split("W", 1)
            return int(y[:4]), int(w[:2])
    raise ValueError(f"Cannot parse weekly label: {x!r}")


def _week_to_monday(iso_year: int, iso_week: int) -> date:
    """Return the Monday date corresponding to an ISO year/week pair."""
    d = date(int(iso_year), 1, 4)  # ISO week 1 contains Jan 4
    d = d - timedelta(days=d.isoweekday() - 1)
    return d + timedelta(weeks=int(iso_week) - 1)


def _iso_weeks_in_year(y: int) -> int:
    """Return the number of ISO weeks in year *y*.

    ISO calendars have either 52 or 53 weeks. This helper is used to construct
    valid labels when shifting a (year, week) pair across years.
    """
    try:
        _ = date.fromisocalendar(int(y), 53, 1)
        return 53
    except Exception:
        return 52


def _format_week_label(d: date) -> str:
    """Format a date into canonical ISO week label 'YYYY-Www'."""
    iso = d.isocalendar()
    return f"{int(iso[0])}-W{int(iso[1]):02d}"


def _parse_month_label(x: Any) -> Tuple[int, int]:
    """Parse a monthly label into (year, month)."""
    if isinstance(x, tuple) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, (datetime, date)):
        return int(x.year), int(x.month)
    if isinstance(x, str):
        s = x.strip()
        if "-" in s:
            y, m = s.split("-", 1)
            return int(y), int(m[:2])
        if len(s) >= 6 and s[:4].isdigit():
            return int(s[:4]), int(s[4:6])
    raise ValueError(f"Cannot parse monthly label: {x!r}")


def _parse_quarter_label(x: Any) -> Tuple[int, int]:
    """Parse a quarterly label into (year, quarter)."""
    if isinstance(x, tuple) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, (datetime, date)):
        q = (int(x.month) - 1) // 3 + 1
        return int(x.year), int(q)
    if isinstance(x, str):
        s = x.strip()
        if "-Q" in s:
            y, q = s.split("-Q", 1)
            return int(y), int(q[:1])
        if "Q" in s:
            y, q = s.split("Q", 1)
            return int(y[:4]), int(q[:1])
    raise ValueError(f"Cannot parse quarterly label: {x!r}")


def _parse_year_label(x: Any) -> int:
    """Parse a yearly label into year."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (datetime, date)):
        return int(x.year)
    if isinstance(x, str):
        return int(x.strip()[:4])
    raise ValueError(f"Cannot parse yearly label: {x!r}")


def _parse_day_label(x: Any) -> date:
    """Parse a daily label into a date."""
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        return date.fromisoformat(x.strip()[:10])
    raise ValueError(f"Cannot parse day label: {x!r}")


def labels_in_period(agg: Agg, start: Any, end: Any) -> List[str]:
    """
    Generate canonical labels from start to end (inclusive) for the given aggregation.
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
        y, m = y0, m0
        while (y, m) <= (y1, m1):
            out.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                m = 1
                y += 1
        return out

    if agg == "quarterly":
        y0, q0 = _parse_quarter_label(start)
        y1, q1 = _parse_quarter_label(end)
        if (y0, q0) > (y1, q1):
            raise ValueError("start must be <= end.")
        out = []
        y, q = y0, q0
        while (y, q) <= (y1, q1):
            out.append(f"{y:04d}-Q{q:d}")
            q += 1
            if q == 5:
                q = 1
                y += 1
        return out

    if agg == "yearly":
        y0 = _parse_year_label(start)
        y1 = _parse_year_label(end)
        if y0 > y1:
            raise ValueError("start must be <= end.")
        return [f"{y:04d}" for y in range(y0, y1 + 1)]

    raise ValueError(f"Unsupported aggregation: {agg}")


def label_to_datetime(agg: Agg, label: str) -> datetime:
    """
    Map a canonical label to a datetime representing the start of the period.

    The returned datetime is naive (no timezone), suitable for plotting.
    """
    if agg == "daily":
        d = _parse_day_label(label)
        return datetime(d.year, d.month, d.day)
    if agg == "weekly":
        y, w = _parse_week_label(label)
        d = _week_to_monday(y, w)
        return datetime(d.year, d.month, d.day)
    if agg == "monthly":
        y, m = _parse_month_label(label)
        return datetime(int(y), int(m), 1)
    if agg == "quarterly":
        y, q = _parse_quarter_label(label)
        m = 1 + 3 * (int(q) - 1)
        return datetime(int(y), int(m), 1)
    if agg == "yearly":
        y = _parse_year_label(label)
        return datetime(int(y), 1, 1)
    raise ValueError(f"Unsupported aggregation: {agg}")


# =============================================================================
# Rolling runner (coherent with BERMRollingModel)
# =============================================================================
@dataclass
class Snapshot:
    """
    Snapshot representation for rolling BFM.

    A:
        Square adjacency matrix (binary or soft), aligned with node_ids.
    s:
        Fitness proxy vector aligned with A.
    node_ids:
        Node identifiers aligned with rows/cols of A. If None, uses 0..N-1.
    """
    A: Array
    s: Array
    node_ids: Optional[Array] = None

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Snapshot.A must be a square matrix.")
        s = _validate_strengths(self.s, "Snapshot.s")
        if s.size != A.shape[0]:
            raise ValueError("Snapshot.s must match Snapshot.A size.")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "s", s)
        if self.node_ids is None:
            object.__setattr__(self, "node_ids", np.arange(A.shape[0], dtype=np.int64))
        else:
            ids = np.asarray(self.node_ids, dtype=np.int64).reshape(-1)
            if ids.size != A.shape[0]:
                raise ValueError("Snapshot.node_ids must match Snapshot.A size.")
            object.__setattr__(self, "node_ids", ids)


@dataclass
class BFMRollingResults:
    """Container for rolling outputs."""
    steps: List[Dict[str, Any]]
    prior_family: str
    mode: str
    update: str
    agg: str
    seed_period: Tuple[str, str]
    analysis_period: Tuple[str, str]


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
        d1 = _parse_day_label(end_label)
        d0 = d1 - timedelta(days=365 * int(years))
        start_bound = d0.isoformat()

    start_eff = max(str(global_start), str(start_bound))
    return labels_in_period(agg, start_eff, end_label)


def _top_L_prediction(q_vec: Array, L: int) -> Array:
    """Return a 0/1 vector selecting the top-L entries of q_vec."""
    q = np.asarray(q_vec, dtype=np.float64).reshape(-1)
    m = int(q.size)
    L = int(max(0, min(L, m)))
    out = np.zeros(m, dtype=np.int8)
    if L == 0:
        return out
    idx = np.argpartition(-q, L - 1)[:L]
    out[idx] = 1
    return out


def _upper_triangle_vec(M: Array) -> Array:
    """Return the strict upper-triangle vector of a square matrix."""
    A = np.asarray(M, dtype=np.float64)
    i, j = _triu_indices_cached(int(A.shape[0]))
    return A[i, j].astype(np.float64)


def _submatrix_on_ids_sum_upper(snap: Snapshot, ids: Array) -> float:
    """
    Compute sum_{i<j} A_ij on the intersection between snap.node_ids and ids.
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    snap_ids = np.asarray(snap.node_ids, dtype=np.int64).reshape(-1)
    if ids.size < 2 or snap_ids.size < 2:
        return 0.0
    idx_map = {int(v): i for i, v in enumerate(snap_ids)}
    keep = [idx_map.get(int(v), -1) for v in ids]
    keep = np.asarray([k for k in keep if k >= 0], dtype=np.int64)
    if keep.size < 2:
        return 0.0
    A_sub = np.asarray(snap.A, dtype=np.float64)[np.ix_(keep, keep)]
    return float(links_from_adjacency(A_sub))


def _vector_on_ids(snap: Snapshot, ids: Array) -> Array:
    """
    Extract s aligned to 'ids' from the snapshot, imputing zeros for missing nodes.
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    snap_ids = np.asarray(snap.node_ids, dtype=np.int64).reshape(-1)
    idx_map = {int(v): i for i, v in enumerate(snap_ids)}
    out = np.zeros(ids.size, dtype=np.float64)
    for k, v in enumerate(ids):
        j = idx_map.get(int(v), -1)
        if j >= 0:
            out[k] = float(snap.s[j])
    return out


def _sum_upper_Q_on_overlap(Q_prev: Array, prev_ids: Array, ids_now: Array) -> float:
    """
    Sum the strict upper triangle of a predicted matrix on the node overlap.
    """
    prev_ids = np.asarray(prev_ids, dtype=np.int64).reshape(-1)
    ids_now = np.asarray(ids_now, dtype=np.int64).reshape(-1)
    if prev_ids.size < 2 or ids_now.size < 2:
        return 0.0
    idx_map = {int(v): i for i, v in enumerate(prev_ids)}
    keep = [idx_map.get(int(v), -1) for v in ids_now]
    keep = np.asarray([k for k in keep if k >= 0], dtype=np.int64)
    if keep.size < 2:
        return 0.0
    Q_sub = np.asarray(Q_prev, dtype=np.float64)[np.ix_(keep, keep)]
    return float(links_from_adjacency(Q_sub))


def _sum_upper_Q(Q: Array) -> float:
    """Compute sum_{i<j} Q_ij for a square matrix Q.

    The returned value is used as a soft analogue of the link count L when
    propagating the self-sustained (counterfactual) chain.
    """
    Qm = np.asarray(Q, dtype=np.float64)
    if Qm.ndim != 2 or Qm.shape[0] != Qm.shape[1]:
        raise ValueError('Q must be a square matrix.')
    if Qm.shape[0] < 2:
        return 0.0
    return float(np.sum(np.triu(Qm, k=1)))


def _reindex_square_matrix(Q: Array, ids_from: Array, ids_to: Array) -> Array:
    """Reindex a square matrix to a new node-id ordering.

    Parameters
    ----------
    Q:
        Square matrix defined on the ordering ``ids_from``.
    ids_from:
        Node identifiers corresponding to the rows/columns of ``Q``.
    ids_to:
        Desired node identifiers ordering. The sets of identifiers must coincide
        with ``ids_from``; only the ordering may differ.

    Returns
    -------
    Array
        The matrix Q reindexed to the ``ids_to`` ordering.
    """
    ids_from = np.asarray(ids_from, dtype=np.int64).reshape(-1)
    ids_to = np.asarray(ids_to, dtype=np.int64).reshape(-1)
    Qm = np.asarray(Q, dtype=np.float64)
    if Qm.ndim != 2 or Qm.shape[0] != Qm.shape[1]:
        raise ValueError('Q must be a square matrix.')
    if ids_from.size != Qm.shape[0]:
        raise ValueError('ids_from must have the same length as Q dimensions.')
    if ids_to.size != Qm.shape[0]:
        raise ValueError('ids_to must have the same length as Q dimensions.')
    if np.array_equal(ids_from, ids_to):
        return Qm
    if set(map(int, ids_from.tolist())) != set(map(int, ids_to.tolist())):
        raise ValueError('ids_from and ids_to must contain the same identifiers.')
    pos = {int(v): i for i, v in enumerate(ids_from)}
    order = np.asarray([pos[int(v)] for v in ids_to], dtype=np.int64)
    return Qm[np.ix_(order, order)]



def _fit_prior_auto_z(
    z: Array,
    *,
    alpha: float,
    fallback: Prior,
    candidates: Sequence[str] = ("gamma", "lognormal", "uniform"),
) -> Tuple[Prior, Dict[str, float]]:
    """
    Fit candidate priors on z and select by (KS pass -> best BIC).

    This routine is designed to be numerically robust in small or nearly-degenerate
    calibration windows, where maximum-likelihood fits (e.g., SciPy's gamma.fit)
    may fail when the sample variance is extremely small.

    Selection rule
    --------------
    - If a KS p-value is available (SciPy present), prefer candidates with p >= alpha
      and choose the one with smallest BIC among those.
    - Otherwise, fall back to the smallest BIC among all candidates.

    Notes
    -----
    - Fitting is performed on a mild winsorized version of the sample to reduce the
      influence of extreme outliers, but evaluation (KS, log-likelihood) is done on
      the full sample.
    - Gamma and LogNormal parameters are obtained with method-of-moments / closed-form
      estimates; SciPy is used only for CDF/PDF evaluations when available.
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z) & (z > 0.0)]
    if z.size < 2:
        return fallback, {"ks_D": np.nan, "ks_p": np.nan, "bic": np.nan, "loglik": np.nan, "chosen": "fallback"}

    # Prefer SciPy for reliable CDF/PDF and KS p-values, but do not depend on it for fitting.
    try:  # pragma: no cover
        from scipy import stats  # type: ignore
        have_scipy = True
    except Exception:  # pragma: no cover
        have_scipy = False

    z_sorted = np.sort(z)
    n = int(z_sorted.size)

    # Winsorize for fitting only (robustification).
    # The quantile levels are chosen to be conservative and stable for small n.
    if n >= 10:
        q_lo, q_hi = np.quantile(z_sorted, [0.01, 0.99])
        if np.isfinite(q_lo) and np.isfinite(q_hi) and (q_hi > q_lo):
            z_fit = z_sorted[(z_sorted >= q_lo) & (z_sorted <= q_hi)]
            if z_fit.size >= 2:
                pass
            else:
                z_fit = z_sorted
        else:
            z_fit = z_sorted
    else:
        z_fit = z_sorted

    def _gamma_mom_params(x: Array) -> Tuple[float, float]:
        """Return (k, theta) via moment matching with a variance floor."""
        mu = float(np.mean(x))
        var = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
        if (not np.isfinite(mu)) or mu <= 0.0:
            return float(fallback.params.get("k", 1.0)), float(fallback.params.get("theta", 1.0))
        # Variance floor prevents k -> inf in nearly-degenerate samples.
        var_floor = max(1e-12 * mu * mu, 1e-18)
        var_eff = max(var, var_floor)
        k = mu * mu / var_eff
        theta = var_eff / mu
        # Extra guards
        k = float(np.clip(k, 1e-6, 1e12))
        theta = float(max(theta, 1e-300))
        return k, theta

    def _lognorm_params(x: Array) -> Tuple[float, float]:
        """Return (mu0, sigma0) for u=log z with a standard-deviation floor."""
        u = np.log(x)
        mu0 = float(np.mean(u))
        sigma0 = float(np.std(u, ddof=1)) if u.size > 1 else 0.0
        if (not np.isfinite(mu0)):
            mu0 = float(np.log(np.mean(x)))
        if (not np.isfinite(sigma0)) or sigma0 <= 0.0:
            sigma0 = 1e-6
        sigma0 = float(np.clip(sigma0, 1e-6, 1e3))
        return mu0, sigma0

    rows: List[Tuple[str, Prior, float, float, float, float]] = []

    for fam in candidates:
        if fam == "gamma":
            k, theta = _gamma_mom_params(z_fit)
            prior = Prior.gamma(float(k), float(theta))

            if have_scipy:
                try:
                    D, pval = stats.kstest(z_sorted, "gamma", args=(k, 0.0, theta))
                    loglik = float(np.sum(stats.gamma.logpdf(z_sorted, k, loc=0.0, scale=theta)))
                    D = float(D); pval = float(pval)
                except Exception:
                    D, pval, loglik = np.nan, np.nan, np.nan
            else:
                # Log-likelihood for Gamma(k, theta) with loc=0
                loglik = float(np.sum((k - 1.0) * np.log(z_sorted) - z_sorted / theta - k * np.log(theta) - math.lgamma(k)))
                D, pval = np.nan, np.nan

            bic = float(-2.0 * loglik + 2.0 * math.log(max(1, n))) if np.isfinite(loglik) else np.inf
            rows.append((fam, prior, float(D), float(pval), float(loglik), float(bic)))

        elif fam == "lognormal":
            mu0, sigma0 = _lognorm_params(z_fit)
            prior = Prior.lognormal(float(mu0), float(sigma0))

            if have_scipy:
                try:
                    D, pval = stats.kstest(z_sorted, "lognorm", args=(sigma0, 0.0, math.exp(mu0)))
                    loglik = float(np.sum(stats.lognorm.logpdf(z_sorted, sigma0, loc=0.0, scale=math.exp(mu0))))
                    D = float(D); pval = float(pval)
                except Exception:
                    D, pval, loglik = np.nan, np.nan, np.nan
            else:
                u = np.log(z_sorted)
                loglik = float(np.sum(-0.5 * ((u - mu0) / sigma0) ** 2 - np.log(z_sorted * sigma0 * math.sqrt(2.0 * math.pi))))
                D, pval = np.nan, np.nan

            bic = float(-2.0 * loglik + 2.0 * math.log(max(1, n))) if np.isfinite(loglik) else np.inf
            rows.append((fam, prior, float(D), float(pval), float(loglik), float(bic)))

        elif fam == "uniform":
            lo = max(1e-12, float(np.min(z_sorted)) * 0.999)
            hi = max(lo * (1.0 + 1e-12), float(np.max(z_sorted)) * 1.001)
            prior = Prior.uniform(lo, hi)

            if have_scipy:
                try:
                    D, pval = stats.kstest(z_sorted, "uniform", args=(lo, hi - lo))
                    loglik = float(np.sum(stats.uniform.logpdf(z_sorted, loc=lo, scale=hi - lo)))
                    D = float(D); pval = float(pval)
                except Exception:
                    D, pval, loglik = np.nan, np.nan, np.nan
            else:
                D, pval = np.nan, np.nan
                loglik = float(-n * math.log(hi - lo))

            bic = float(-2.0 * loglik + 2.0 * math.log(max(1, n))) if np.isfinite(loglik) else np.inf
            rows.append((fam, prior, float(D), float(pval), float(loglik), float(bic)))

        else:
            raise ValueError(f"Unknown prior family: {fam}")

    # Prefer KS-pass if available, otherwise fall back to best BIC (or the first finite).
    rows.sort(key=lambda r: r[-1])  # by BIC
    passed = [r for r in rows if np.isfinite(r[3]) and (r[3] >= float(alpha))]
    chosen = passed[0] if len(passed) > 0 else rows[0]
    fam, prior, D, pval, loglik, bic = chosen
    return prior, {"ks_D": float(D), "ks_p": float(pval), "loglik": float(loglik), "bic": float(bic), "chosen": str(fam)}


class BFMRollingModel:
    """
    End-to-end rolling BFM runner driven by a snapshot loader.

    Required inputs:
    - seed_period: (start, end) used to build the initial prior sample of z-hats
    - analysis_period: (start, end) used for the t -> t+1 evaluation loop
    - loader(label) -> Snapshot
    """

    def __init__(
        self,
        *,
        agg: Agg,
        seed_period: Tuple[Any, Any],
        analysis_period: Tuple[Any, Any],
        loader: Callable[[str], Snapshot],
        prior_family: Literal["auto", "gamma", "lognormal", "uniform"] = "auto",
        mode: Literal["mc", "gh"] = "gh",
        update: Literal["observed", "counterfactual"] = "observed",
        cal_window_years: Optional[int] = None,
        seed_end: Optional[Any] = None,
        # Prior fitting controls
        ks_alpha: float = 0.05,
        fallback_prior: Optional[Prior] = None,
        # Likelihood controls
        clip: float = 1e-9,
        # Posterior inference controls
        mc_samples: int = 3000,
        mc_burnin: int = 600,
        mc_width: float = 1.0,
        mc_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
        gh_nodes: int = 40,
        gh_max_iter: int = 80,
        gh_tol: float = 1e-10,
        gh_hard_bounds_u: Tuple[float, float] = (-40.0, 40.0),
        rng: Optional[np.random.Generator] = None,
        # Optional z-hat estimator (jackknife etc.)
        z_estimator: Optional[Callable[[Snapshot], Sequence[float]]] = None,
        # Outputs
        return_Q: bool = False,
        return_degrees: bool = True,
    ) -> None:
        self.agg: Agg = agg
        self.seed_period_raw = seed_period
        self.analysis_period_raw = analysis_period
        self.loader = loader
        self.prior_family = prior_family
        self.mode = mode
        self.update = update
        self.cal_window_years = cal_window_years
        self.ks_alpha = float(ks_alpha)
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
            fallback_prior = Prior.gamma(1.0, 1.0)
        fallback_prior.validate()
        self.fallback_prior = fallback_prior

        if z_estimator is None:
            self.z_estimator = self._default_z_estimator
        else:
            self.z_estimator = z_estimator

        self.return_Q = bool(return_Q)
        self.return_degrees = bool(return_degrees)

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
        self._z_pool_by_label: Dict[str, List[float]] = {}
        self._z_obs_by_label: Dict[str, List[float]] = {}

        # Build the initial prior pool from the seed period (observed data only)
        for lbl in seed_labels:
            zs = list(self._get_z_samples_observed(lbl))
            if zs:
                self._z_pool_by_label.setdefault(lbl, []).extend(zs)

        if not self._z_pool_by_label:
            raise RuntimeError("No valid z-hats collected in the seed period; check loader/z_estimator.")

    @staticmethod
    def _default_z_estimator(snap: Snapshot) -> Sequence[float]:
        """
        Default z-hat estimator: calibrate z by matching E[L] to observed L.

        This estimator is consistent with the logistic kernel used by the BFM.
        """
        A = np.asarray(snap.A, dtype=np.float64)
        s = np.asarray(snap.s, dtype=np.float64)
        L = float(links_from_adjacency(A))
        if not np.isfinite(L) or L <= 0.0:
            return []
        try:
            z_hat = float(dcgm_calibrate_z(s, L))
        except Exception:
            return []
        if not np.isfinite(z_hat) or z_hat <= 0.0:
            return []
        return [z_hat]

    def _load(self, label: str) -> Snapshot:
        """Load and cache a snapshot for a label."""
        if label in self._snap_cache:
            return self._snap_cache[label]
        snap = self.loader(label)
        if not isinstance(snap, Snapshot):
            raise TypeError("loader(label) must return a Snapshot instance.")
        self._snap_cache[label] = snap
        return snap

    def _get_z_samples_observed(self, label: str) -> Sequence[float]:
        """
        Return observed z-hat samples for a given label.

        Notes
        -----
        Observed samples are cached in a dedicated container (``_z_obs_by_label``) to
        avoid contamination by counterfactual pooled values when running in
        self-sustained mode.
        """
        label = str(label)
        if label in self._z_obs_by_label:
            return self._z_obs_by_label[label]
        try:
            snap = self._load(label)
            z_samps = list(self.z_estimator(snap))
        except Exception:
            z_samps = []
        self._z_obs_by_label[label] = list(z_samps)
        return z_samps

    def _fit_prior_from_window(self, end_label: str) -> Tuple[Prior, Dict[str, Any]]:
        """Fit the rolling prior using samples in the calibration window ending at end_label."""
        labels_cal = _labels_cal_window(self.agg, end_label, self.seed_start, self.cal_window_years)

        # Collect prior samples. The seed period is always treated as observed, even when
        # running in counterfactual (self-sustained) mode.
        z_vals: List[float] = []
        end_lbl = str(end_label)
        seed_end = str(self.seed_end)

        for lbl in labels_cal:
            lbl_s = str(lbl)

            if end_lbl <= seed_end:
                z_vals.extend(list(self._get_z_samples_observed(lbl_s)))
                continue

            if self.update == "observed":
                z_vals.extend(list(self._get_z_samples_observed(lbl_s)))
                continue

            # Counterfactual regime: keep seed labels observed, post-seed labels counterfactual.
            if lbl_s <= seed_end:
                z_vals.extend(list(self._get_z_samples_observed(lbl_s)))
            else:
                vv = self._z_pool_by_label.get(lbl_s, [])
                if vv:
                    z_vals.extend(list(vv))
                else:
                    z_vals.extend(list(self._get_z_samples_observed(lbl_s)))

        z = np.asarray(z_vals, dtype=np.float64)
        z = z[np.isfinite(z) & (z > 0.0)]
        if z.size < 2:
            all_vals = [v for lst in self._z_pool_by_label.values() for v in lst]
            z = np.asarray(all_vals, dtype=np.float64)
            z = z[np.isfinite(z) & (z > 0.0)]

        if z.size < 2:
            return self.fallback_prior, {"chosen": "fallback", "ks_D": np.nan, "ks_p": np.nan, "bic": np.nan, "loglik": np.nan}

        if self.prior_family == "auto":
            return _fit_prior_auto_z(z, alpha=self.ks_alpha, fallback=self.fallback_prior)

        if self.prior_family == "gamma":
            return Prior.gamma_from_samples(z, fallback=self.fallback_prior), {"chosen": "gamma"}
        if self.prior_family == "lognormal":
            return Prior.lognormal_from_samples(z, fallback=self.fallback_prior), {"chosen": "lognormal"}
        if self.prior_family == "uniform":
            return Prior.uniform_from_samples(z, fallback=self.fallback_prior), {"chosen": "uniform"}

        raise ValueError("prior_family must be one of: auto, gamma, lognormal, uniform.")

    def run(self) -> BFMRollingResults:
        """
        Execute the rolling pipeline.

        Per-step metrics:
        - L_true, L_hat_sumq, RE_L
        - MRE_k (max relative degree error)
        - PI, JI, ACC_soft
        """
        labels_eval = labels_in_period(self.agg, self.analysis_start, self.analysis_end)

        steps: List[Dict[str, Any]] = []

        prev_Q: Optional[Array] = None
        prev_ids: Optional[Array] = None

        hard_used_after_seed = 0

        for t_lbl, t1_lbl in zip(labels_eval[:-1], labels_eval[1:]):
            prior, prior_diag = self._fit_prior_from_window(t_lbl)

            snap_t = self._load(t_lbl)
            snap_t1 = self._load(t1_lbl)

            # Training and prediction domains
            #
            # In the manuscript formulation, the posterior is conditioned on the snapshot at time t
            # (node set V_t), while the prediction targets the snapshot at time t+1 (node set V_{t+1}).
            train_ids = np.asarray(snap_t.node_ids, dtype=np.int64)
            eval_ids = np.asarray(snap_t1.node_ids, dtype=np.int64)

            if train_ids.size < 2 or eval_ids.size < 2:
                continue

            # Training strengths at time t (defined on V_t)
            s_train = np.asarray(snap_t.s, dtype=np.float64)

            # Prediction strengths at time t+1 (defined on V_{t+1})
            s_next = np.asarray(snap_t1.s, dtype=np.float64)

            # Training link count L_t on V_t
            if self.update == "observed":
                L_train = float(links_from_adjacency(np.asarray(snap_t.A, dtype=np.float64)))
                hard_used = True
            else:
                if t_lbl <= self.seed_end:
                    L_train = float(links_from_adjacency(np.asarray(snap_t.A, dtype=np.float64)))
                    hard_used = True
                else:
                    hard_used = False
                    if prev_Q is None or prev_ids is None:
                        continue
                    # In self-sustained mode, replace A_t by the previous predictive matrix Q_t.
                    Q_t = _reindex_square_matrix(prev_Q, prev_ids, train_ids)
                    L_train = _sum_upper_Q(Q_t)

            if (t_lbl > self.seed_end) and hard_used and (self.update == "counterfactual"):
                hard_used_after_seed += 1

            Q_pred, L_pred, info = bfm_predictive_probabilities(
                s_t=s_train,
                L_t=float(L_train),
                s_next=s_next,
                prior=prior,
                mode=self.mode,
                rng=self.rng,
                mc_samples=self.mc_samples,
                mc_burnin=self.mc_burnin,
                mc_width=self.mc_width,
                mc_hard_bounds_u=self.mc_hard_bounds_u,
                gh_nodes=self.gh_nodes,
                gh_max_iter=self.gh_max_iter,
                gh_tol=self.gh_tol,
                gh_hard_bounds_u=self.gh_hard_bounds_u,
                clip=self.clip,
            )

            A_true = np.asarray(snap_t1.A, dtype=np.float64)

            q_vec = _upper_triangle_vec(Q_pred)
            a_vec = _upper_triangle_vec(A_true)

            L_hat_sumq = float(np.sum(q_vec))
            L_true = float(np.sum(a_vec))

            L_hat_int = int(round(L_hat_sumq))
            pred_vec = _top_L_prediction(q_vec, L_hat_int)

            M_pairs = int(q_vec.size)
            TP_soft = float(np.sum(a_vec * q_vec))
            TN_soft = float(np.sum((1.0 - a_vec) * (1.0 - q_vec)))
            ACC_soft = (TP_soft + TN_soft) / float(M_pairs) if M_pairs > 0 else np.nan

            TP_hard_topL = int(np.sum((pred_vec == 1) & (a_vec == 1)))
            union_topL = int(np.sum((pred_vec == 1) | (a_vec == 1)))
            PI = (TP_hard_topL / L_true) if L_true > 0 else np.nan
            JI = (TP_hard_topL / union_topL) if union_topL > 0 else np.nan

            FP_hard_topL = int(np.sum((pred_vec == 1) & (a_vec == 0)))
            FN_hard_topL = int(np.sum((pred_vec == 0) & (a_vec == 1)))
            precision_topL = (TP_hard_topL / (TP_hard_topL + FP_hard_topL)) if (TP_hard_topL + FP_hard_topL) > 0 else np.nan

            k_hat = np.sum(Q_pred, axis=1)
            k_true = np.sum(A_true, axis=1).astype(np.float64)
            rel_err_k = np.abs(k_hat - k_true) / np.maximum(1.0, k_true)
            MRE_k = float(np.max(rel_err_k)) if rel_err_k.size else np.nan
            RE_L = (abs(L_hat_sumq - L_true) / L_true) if L_true > 0 else np.nan

            # Counterfactual propagation: update the self-sustained state only after the seed period.
            if self.update == "counterfactual":
                seed_end = str(self.seed_end)
                t_lbl_s = str(t_lbl)
                t1_lbl_s = str(t1_lbl)

                # Start the counterfactual chain from the last observed seed snapshot.
                if t_lbl_s >= seed_end:
                    prev_Q = np.asarray(Q_pred, dtype=np.float64).copy()
                    prev_ids = eval_ids.copy()

                # Store counterfactual z estimates only for post-seed labels.
                if t1_lbl_s > seed_end:
                    z_map = float(info.get("z_map", np.nan))
                    if np.isfinite(z_map) and z_map > 0.0:
                        self._z_pool_by_label.setdefault(t1_lbl_s, []).append(z_map)

            step: Dict[str, Any] = {
                "label_t": t_lbl,
                "label_t1": t1_lbl,
                "N_active": int(eval_ids.size),
                "N_train": int(train_ids.size),
                "M_active": float(_M_pairs(int(eval_ids.size))),
                "M_train": float(_M_pairs(int(train_ids.size))),
                "L_train": float(L_train),
                "L_true": float(L_true),
                "L_hat_sumq": float(L_hat_sumq),
                "L_hat": float(L_hat_sumq),
                "L_hat_int": int(L_hat_int),
                "L_pred": float(L_pred),
                "RE_L": float(RE_L),
                "MRE_k": float(MRE_k),
                "PI": float(PI),
                "JI": float(JI),
                "PREC_topL": float(precision_topL),
                "precision_topL": float(precision_topL),
                "ACC_soft": float(ACC_soft),
                "prior_family": prior.family,
                "mode": self.mode,
                "update": self.update,
                "hard_used": bool(hard_used),
                "prior_diag": dict(prior_diag),
                "info": dict(info),
            }

            if self.return_degrees:
                step["k_true"] = np.asarray(k_true, dtype=np.float64)
                step["k_hat"] = np.asarray(k_hat, dtype=np.float64)

            if self.return_Q:
                step["Q"] = np.asarray(Q_pred, dtype=np.float64)

            steps.append(step)

        if self.update == "counterfactual" and hard_used_after_seed != 0:
            raise RuntimeError(f"Leakage detected: used observed A_t after seed in {hard_used_after_seed} step(s).")

        return BFMRollingResults(
            steps=steps,
            prior_family=self.prior_family,
            mode=self.mode,
            update=self.update,
            agg=self.agg,
            seed_period=(self.seed_start, self.seed_end),
            analysis_period=(self.analysis_start, self.analysis_end),
        )
