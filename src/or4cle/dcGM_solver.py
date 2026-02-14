#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# modules/solvers/dcGM_solver.py

"""
density-corrected Gravity Model (dcGM) solver for binary undirected simple graphs.

Model
-----
For i != j the link probabilities are

    p_ij(z) = (z s_i s_j) / (1 + z s_i s_j),

where s_i >= 0 are node fitness proxies (often a normalized strength sequence),
and z >= 0 is a single global scale calibrated by matching the expected number
of undirected links

    L_hat(z) = sum_{i<j} p_ij(z) = L_target.

Numerical strategy
------------------
The solver calibrates theta = log z by damped Newton iterations on L_hat(theta),
with an Armijo-type backtracking line search on the merit function

    phi(theta) = 0.5 * (L_hat(theta) - L_target)^2.

Probabilities are evaluated in log-space using a stable logistic sigmoid:

    p_ij = sigma(theta + log s_i + log s_j),
    d p_ij / dtheta = p_ij (1 - p_ij).

Implementation notes
--------------------
- Input validation and error handling are performed in Python wrappers.
- Low-level numerical kernels are optionally accelerated with Numba. When Numba
  is unavailable, the code falls back to pure-Python/NumPy paths.
- Zero strengths imply p_ij = 0 for any incident pair at any finite z.
- The diagonal is always set to zero (simple graph without self-loops).

Public API
----------
- dcgm_calibrate_z(strengths, L_target, ...) -> z
- dcgm_probabilities(strengths, z) -> P
- dcgm_fit(strengths, L_target, ...) -> (z, P)
- DcGMModel: small container to cache calibration outputs.

This module is intended as a lightweight, deterministic baseline used by ORBIT:
(i) to fit z on empirical snapshots (e.g., to build empirical priors elsewhere),
(ii) to compute plug-in probabilities p_ij(z*) under the same L_target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Optional accelerator (Numba)
# -----------------------------------------------------------------------------
try:  # pragma: no cover
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    def njit(*_args, **_kwargs):  # type: ignore
        """Return an identity decorator when Numba is unavailable."""
        def _decorator(func):
            return func
        return _decorator


# -----------------------------------------------------------------------------
# Low-level numerics
# -----------------------------------------------------------------------------
@njit(cache=True)
def _expit_scalar(t: float) -> float:
    """
    Logistic sigmoid sigma(t)=1/(1+exp(-t)) evaluated without overflow.

    The piecewise formulation ensures numerical stability for large |t|.
    """
    if t >= 0.0:
        e = np.exp(-t)
        return 1.0 / (1.0 + e)
    e = np.exp(t)
    return e / (1.0 + e)


@njit(cache=True)
def _log_strengths(s: np.ndarray) -> np.ndarray:
    """
    Map nonnegative strengths to logarithms, with log(0) = -inf.

    Under sigma(theta + log s_i + log s_j), a -inf term implies p_ij = 0.
    """
    n = s.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        si = s[i]
        out[i] = np.log(si) if si > 0.0 else -np.inf
    return out


@njit(cache=True)
def _Lhat_and_dL(theta: float, log_s: np.ndarray) -> Tuple[float, float]:
    """
    Compute L_hat(theta) and dL_hat/dtheta for dcGM.

    With p_ij = sigma(theta + log s_i + log s_j),
    d p_ij / dtheta = p_ij (1 - p_ij).
    """
    n = log_s.size
    L_hat = 0.0
    dL = 0.0
    for i in range(n):
        for j in range(i):
            t = theta + log_s[i] + log_s[j]
            p = _expit_scalar(t)
            L_hat += p
            dL += p * (1.0 - p)
    return L_hat, dL


@njit(cache=True)
def _probabilities_from_theta(log_s: np.ndarray, theta: float) -> np.ndarray:
    """
    Construct the symmetric probability matrix P for a given (log_s, theta).

    The diagonal is set to zero to represent a simple graph.
    """
    n = log_s.size
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i):
            t = theta + log_s[i] + log_s[j]
            p = _expit_scalar(t)
            P[i, j] = p
            P[j, i] = p
    return P


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------
def _validate_strengths(strengths: np.ndarray) -> np.ndarray:
    """Validate and coerce strengths to a one-dimensional float64 array."""
    s = np.asarray(strengths, dtype=np.float64).reshape(-1)
    if s.ndim != 1:
        raise ValueError("strengths must be a one-dimensional array.")
    if np.any(~np.isfinite(s)):
        raise ValueError("strengths must contain only finite values.")
    if np.any(s < 0.0):
        raise ValueError("strengths must be non-negative.")
    return s


def _effective_L_max_from_strengths(s: np.ndarray) -> float:
    """
    Maximum achievable expected link count given zero-strength nodes.

    If s_i = 0 then p_ij = 0 for all j at any finite z, so only nodes with s_i>0
    can contribute to links. Hence the effective upper bound is C(m,2).
    """
    m = int(np.sum(s > 0.0))
    return 0.5 * m * (m - 1)


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------
def dcgm_probabilities(strengths: np.ndarray, z: float) -> np.ndarray:
    """
    Return the dcGM probability matrix for an undirected simple graph.

    Parameters
    ----------
    strengths:
        Nonnegative fitness proxies s_i, shape (N,).
    z:
        Nonnegative scalar parameter.

    Returns
    -------
    P:
        Symmetric matrix of link probabilities, shape (N,N), with zero diagonal.
    """
    s = _validate_strengths(strengths)
    if not np.isfinite(z) or z < 0.0:
        raise ValueError("z must be a finite non-negative scalar.")

    n = int(s.size)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 1), dtype=np.float64)

    log_s = _log_strengths(s)
    theta = float(np.log(z)) if z > 0.0 else -np.inf
    P = _probabilities_from_theta(log_s, theta)
    # Diagonal already zero by construction; keep as explicit contract.
    np.fill_diagonal(P, 0.0)
    return P


def dcgm_expected_links(P: np.ndarray) -> float:
    """
    Return the expected number of undirected links from a probability matrix.

    Parameters
    ----------
    P:
        Square probability matrix.

    Returns
    -------
    L_hat:
        Expected link count, sum_{i<j} P_ij.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix.")
    return 0.5 * float(P.sum())


def dcgm_calibrate_z(
    strengths: np.ndarray,
    L_target: float,
    *,
    tol_L: float = 1e-10,
    max_iter: int = 200,
    line_search: bool = True,
    theta0: Optional[float] = None,
) -> float:
    """
    Calibrate z by Newton iterations on theta = log z, matching the expected link count.

    Parameters
    ----------
    strengths:
        Nonnegative strengths s_i.
    L_target:
        Target number of undirected links (a nonnegative scalar).
    tol_L:
        Absolute tolerance on |L_hat - L_target|.
    max_iter:
        Maximum Newton iterations.
    line_search:
        If True, use Armijo backtracking on phi(theta)=0.5*(L_hat-L_target)^2.
    theta0:
        Optional initial guess for theta=log(z). If None, an analytic small-z
        approximation is used to initialize z.

    Returns
    -------
    z:
        Calibrated nonnegative scalar parameter.

    Raises
    ------
    ValueError:
        If L_target is infeasible given strengths (e.g., too large with many zeros).
    RuntimeError:
        If numerical degeneracy is encountered (e.g., derivative collapse).
    """
    if not np.isfinite(L_target) or L_target < 0.0:
        raise ValueError("L_target must be a finite non-negative scalar.")

    s = _validate_strengths(strengths)
    n = int(s.size)

    if n < 2:
        if L_target <= tol_L:
            return 0.0
        raise ValueError("Infeasible L_target for n<2.")

    # Feasibility under zero strengths
    L_max_eff = _effective_L_max_from_strengths(s)
    if L_target <= tol_L:
        return 0.0
    if L_target > L_max_eff + tol_L:
        raise ValueError(
            f"Infeasible L_target={L_target} given strengths: maximum achievable is {L_max_eff} "
            "because nodes with s_i=0 cannot form links."
        )
    if L_target >= L_max_eff - tol_L:
        # For sufficiently large z, p_ij ≈ 1 on the active (s_i>0) subgraph.
        # A very large finite value is returned to avoid infinities in downstream code.
        return 1e300

    log_s = _log_strengths(s)

    # Initialization: small-z approximation p_ij ≈ z s_i s_j  =>  L ≈ z * sum_{i<j} s_i s_j.
    if theta0 is None:
        sum_s = float(s.sum())
        sum_s2 = float((s * s).sum())
        sum_sisj = 0.5 * (sum_s * sum_s - sum_s2)
        if not np.isfinite(sum_sisj) or sum_sisj <= 0.0:
            raise RuntimeError("Sum of strength products is non-positive, dcGM is ill-defined.")
        z0 = max(1e-300, float(L_target) / sum_sisj)
        theta = float(np.log(z0))
    else:
        if not np.isfinite(theta0):
            raise ValueError("theta0 must be finite if provided.")
        theta = float(theta0)

    # Newton iterations on theta
    for _it in range(int(max_iter)):
        L_hat, dL = _Lhat_and_dL(theta, log_s)
        diff = float(L_hat - L_target)

        if abs(diff) <= tol_L:
            return float(np.exp(theta))

        if (not np.isfinite(dL)) or dL <= 0.0:
            raise RuntimeError("Non-positive derivative encountered during dcGM calibration.")

        # Newton step on theta
        step = -diff / float(dL)

        if not line_search:
            theta = float(theta + step)
            continue

        # Armijo backtracking on phi(theta)=0.5*diff^2
        phi_old = 0.5 * diff * diff
        alpha = 1.0
        c1 = 1e-4
        accepted = False

        for _ls in range(40):
            theta_trial = float(theta + alpha * step)
            L_trial, _ = _Lhat_and_dL(theta_trial, log_s)
            diff_trial = float(L_trial - L_target)
            phi_new = 0.5 * diff_trial * diff_trial
            if phi_new <= phi_old * (1.0 - c1 * alpha):
                theta = theta_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # As a stable fallback, accept a strongly damped step.
            theta = float(theta + 1e-3 * step)

    return float(np.exp(theta))


def dcgm_fit(
    strengths: np.ndarray,
    L_target: float,
    *,
    tol_L: float = 1e-10,
    max_iter: int = 200,
    line_search: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Calibrate z and return (z, P).

    This is a convenience wrapper around dcgm_calibrate_z and dcgm_probabilities.
    """
    z = dcgm_calibrate_z(
        strengths,
        L_target,
        tol_L=tol_L,
        max_iter=max_iter,
        line_search=line_search,
    )
    P = dcgm_probabilities(strengths, z)
    return z, P


# -----------------------------------------------------------------------------
# Model container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DcGMResult:
    """Container for dcGM calibration outputs."""
    z: float
    P: np.ndarray
    expected_degrees: np.ndarray
    L_target: float
    L_hat: float
    abs_error_L: float


class DcGMModel:
    """
    Degree-corrected gravity model (dcGM) calibrated from strengths and link count.

    After calling fit(), the calibrated probability matrix is cached and can be
    retrieved via probabilities().
    """

    def __init__(self, strengths: np.ndarray, L_target: float) -> None:
        self.strengths = _validate_strengths(strengths)
        self.L_target = float(L_target)
        if (not np.isfinite(self.L_target)) or self.L_target < 0.0:
            raise ValueError("L_target must be a finite non-negative scalar.")
        self._result: Optional[DcGMResult] = None

    def fit(self, *, tol_L: float = 1e-10, max_iter: int = 200, line_search: bool = True) -> DcGMResult:
        """
        Calibrate z and cache the fitted probability matrix.

        Parameters
        ----------
        tol_L:
            Absolute tolerance on |L_hat - L_target|.
        max_iter:
            Maximum Newton iterations on theta=log(z).
        line_search:
            If True, Armijo backtracking is used for robustness.

        Returns
        -------
        DcGMResult:
            Frozen dataclass with calibration outputs.
        """
        z = dcgm_calibrate_z(
            self.strengths,
            self.L_target,
            tol_L=tol_L,
            max_iter=max_iter,
            line_search=line_search,
        )
        P = dcgm_probabilities(self.strengths, z)
        k_hat = P.sum(axis=1).astype(np.float64)
        L_hat = dcgm_expected_links(P)

        res = DcGMResult(
            z=float(z),
            P=P,
            expected_degrees=k_hat,
            L_target=float(self.L_target),
            L_hat=float(L_hat),
            abs_error_L=float(abs(L_hat - self.L_target)),
        )
        self._result = res
        return res

    def probabilities(self) -> np.ndarray:
        """Return the calibrated probability matrix."""
        if self._result is None:
            raise RuntimeError("Model is not calibrated. Call fit() first.")
        return self._result.P

    def result(self) -> DcGMResult:
        """Return the cached DcGMResult."""
        if self._result is None:
            raise RuntimeError("Model is not calibrated. Call fit() first.")
        return self._result