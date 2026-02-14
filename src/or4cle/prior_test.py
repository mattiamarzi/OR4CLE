#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# prior_test.py

"""
Prior fit and Kolmogorov–Smirnov (KS) goodness-of-fit testing for empirical priors.

This script is designed to validate prior families used in:
- BFM (z > 0): Gamma, Lognormal, Normal (on z)
- BERM (p in (0,1)): Beta, Normal (on p)
and optionally a user-provided custom distribution.

Given a 1D sample x, the program:
1) Fits selected candidate distributions (with MLE or method-of-moments fallbacks).
2) Computes the one-sample two-sided KS statistic D and an approximate p-value.
3) Reports which candidates pass the KS test at a chosen significance level alpha.
4) Produces two plots:
   (i)  Histogram with fitted PDF overlays (when a PDF is available),
   (ii) Empirical CDF (ECDF) with fitted CDF overlays.

Important methodological note
-----------------------------
If parameters are estimated from the same data used for the KS test, the classical KS
p-values are not exact (the null distribution differs, e.g., Lilliefors-type issues).
This script follows the same pragmatic approach as in your notebook: it reports an
exact p-value for D_n only when SciPy is available (kstwo), otherwise an asymptotic
approximation. Interpret the p-values as heuristic diagnostics.

Custom distribution interface
-----------------------------
Provide --custom /path/to/custom_dist.py. The module must define:

    def fit(data: np.ndarray) -> dict:
        # returns a parameter dictionary

    def cdf(x: np.ndarray, params: dict) -> np.ndarray:
        # returns CDF evaluated at x

Optional:

    def pdf(x: np.ndarray, params: dict) -> np.ndarray:
        # returns PDF evaluated at x (for histogram overlays)

    def n_params(params: dict) -> int:
        # returns the number of free parameters, otherwise len(params) is used.

Usage examples
--------------
# BFM-style (z > 0) from a CSV column:
python prior_test.py --kind z --input z_samples.csv --column z

# BERM-style (p in (0,1)) from a NumPy array:
python prior_test.py --kind p --input p_samples.npy

# Select distributions explicitly:
python prior_test.py --kind z --input z.npy --dists gamma,lognormal,normal

# Add a custom distribution:
python prior_test.py --kind z --input z.npy --custom ./mydist.py

# Save plots:
python prior_test.py --kind z --input z.npy --save-prefix ./images/weekly/z_prior
"""

from __future__ import annotations

import argparse
import warnings
import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

# Matplotlib is used only for plotting; if unavailable, plotting is disabled gracefully.
try:  # pragma: no cover
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    plt = None
    _HAVE_MPL = False

# SciPy accelerates MLE fitting and exact KS p-values; the script has no hard SciPy dependency.
try:  # pragma: no cover
    import scipy.stats as st  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    st = None
    _HAVE_SCIPY = False

# Pandas is optional and is used only to read CSV/parquet robustly.
try:  # pragma: no cover
    import pandas as pd  # type: ignore
    _HAVE_PANDAS = True
except Exception:  # pragma: no cover
    pd = None
    _HAVE_PANDAS = False


# =============================================================================
# Numerical helpers: erf approximation, Phi, ECDF, KS p-value approximation
# =============================================================================
def _erf_vec(x: np.ndarray) -> np.ndarray:
    """
    Vectorized approximation to erf(x) using Abramowitz–Stegun 7.1.26.

    This approximation provides a stable, SciPy-free evaluation of the Normal CDF.
    """
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    t = 1.0 / (1.0 + p * np.abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return s * y


def Phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF evaluated via an erf approximation (SciPy-free)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _erf_vec(x / math.sqrt(2.0)))


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical CDF evaluated at sample points (right-continuous step convention).

    Returns
    -------
    xs:
        Sorted sample.
    Fy:
        Values i/n for i=1..n corresponding to xs.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    Fy = np.arange(1, n + 1, dtype=float) / float(n)
    return xs, Fy


def ks_distance(xs_sorted: np.ndarray, F_model_at_xs: np.ndarray) -> float:
    """
    Two-sided one-sample KS distance computed on the sample grid.

    The statistic is max(D_plus, D_minus), where:
      D_plus  = sup_i (i/n - F(x_i)),
      D_minus = sup_i (F(x_i) - (i-1)/n).
    """
    xs_sorted = np.asarray(xs_sorted, dtype=float)
    F_model_at_xs = np.asarray(F_model_at_xs, dtype=float)
    n = xs_sorted.size
    i = np.arange(1, n + 1, dtype=float)
    D_plus = np.max(i / n - F_model_at_xs)
    D_minus = np.max(F_model_at_xs - (i - 1.0) / n)
    return float(max(D_plus, D_minus))


def ks_pvalue_asymp(D: float, n: int, *, terms: int = 200, stephens: bool = True) -> float:
    """
    Asymptotic p-value approximation for the one-sample two-sided KS test.

    The approximation is based on the Kolmogorov tail series:
      P( sqrt(n) D >= lam ) = 2 sum_{k>=1} (-1)^{k-1} exp(-2 k^2 lam^2)

    Stephens' correction improves finite-sample accuracy.
    """
    if D <= 0.0 or n <= 0:
        return 1.0
    n = int(n)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D if stephens else math.sqrt(n) * D
    lam2 = lam * lam
    s = 0.0
    for k in range(1, int(terms) + 1):
        s += ((-1.0) ** (k - 1)) * math.exp(-2.0 * (k * k) * lam2)
    p = 2.0 * s
    return float(np.clip(p, 0.0, 1.0))


# =============================================================================
# Special functions: regularized incomplete gamma and beta (SciPy-free fallbacks)
# =============================================================================
def _gammainc_P_vec(a: float, x: np.ndarray, *, tol: float = 1e-12, itmax: int = 200) -> np.ndarray:
    """
    Regularized lower incomplete gamma P(a, x) for a>0, x>=0.

    The routine combines a power series (small x) and a continued fraction (large x),
    following standard Numerical Recipes constructions.
    """
    a = float(a)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    m_pos = x > 0.0
    if not np.any(m_pos):
        return out

    x_pos = x[m_pos]
    m_small = x_pos < (a + 1.0)
    m_large = ~m_small

    # Series expansion for P(a, x) when x < a+1
    if np.any(m_small):
        xs = x_pos[m_small]
        term = np.ones_like(xs) / a
        summ = term.copy()
        ap = a
        for _ in range(int(itmax)):
            ap += 1.0
            term *= xs / ap
            summ_new = summ + term
            if np.all(np.abs(term) < np.abs(summ_new) * tol):
                summ = summ_new
                break
            summ = summ_new
        log_pref = -xs + a * np.log(xs) - math.lgamma(a)
        out_pos_small = np.exp(log_pref) * summ
        out[m_pos] = 0.0
        out[m_pos][m_small] = out_pos_small  # type: ignore[index]

    # Continued fraction for Q(a, x) when x >= a+1, then P = 1-Q
    if np.any(m_large):
        xl = x_pos[m_large]
        FPMIN = 1e-300
        b = xl + 1.0 - a
        c = np.full_like(xl, 1.0 / FPMIN)
        d = 1.0 / b
        h = d.copy()

        for i in range(1, int(itmax) + 1):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            d = np.where(np.abs(d) < FPMIN, FPMIN, d)
            c = b + an / c
            c = np.where(np.abs(c) < FPMIN, FPMIN, c)
            d = 1.0 / d
            delta = d * c
            h *= delta
            if np.all(np.abs(delta - 1.0) < tol):
                break

        log_pref_Q = -xl + (a - 1.0) * np.log(xl) - math.lgamma(a)
        Q = np.exp(log_pref_Q) * h
        P = 1.0 - Q
        out[m_pos][m_large] = P  # type: ignore[index]

    return np.clip(out, 0.0, 1.0)


def _betacf(a: float, b: float, x: float, *, itmax: int = 200, eps_cf: float = 3e-14) -> float:
    """
    Continued fraction for the incomplete beta function (Numerical Recipes form).

    The output is the continued fraction value used in I_x(a,b) evaluation.
    """
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, int(itmax) + 1):
        m2 = 2 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delh = d * c
        h *= delh

        if abs(delh - 1.0) < eps_cf:
            break

    return float(h)


def betainc_reg(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta I_x(a,b) for 0<=x<=1.

    The implementation uses symmetry to improve accuracy in the upper tail.
    """
    x = float(x)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lnB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    # Evaluate using the more stable branch based on x relative to mean proxy.
    if x < (a + 1.0) / (a + b + 2.0):
        cf = _betacf(a, b, x)
        return float(math.exp(a * math.log(x) + b * math.log(1.0 - x) - lnB) * cf / a)
    cf = _betacf(b, a, 1.0 - x)
    return float(1.0 - math.exp(b * math.log(1.0 - x) + a * math.log(x) - lnB) * cf / b)


# =============================================================================
# Distribution abstractions
# =============================================================================
FitKind = Literal["z", "p", "real"]


@dataclass
class FitResult:
    """Container for fitted parameters and KS diagnostics."""
    name: str
    n: int
    params: Dict[str, float]
    ks_D: float
    ks_p: float
    passed: bool


class Distribution:
    """Minimal interface for a candidate distribution used in the prior test."""

    name: str

    def fit(self, x: np.ndarray) -> Dict[str, float]:
        """Estimate distribution parameters from the input sample."""
        raise NotImplementedError

    def cdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Evaluate the distribution CDF at x."""
        raise NotImplementedError

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> Optional[np.ndarray]:
        """Evaluate the distribution PDF at x when available; otherwise return None."""
        return None

    def format_params(self, params: Dict[str, float]) -> str:
        """Return a compact string representation of parameters for console output."""
        return ", ".join(f"{k}={v:.6g}" for k, v in params.items())


class GammaDist(Distribution):
    name = "gamma"

    def fit(self, x: np.ndarray) -> Dict[str, float]:
        """
        Estimate Gamma parameters for positive data using a robust MoM estimator.

        Notes
        -----
        The method-of-moments estimator is numerically stable and is used as the
        default. When SciPy is available and the sample size is sufficiently
        large, an MLE refinement is attempted with ``loc`` fixed to zero;
        otherwise the MoM estimate is returned.
        """
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            raise ValueError("Empty sample passed to GammaDist.fit().")

        m = float(np.mean(x))
        v = float(np.var(x))
        if not np.isfinite(m) or (m <= 0.0):
            raise ValueError("GammaDist.fit() requires strictly positive data.")

        # Stabilize the MoM variance for nearly-degenerate samples.
        v = float(max(v, 1e-12 * m * m))
        k_mom = (m * m) / v
        theta_mom = v / m

        # For small samples, MoM is preferable and avoids numerical issues in SciPy.
        if (not _HAVE_SCIPY) or (x.size < 10):
            return {"k": float(k_mom), "theta": float(theta_mom), "loc": 0.0}

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a_hat, loc_hat, scale_hat = st.gamma.fit(x, floc=0.0)
            k_hat = float(a_hat)
            theta_hat = float(scale_hat)
            loc_hat = float(loc_hat)
            ok = (
                np.isfinite(k_hat) and np.isfinite(theta_hat) and np.isfinite(loc_hat)
                and (k_hat > 0.0) and (theta_hat > 0.0)
                and abs(loc_hat) < 1e-10
            )
            if ok:
                return {"k": k_hat, "theta": theta_hat, "loc": loc_hat}
        except Exception:
            pass

        return {"k": float(k_mom), "theta": float(theta_mom), "loc": 0.0}

    def cdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        k = float(params["k"])
        theta = float(params["theta"])
        loc = float(params.get("loc", 0.0))
        if _HAVE_SCIPY:
            return st.gamma.cdf(x, k, loc=loc, scale=theta)  # type: ignore[union-attr]
        xx = np.maximum(x - loc, 0.0) / max(theta, 1e-300)
        return _gammainc_P_vec(k, xx)

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> Optional[np.ndarray]:
        """Evaluate the Gamma PDF with numerically stable log-density arithmetic.

        This implementation avoids SciPy underflow/overflow for extreme shape/scale
        values by evaluating the log-density and renormalizing on the requested grid.
        The renormalization is intended for visualization (histogram overlays).
        """
        x = np.asarray(x, dtype=float)
        k = float(params["k"])
        theta = float(params["theta"])
        loc = float(params.get("loc", 0.0))

        y = x - loc
        out = np.zeros_like(y)
        m = (y > 0.0) & np.isfinite(y)
        if (not np.any(m)) or (k <= 0.0) or (theta <= 0.0):
            return out

        ym = y[m]
        # Log-density: (k-1)log(y) - y/theta - k log(theta) - log(Gamma(k))
        logpdf = (k - 1.0) * np.log(ym) - ym / theta - k * math.log(theta) - math.lgamma(k)

        # Stabilize and renormalize on the provided grid to obtain a well-scaled curve.
        maxlog = float(np.max(logpdf))
        pdf_tilde = np.exp(np.clip(logpdf - maxlog, -745.0, 50.0))

        # Numerical renormalization (for plotting): integral(pdf) ≈ 1.
        xm = x[m]
        order = np.argsort(xm)
        xm_s = xm[order]
        pdf_s = pdf_tilde[order]
        area = float(np.trapz(pdf_s, xm_s))
        if area > 0.0 and np.isfinite(area):
            pdf_s = pdf_s / area
        # Undo sorting and store.
        pdf_unsorted = np.empty_like(pdf_s)
        pdf_unsorted[order] = pdf_s
        out[m] = pdf_unsorted
        return out
    def format_params(self, params: Dict[str, float]) -> str:
        return f"k={params['k']:.6g}, theta={params['theta']:.6g}"


class LogNormalDist(Distribution):
    name = "lognormal"

    def fit(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0.0)]
        if x.size < 2:
            raise RuntimeError("Lognormal fit requires at least two positive observations.")
        y = np.log(x)
        mu = float(np.mean(y))
        sig = float(np.std(y, ddof=0))
        sig = max(sig, 1e-12)
        return {"mu": mu, "sigma": sig, "loc": 0.0}

    def cdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = float(params["mu"])
        sig = float(params["sigma"])
        out = np.zeros_like(x)
        m = x > 0.0
        if sig <= 0.0:
            out[m] = (x[m] >= math.exp(mu)).astype(float)
            return out
        out[m] = Phi((np.log(x[m]) - mu) / sig)
        return out

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> Optional[np.ndarray]:
        x = np.asarray(x, dtype=float)
        mu = float(params["mu"])
        sig = float(params["sigma"])
        out = np.zeros_like(x)
        m = (x > 0.0) & (sig > 0.0)
        if not np.any(m):
            return out
        xm = x[m]
        out[m] = (1.0 / (xm * sig * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((np.log(xm) - mu) / sig) ** 2)
        return out

    def format_params(self, params: Dict[str, float]) -> str:
        return f"mu={params['mu']:.6g}, sigma={params['sigma']:.6g}  (on log x)"


class NormalDist(Distribution):
    name = "normal"

    def fit(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            raise RuntimeError("Normal fit requires at least two observations.")
        mu = float(np.mean(x))
        sig = float(np.std(x, ddof=0))
        sig = max(sig, 1e-12)
        return {"mu": mu, "sigma": sig}

    def cdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = float(params["mu"])
        sig = float(params["sigma"])
        if sig <= 0.0:
            return (x >= mu).astype(float)
        return Phi((x - mu) / sig)

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> Optional[np.ndarray]:
        x = np.asarray(x, dtype=float)
        mu = float(params["mu"])
        sig = float(params["sigma"])
        if sig <= 0.0:
            return np.zeros_like(x)
        return (1.0 / (sig * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    def format_params(self, params: Dict[str, float]) -> str:
        return f"mu={params['mu']:.6g}, sigma={params['sigma']:.6g}"


class BetaDist(Distribution):
    name = "beta"

    def fit(self, x: np.ndarray) -> Dict[str, float]:
        """Fit a Beta distribution on x in (0,1), with robust fallbacks."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            raise RuntimeError("Beta fit requires at least two observations.")
        if np.any((x <= 0.0) | (x >= 1.0)):
            raise RuntimeError("Beta fit requires data strictly in (0,1). Use --clip-eps for endpoints.")

        # Method-of-moments baseline.
        m = float(np.mean(x))
        v = float(np.var(x, ddof=0))
        v_max = float(0.9999 * m * (1.0 - m))
        v = float(max(1e-12, min(v, v_max)))
        t = m * (1.0 - m) / v - 1.0
        alpha_mom = float(max(1e-12, m * t))
        beta_mom = float(max(1e-12, (1.0 - m) * t))

        if (not _HAVE_SCIPY) or x.size < 8:
            return {"alpha": alpha_mom, "beta": beta_mom, "loc": 0.0, "scale": 1.0}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a_hat, b_hat, loc_hat, scale_hat = st.beta.fit(x, floc=0.0, fscale=1.0)  # type: ignore[union-attr]
            a_hat = float(a_hat)
            b_hat = float(b_hat)
            loc_hat = float(loc_hat)
            scale_hat = float(scale_hat)
            ok = (
                np.isfinite(a_hat) and np.isfinite(b_hat) and np.isfinite(loc_hat) and np.isfinite(scale_hat)
                and (a_hat > 0.0) and (b_hat > 0.0)
                and abs(loc_hat) < 1e-10 and abs(scale_hat - 1.0) < 1e-10
            )
            if ok:
                return {"alpha": a_hat, "beta": b_hat, "loc": loc_hat, "scale": scale_hat}
        except Exception:
            pass

        return {"alpha": alpha_mom, "beta": beta_mom, "loc": 0.0, "scale": 1.0}

    def cdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        a = float(params["alpha"])
        b = float(params["beta"])
        if _HAVE_SCIPY:
            return st.beta.cdf(x, a, b, loc=0.0, scale=1.0)  # type: ignore[union-attr]
        out = np.zeros_like(x)
        m = (x > 0.0) & (x < 1.0)
        if np.any(m):
            out[m] = np.array([betainc_reg(a, b, float(xx)) for xx in x[m]], dtype=float)
        out[x >= 1.0] = 1.0
        return out

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> Optional[np.ndarray]:
        x = np.asarray(x, dtype=float)
        a = float(params["alpha"])
        b = float(params["beta"])
        if _HAVE_SCIPY:
            return st.beta.pdf(x, a, b, loc=0.0, scale=1.0)  # type: ignore[union-attr]
        out = np.zeros_like(x)
        m = (x > 0.0) & (x < 1.0)
        if not np.any(m):
            return out
        lnB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        out[m] = np.exp((a - 1.0) * np.log(x[m]) + (b - 1.0) * np.log(1.0 - x[m]) - lnB)
        return out

    def format_params(self, params: Dict[str, float]) -> str:
        return f"alpha={params['alpha']:.6g}, beta={params['beta']:.6g}"


class CustomDist(Distribution):
    name = "custom"

    def __init__(self, module_path: str) -> None:
        self.module_path = module_path
        self._fit_fn: Callable[[np.ndarray], Dict[str, float]]
        self._cdf_fn: Callable[[np.ndarray, Dict[str, float]], np.ndarray]
        self._pdf_fn: Optional[Callable[[np.ndarray, Dict[str, float]], np.ndarray]] = None
        self._npar_fn: Optional[Callable[[Dict[str, float]], int]] = None
        self._load_module()

    def _load_module(self) -> None:
        """Load a user module and extract the required callables."""
        path = os.path.abspath(self.module_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Custom distribution file not found: {path}")

        spec = importlib.util.spec_from_file_location("prior_test_custom", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import custom module: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        if not hasattr(mod, "fit") or not callable(mod.fit):
            raise RuntimeError("Custom module must define a callable fit(data) -> dict.")
        if not hasattr(mod, "cdf") or not callable(mod.cdf):
            raise RuntimeError("Custom module must define a callable cdf(x, params) -> array.")

        self._fit_fn = mod.fit  # type: ignore[assignment]
        self._cdf_fn = mod.cdf  # type: ignore[assignment]

        if hasattr(mod, "pdf") and callable(mod.pdf):
            self._pdf_fn = mod.pdf  # type: ignore[assignment]
        if hasattr(mod, "n_params") and callable(mod.n_params):
            self._npar_fn = mod.n_params  # type: ignore[assignment]

    def fit(self, x: np.ndarray) -> Dict[str, float]:
        pars = self._fit_fn(np.asarray(x, dtype=float))
        if not isinstance(pars, dict):
            raise RuntimeError("custom.fit must return a dict of parameters.")
        # Enforce float parameters for consistent reporting.
        out: Dict[str, float] = {}
        for k, v in pars.items():
            out[str(k)] = float(v)
        return out

    def cdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        y = self._cdf_fn(np.asarray(x, dtype=float), params)
        y = np.asarray(y, dtype=float)
        return np.clip(y, 0.0, 1.0)

    def pdf(self, x: np.ndarray, params: Dict[str, float]) -> Optional[np.ndarray]:
        if self._pdf_fn is None:
            return None
        y = self._pdf_fn(np.asarray(x, dtype=float), params)
        return np.asarray(y, dtype=float)

    def n_params(self, params: Dict[str, float]) -> int:
        if self._npar_fn is not None:
            return int(self._npar_fn(params))
        return int(len(params))

    def format_params(self, params: Dict[str, float]) -> str:
        keys = sorted(params.keys())
        return ", ".join(f"{k}={params[k]:.6g}" for k in keys)


# =============================================================================
# Data loading
# =============================================================================
def _load_vector_from_path(path: str, *, column: Optional[str], key: Optional[str]) -> np.ndarray:
    """
    Load a 1D numeric sample from disk.

    Supported formats:
    - .npy: NumPy array
    - .npz: NumPy archive (use --key to select an array, otherwise take the first)
    - .csv/.txt: CSV or whitespace text (prefer pandas when available)
    - .parquet: parquet table (requires pandas + pyarrow/fastparquet), select --column
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=float).reshape(-1)

    if ext == ".npz":
        zf = np.load(path)
        if key is not None:
            if key not in zf:
                raise KeyError(f"Key {key!r} not in npz archive.")
            return np.asarray(zf[key], dtype=float).reshape(-1)
        # Deterministic choice: take the first key in sorted order.
        k0 = sorted(zf.files)[0]
        return np.asarray(zf[k0], dtype=float).reshape(-1)

    if ext in {".csv", ".txt"}:
        if _HAVE_PANDAS:
            df = pd.read_csv(path)  # type: ignore[union-attr]
            if column is None:
                # Deterministic: use the first numeric column.
                num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
                if not num_cols:
                    raise RuntimeError("No numeric columns found in CSV. Specify --column.")
                column_eff = num_cols[0]
            else:
                column_eff = column
            return np.asarray(df[column_eff].values, dtype=float).reshape(-1)
        # Minimal fallback for headerless numeric text.
        arr = np.loadtxt(path, dtype=float, delimiter="," if ext == ".csv" else None)
        return np.asarray(arr, dtype=float).reshape(-1)

    if ext == ".parquet":
        if not _HAVE_PANDAS:
            raise RuntimeError("Reading parquet requires pandas. Install pandas + pyarrow/fastparquet.")
        if column is None:
            raise RuntimeError("For parquet input, specify --column.")
        df = pd.read_parquet(path, columns=[column])  # type: ignore[union-attr]
        return np.asarray(df[column].values, dtype=float).reshape(-1)

    raise RuntimeError(f"Unsupported input format: {ext}")


def _load_vector(args: argparse.Namespace) -> np.ndarray:
    """Load the input sample either from file, CLI values, or stdin."""
    if args.values is not None and len(args.values) > 0:
        # Accept comma or whitespace separated values.
        s = " ".join(args.values).replace(",", " ")
        arr = np.fromstring(s, sep=" ", dtype=float)
        return np.asarray(arr, dtype=float).reshape(-1)

    if args.input is not None:
        return _load_vector_from_path(args.input, column=args.column, key=args.key)

    # Stdin fallback
    raw = np.fromstring(os.sys.stdin.read(), sep=" ", dtype=float)
    return np.asarray(raw, dtype=float).reshape(-1)


# =============================================================================
# Reporting and plotting
# =============================================================================

# -----------------------------------------------------------------------------
# Default plotting policy
# -----------------------------------------------------------------------------
#
# The OR4CLE notebooks typically want a single histogram panel for the prior
# diagnostic. The standalone CLI, however, is useful for producing both
# histogram and ECDF diagnostics.
#
# The helper functions below therefore honor a module-level flag that can be
# toggled by notebooks without requiring changes elsewhere in the codebase.

PLOT_ECDF_DEFAULT: bool = True


def set_plot_options(*, plot_ecdf: bool = True) -> None:
    """Set module-level default plot options.

    Parameters
    ----------
    plot_ecdf:
        If False, the ECDF+CDF diagnostic panel is suppressed by default.
    """
    global PLOT_ECDF_DEFAULT
    PLOT_ECDF_DEFAULT = bool(plot_ecdf)


def _format_table(rows: List[FitResult], *, alpha: float) -> str:
    """
    Format results into a fixed-width console table.
    """
    headers = ["family", "n", "params", "KS_D", "KS_p", f"pass(p>{alpha:g})"]
    lines: List[List[str]] = []
    for r in rows:
        lines.append([
            r.name,
            str(r.n),
            r.params_str,  # type: ignore[attr-defined]
            f"{r.ks_D:.6g}",
            f"{r.ks_p:.6g}",
            "OK" if r.passed else "REJECT",
        ])

    # Compute column widths.
    widths = [len(h) for h in headers]
    for row in lines:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    sep = "  "
    out_lines = []
    out_lines.append(sep.join(h.ljust(widths[j]) for j, h in enumerate(headers)))
    out_lines.append(sep.join("-" * widths[j] for j in range(len(headers))))
    for row in lines:
        out_lines.append(sep.join(row[j].ljust(widths[j]) for j in range(len(headers))))
    return "\n".join(out_lines)


def _plot_hist_with_pdfs(
    x: np.ndarray,
    fits: Sequence[Tuple[Distribution, Dict[str, float]]],
    *,
    kind: FitKind,
    title: str,
    save_path: Optional[str],
) -> None:
    """
    Plot a histogram of the sample with PDF overlays for candidates providing a PDF.
    """
    if not _HAVE_MPL:
        return

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # Robust bin selection (Freedman–Diaconis) with safe fallback.
    n = x.size
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = float(q3 - q1)
    if n > 1 and iqr > 0.0:
        bw = 2.0 * iqr * (n ** (-1.0 / 3.0))
        bw = max(bw, 1e-6)
        nbins = int(np.clip((float(np.max(x)) - float(np.min(x))) / bw, 10, 80))
    else:
        nbins = 30

    ax.hist(x, bins=nbins, density=True, alpha=0.7, edgecolor="black", label="sample histogram")

    if kind == "p":
        # Adaptive plotting window: zoom to the central mass of p.
        q_lo, q_hi = np.quantile(x, [0.005, 0.995])
        span = float(q_hi - q_lo)

        # If p is extremely concentrated, enforce a minimum visible span.
        min_span = 0.02
        span_eff = max(span, min_span)

        mid = 0.5 * float(q_lo + q_hi)
        lo = max(1e-6, mid - 0.55 * span_eff)
        hi = min(1.0 - 1e-6, mid + 0.55 * span_eff)
        if hi <= lo:
            lo, hi = 1e-6, 1.0 - 1e-6

        x_grid = np.linspace(lo, hi, 500)

        # Force the histogram view to match the zoom.
        ax.set_xlim(float(lo), float(hi))
    else:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        pad = 0.25 * (x_max - x_min + 1e-12)
        lo = max(1e-12, x_min - pad) if kind == "z" else (x_min - pad)
        hi = x_max + pad
        x_grid = np.linspace(lo, hi, 600)

    for dist, pars in fits:
        pdf = dist.pdf(x_grid, pars)
        if pdf is None:
            continue
        ax.plot(x_grid, pdf, linewidth=2.0, label=f"{dist.name}: {dist.format_params(pars)}")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, frameon=True, loc="best")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    plt.show()


# =============================================================================
# Notebook-oriented helper API (histogram-only prior plot)
# =============================================================================
def fit_prior_candidates(
    x_raw: np.ndarray,
    *,
    kind: FitKind,
    dists: Sequence[str],
    alpha: float,
    clip_eps: float = 1e-12,
    custom_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[FitResult], List[Tuple[Distribution, Dict[str, float]]]]:
    """
    Fit candidate priors and compute KS diagnostics.

    This helper mirrors the CLI logic but is intended for notebook usage.
    It is robust to fitting failures: candidates that cannot be fitted are skipped.
    If verbose=True, failures are printed.

    Returns
    -------
    x:
        Preprocessed sample (after support enforcement and optional clipping).
    results_sorted:
        List of FitResult sorted by decreasing KS p-value and increasing KS D.
    fits_for_plot:
        List of (Distribution, params) pairs suitable for plotting overlays.
    """
    x = _preprocess_sample(np.asarray(x_raw, dtype=float), kind=kind, clip_eps=float(clip_eps))
    if x.size < 2:
        return x, [], []

    candidates = _make_candidates(list(dists), custom_path)

    results: List[FitResult] = []
    fits_for_plot: List[Tuple[Distribution, Dict[str, float]]] = []

    for dist in candidates:
        x_use = np.asarray(x, dtype=float)

        # Enforce per-distribution support constraints deterministically.
        if dist.name in {"gamma", "lognormal"}:
            x_use = x_use[x_use > 0.0]
        if dist.name == "beta":
            x_use = x_use[(x_use > 0.0) & (x_use < 1.0)]

        if x_use.size < 2:
            continue

        try:
            pars = dist.fit(x_use)
        except Exception as e:
            if verbose:
                print(f"[prior_test] skipping {dist.name}: fit failed ({type(e).__name__}: {e})")
            continue

        xs, _Fy_emp = ecdf(x_use)
        Fm = dist.cdf(xs, pars)
        D = ks_distance(xs, Fm)

        if _HAVE_SCIPY:
            try:
                pval = float(st.kstwo.sf(D, xs.size))  # type: ignore[union-attr]
            except Exception:
                pval = ks_pvalue_asymp(D, xs.size)
        else:
            pval = ks_pvalue_asymp(D, xs.size)

        passed = bool(pval > float(alpha))
        r = FitResult(name=dist.name, n=int(xs.size), params=pars, ks_D=float(D), ks_p=float(pval), passed=passed)
        setattr(r, "params_str", dist.format_params(pars))
        results.append(r)
        fits_for_plot.append((dist, pars))

    results_sorted = sorted(results, key=lambda rr: (-rr.ks_p, rr.ks_D, rr.name))
    return x, results_sorted, fits_for_plot


def format_results_table(rows: List[FitResult], *, alpha: float) -> str:
    """Public wrapper for the CLI table formatter."""
    return _format_table(rows, alpha=float(alpha))


def plot_prior_histogram(
    x: np.ndarray,
    fits: Sequence[Tuple[Distribution, Dict[str, float]]],
    *,
    kind: FitKind,
    xlabel_tex: str,
    save_path: Optional[str] = None,
    legend_fontsize: int = 14,
) -> None:
    """
    Histogram-only prior plot with the same style used in the legacy notebook.

    The overlay line styles are chosen deterministically by distribution name:
    gamma: '-', normal: '--', lognormal: ':', beta: '-'.
    """
    if not _HAVE_MPL:
        return

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    # Grid consistent with the legacy plot.
    if kind == "p":
        q_lo, q_hi = np.quantile(x, [0.005, 0.995])
        span = float(q_hi - q_lo)
        min_span = 0.02
        span_eff = max(span, min_span)

        mid = 0.5 * float(q_lo + q_hi)
        lo = max(1e-6, mid - 0.55 * span_eff)
        hi = min(1.0 - 1e-6, mid + 0.55 * span_eff)
        if hi <= lo:
            lo, hi = 1e-6, 1.0 - 1e-6

        x_grid = np.linspace(lo, hi, 500)
        ax.set_xlim(float(lo), float(hi))
    else:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        pad = 0.25 * (x_max - x_min + 1e-12)
        lo = max(1e-12, x_min - pad) if kind == "z" else (x_min - pad)
        hi = x_max + pad
        x_grid = np.linspace(lo, hi, 400)

    style_map = {
        "gamma": ("-", 2.0),
        "normal": ("--", 2.0),
        "lognormal": (":", 2.0),
        "beta": ("-", 2.0),
        "custom": ("-", 2.0),
    }

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    # Robust bin selection (Freedman–Diaconis) with safe fallback.
    n = x.size
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = float(q3 - q1)
    if n > 1 and iqr > 0.0:
        bw = 2.0 * iqr * (n ** (-1.0 / 3.0))
        bw = max(bw, 1e-6)
        nbins = int(np.clip((float(np.max(x)) - float(np.min(x))) / bw, 10, 80))
    else:
        nbins = 30

    ax.hist(x, bins=nbins, density=True, alpha=0.7, edgecolor="black")

    for dist, pars in fits:
        pdf = dist.pdf(x_grid, pars)
        if pdf is None:
            continue
        ls, lw = style_map.get(dist.name, ("-", 2.0))

        # Parameter strings in LaTeX-like notation, matching the legacy labels.
        if dist.name == "gamma":
            lab = fr"Gamma fit: $k={pars['k']:.3f}$, $\theta={pars['theta']:.3f}$"
        elif dist.name == "normal":
            lab = fr"Normal fit: $\mu={pars['mu']:.3f}$, $\sigma={pars['sigma']:.3f}$"
        elif dist.name == "lognormal":
            lab = fr"Lognormal fit: $\mu={pars['mu']:.3f}$, $\sigma={pars['sigma']:.3f}$"
        elif dist.name == "beta":
            lab = fr"Beta fit: $\alpha={pars['alpha']:.3f}$, $\beta={pars['beta']:.3f}$"
        else:
            lab = f"{dist.name}: {dist.format_params(pars)}"

        ax.plot(x_grid, pdf, linewidth=lw, linestyle=ls, label=lab)

    ax.set_xlabel(xlabel_tex, fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=legend_fontsize, frameon=True, loc="best")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    plt.show()


def _plot_ecdf_with_cdfs(
    x: np.ndarray,
    fits: Sequence[Tuple[Distribution, Dict[str, float]]],
    *,
    title: str,
    save_path: Optional[str],
) -> None:
    """
    Plot ECDF of the sample with candidate CDF overlays.
    """
    # Notebook-facing workflows often prefer a single histogram panel.
    # This early return allows suppressing the ECDF diagnostic without
    # changing call sites (e.g., utils.plot_prior_from_window).
    if not PLOT_ECDF_DEFAULT:
        return
    if not _HAVE_MPL:
        return

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    xs, Fy = ecdf(x)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    ax.plot(xs, Fy, drawstyle="steps-post", linewidth=2.0, label="ECDF")

    for dist, pars in fits:
        Fm = dist.cdf(xs, pars)
        ax.plot(xs, Fm, linewidth=2.0, label=f"{dist.name}: {dist.format_params(pars)}")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, frameon=True, loc="best")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    plt.show()


# =============================================================================
# Main execution
# =============================================================================
def _preprocess_sample(x: np.ndarray, *, kind: FitKind, clip_eps: float) -> np.ndarray:
    """
    Preprocess the sample to match the declared support.

    - kind="z": enforce strict positivity.
    - kind="p": enforce [0,1] with optional endpoint clipping.
    - kind="real": finite values only.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    if kind == "z":
        x = x[x > 0.0]
        return x

    if kind == "p":
        x = x[(x >= 0.0) & (x <= 1.0)]
        if clip_eps is not None and clip_eps > 0.0:
            x = np.clip(x, float(clip_eps), 1.0 - float(clip_eps))
        return x

    return x


def _make_candidates(dists: List[str], custom_path: Optional[str]) -> List[Distribution]:
    """Instantiate distribution objects in a deterministic order."""
    out: List[Distribution] = []
    for d in dists:
        dd = d.strip().lower()
        if dd == "gamma":
            out.append(GammaDist())
        elif dd == "lognormal":
            out.append(LogNormalDist())
        elif dd == "normal":
            out.append(NormalDist())
        elif dd == "beta":
            out.append(BetaDist())
        elif dd == "custom":
            if custom_path is None:
                raise RuntimeError("Requested 'custom' but --custom was not provided.")
            out.append(CustomDist(custom_path))
        else:
            raise ValueError(f"Unknown distribution: {d!r}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit candidate priors and run KS tests.")
    parser.add_argument("--kind", choices=["z", "p", "real"], required=True, help="Support of the data (z>0, p in (0,1), or real).")

    parser.add_argument("--input", type=str, default=None, help="Path to input data (.npy/.npz/.csv/.txt/.parquet). If omitted, reads from stdin.")
    parser.add_argument("--column", type=str, default=None, help="Column name for CSV/parquet input.")
    parser.add_argument("--key", type=str, default=None, help="Array key for NPZ input.")
    parser.add_argument("--values", nargs="*", default=None, help="Inline numeric values (comma or space separated).")

    parser.add_argument("--dists", type=str, default=None,
                        help="Comma-separated list of distributions to test. "
                             "Allowed: gamma,lognormal,normal,beta,custom. "
                             "If omitted, defaults depend on --kind.")
    parser.add_argument("--custom", type=str, default=None, help="Path to a custom distribution module (see docstring).")

    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for KS acceptance.")
    parser.add_argument("--clip-eps", type=float, default=1e-9,
                        help="Endpoint clipping epsilon for p-kind to avoid {0,1} issues in bounded fits.")

    parser.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    parser.add_argument("--save-prefix", type=str, default=None,
                        help="If provided, saves plots to '<prefix>_hist.pdf' and '<prefix>_cdf.pdf'.")

    args = parser.parse_args()

    kind: FitKind = args.kind

    # Select defaults consistent with your requested families.
    if args.dists is None:
        if kind == "z":
            dists = ["gamma", "lognormal", "normal"]
        elif kind == "p":
            dists = ["beta", "normal"]
        else:
            dists = ["normal"]
        if args.custom is not None:
            dists.append("custom")
    else:
        dists = [s.strip() for s in args.dists.split(",") if s.strip()]
        if args.custom is not None and "custom" not in [x.lower() for x in dists]:
            dists.append("custom")

    # Load and preprocess data.
    x_raw = _load_vector(args)
    x = _preprocess_sample(x_raw, kind=kind, clip_eps=float(args.clip_eps))

    if x.size == 0:
        raise RuntimeError("No usable data after preprocessing. Check --kind and input values.")

    # Fit candidates and compute KS diagnostics on each candidate’s feasible sample.
    candidates = _make_candidates(dists, args.custom)
    alpha = float(args.alpha)

    results: List[FitResult] = []
    fitted_for_plots: List[Tuple[Distribution, Dict[str, float]]] = []

    for dist in candidates:
        # Enforce per-distribution support constraints deterministically.
        x_use = np.asarray(x, dtype=float)
        if dist.name in {"gamma", "lognormal"}:
            x_use = x_use[x_use > 0.0]
        if dist.name == "beta":
            x_use = x_use[(x_use > 0.0) & (x_use < 1.0)]

        if x_use.size < 2:
            continue

        pars = dist.fit(x_use)

        xs, _Fy_emp = ecdf(x_use)
        Fm = dist.cdf(xs, pars)
        D = ks_distance(xs, Fm)

        if _HAVE_SCIPY:
            # Exact tail probability for D_n (note the caveat for estimated parameters).
            try:
                pval = float(st.kstwo.sf(D, xs.size))  # type: ignore[union-attr]
            except Exception:
                pval = ks_pvalue_asymp(D, xs.size)
        else:
            pval = ks_pvalue_asymp(D, xs.size)

        passed = bool(pval > alpha)

        r = FitResult(name=dist.name, n=int(xs.size), params=pars, ks_D=float(D), ks_p=float(pval), passed=passed)
        # Attach a preformatted param string for table printing.
        setattr(r, "params_str", dist.format_params(pars))
        results.append(r)
        fitted_for_plots.append((dist, pars))

    # Sort output by decreasing p-value (most plausible first), tie-break by smaller D.
    results_sorted = sorted(results, key=lambda rr: (-rr.ks_p, rr.ks_D, rr.name))

    print(f"\nInput summary: kind={kind}, n={x.size}, scipy={'yes' if _HAVE_SCIPY else 'no'}, matplotlib={'yes' if _HAVE_MPL else 'no'}")
    print(f"KS threshold: alpha={alpha:g}\n")

    if len(results_sorted) == 0:
        raise RuntimeError("No candidates could be fitted on the provided sample/support.")

    print(_format_table(results_sorted, alpha=alpha))

    passed = [r.name for r in results_sorted if r.passed]
    if passed:
        print("\nPassed KS:", ", ".join(passed))
    else:
        print("\nPassed KS: (none)")

    # Plotting
    if (not args.no_plot) and _HAVE_MPL:
        save_hist = f"{args.save_prefix}_hist.pdf" if args.save_prefix else None
        save_cdf = f"{args.save_prefix}_cdf.pdf" if args.save_prefix else None
        _plot_hist_with_pdfs(
            x=x,
            fits=fitted_for_plots,
            kind=kind,
            title="Sample histogram with fitted PDFs (where available)",
            save_path=save_hist,
        )
        _plot_ecdf_with_cdfs(
            x=x,
            fits=fitted_for_plots,
            title="Empirical CDF with fitted CDFs",
            save_path=save_cdf,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())