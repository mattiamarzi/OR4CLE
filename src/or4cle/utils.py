#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# modules/utils.py

"""
High-level plotting and reporting utilities for ORBIT rolling experiments.

This module centralizes the notebook-facing plotting/printing logic so that
the notebooks remain minimal and rely primarily on the solver modules.

The plotting functions reproduce the exact definitions and styles used in the
original weekly notebook, including axis formatting, markers, sizes, and
save-to-disk conventions.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import prior_test
from . import BFM_solver
from . import BERM_solver


Array = np.ndarray


# =============================================================================
# Small helpers
# =============================================================================
def ensure_images_dir(agg: str, *, base_dir: str = "./images") -> str:
    """
    Ensure the image directory exists and return its path.

    Parameters
    ----------
    agg:
        Aggregation level string (e.g., "weekly", "monthly").
    base_dir:
        Base directory for storing figures.

    Returns
    -------
    images_dir:
        The created directory path.
    """
    images_dir = os.path.join(base_dir, str(agg))
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def _as_float_list(values: Iterable[Any]) -> List[float]:
    """Convert an iterable to a list of float with NaN for missing values."""
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return out


def _middle_label(labels: Sequence[str]) -> Optional[str]:
    """Return the middle label of a sequence, or None if empty."""
    if not labels:
        return None
    return labels[len(labels) // 2]


def _pick_three(labels: Sequence[str]) -> List[str]:
    """Select three representative labels (first, middle, last) from a list."""
    if not labels:
        return []
    if len(labels) <= 3:
        return list(labels)
    i0, i1, i2 = 0, len(labels) // 2, len(labels) - 1
    return [labels[i] for i in sorted({i0, i1, i2})]


# =============================================================================
# Data extraction from rolling outputs
# =============================================================================
@dataclass(frozen=True)
class PlotInputs:
    """
    Container for the minimal structures required by the plotting routines.
    """
    pair_metrics: List[Dict[str, Any]]
    t1_labels: List[str]
    dates: List[datetime]
    L_true_list: List[float]
    L_hat_list: List[float]
    RE_L_list: List[float]
    MRE_k_list: List[float]
    PI_list: List[float]
    JI_list: List[float]
    ACC_soft_list: List[float]
    degrees_by_label_t1: Dict[str, Tuple[Array, Array]]


def build_plot_inputs(agg: str, steps: Sequence[Mapping[str, Any]]) -> PlotInputs:
    """
    Build the exact list/dict objects used by the plotting cells.

    Parameters
    ----------
    agg:
        Aggregation level.
    steps:
        Rolling step dictionaries (BFM/BERM rolling result).

    Returns
    -------
    PlotInputs:
        Extracted lists and dictionaries.
    """
    pair_metrics = [dict(s) for s in steps]  # keep compatibility name

    t1_labels = [str(m.get("label_t1")) for m in pair_metrics]
    dates = [BFM_solver.label_to_datetime(agg, lab) for lab in t1_labels]

    L_true_list   = _as_float_list(m.get("L_true", np.nan) for m in pair_metrics)
    L_hat_list    = _as_float_list(m.get("L_hat_sumq", np.nan) for m in pair_metrics)
    RE_L_list     = _as_float_list(m.get("RE_L", np.nan) for m in pair_metrics)
    MRE_k_list    = _as_float_list(m.get("MRE_k", np.nan) for m in pair_metrics)
    PI_list       = _as_float_list(m.get("PI", np.nan) for m in pair_metrics)
    JI_list       = _as_float_list(m.get("JI", np.nan) for m in pair_metrics)
    ACC_soft_list = _as_float_list(m.get("ACC_soft", np.nan) for m in pair_metrics)

    degrees_by_label_t1: Dict[str, Tuple[Array, Array]] = {}
    for m in pair_metrics:
        lab = m.get("label_t1", None)
        kt = m.get("k_true", None)
        kh = m.get("k_hat", None)
        if (lab is None) or (kt is None) or (kh is None):
            continue
        degrees_by_label_t1[str(lab)] = (np.asarray(kt, float), np.asarray(kh, float))

    return PlotInputs(
        pair_metrics=pair_metrics,
        t1_labels=t1_labels,
        dates=dates,
        L_true_list=L_true_list,
        L_hat_list=L_hat_list,
        RE_L_list=RE_L_list,
        MRE_k_list=MRE_k_list,
        PI_list=PI_list,
        JI_list=JI_list,
        ACC_soft_list=ACC_soft_list,
        degrees_by_label_t1=degrees_by_label_t1,
    )


def _date_axis_formatters(dates: Sequence[datetime]) -> Tuple[mdates.DateLocator, mdates.DateFormatter]:
    """
    Reproduce the notebook's date tick logic (year ticks if span >= 3 years).
    """
    years_in_span = {d.year for d in dates}
    span_years = (max(dates) - min(dates)).days / 365.25
    if span_years >= 3:
        major_loc = mdates.YearLocator()
        major_fmt = mdates.DateFormatter("%Y")
    else:
        major_loc = mdates.MonthLocator(interval=1)
        major_fmt = mdates.DateFormatter("%b" if len(years_in_span) == 1 else "%b %Y")
    return major_loc, major_fmt


# =============================================================================
# Prior plotting (via prior_test)
# =============================================================================
def _fit_candidates_for_plot(
    x: Array,
    *,
    kind: str,
    dists: Sequence[str],
    alpha: float,
) -> Tuple[List[Tuple[prior_test.Distribution, Dict[str, float]]], Dict[str, float]]:
    """
    Fit candidates from prior_test and compute KS diagnostics for each.

    Returns
    -------
    fits_for_plots:
        List of (Distribution, fitted_params).
    diag:
        Dictionary with the best (by p-value) distribution diagnostic.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if kind == "z":
        x = x[x > 0.0]
    elif kind == "p":
        x = x[(x > 0.0) & (x < 1.0)]

    candidates: List[prior_test.Distribution] = []
    for d in dists:
        dd = d.strip().lower()
        if dd == "gamma":
            candidates.append(prior_test.GammaDist())
        elif dd == "lognormal":
            candidates.append(prior_test.LogNormalDist())
        elif dd == "normal":
            candidates.append(prior_test.NormalDist())
        elif dd == "beta":
            candidates.append(prior_test.BetaDist())
        else:
            raise ValueError(f"Unknown distribution for prior plotting: {d!r}")

    results: List[Dict[str, float]] = []
    fits_for_plots: List[Tuple[prior_test.Distribution, Dict[str, float]]] = []

    for dist in candidates:
        x_use = np.asarray(x, dtype=float)
        if dist.name in {"gamma", "lognormal"}:
            x_use = x_use[x_use > 0.0]
        if dist.name == "beta":
            x_use = x_use[(x_use > 0.0) & (x_use < 1.0)]

        if x_use.size < 2:
            continue

        pars = dist.fit(x_use)
        xs, _Fy = prior_test.ecdf(x_use)
        Fm = dist.cdf(xs, pars)
        D = float(prior_test.ks_distance(xs, Fm))

        # p-value: use SciPy exact survival when available, else asymptotic approximation.
        if getattr(prior_test, "_HAVE_SCIPY", False):
            try:
                import scipy.stats as st  # type: ignore
                pval = float(st.kstwo.sf(D, xs.size))
            except Exception:
                pval = float(prior_test.ks_pvalue_asymp(D, xs.size))
        else:
            pval = float(prior_test.ks_pvalue_asymp(D, xs.size))

        results.append({"name": float(0.0), "ks_D": D, "ks_p": pval})  # placeholder for sorting
        fits_for_plots.append((dist, pars))

    # Best by p-value then smaller D.
    best = {"ks_D": float("nan"), "ks_p": float("nan")}
    if fits_for_plots:
        scored = []
        for dist, pars in fits_for_plots:
            xs, _Fy = prior_test.ecdf(x)
            Fm = dist.cdf(xs, pars)
            D = float(prior_test.ks_distance(xs, Fm))
            if getattr(prior_test, "_HAVE_SCIPY", False):
                try:
                    import scipy.stats as st  # type: ignore
                    pval = float(st.kstwo.sf(D, xs.size))
                except Exception:
                    pval = float(prior_test.ks_pvalue_asymp(D, xs.size))
            else:
                pval = float(prior_test.ks_pvalue_asymp(D, xs.size))
            scored.append((pval, -D, dist.name, D))
        scored.sort(reverse=True)
        p_best, _negD, name_best, D_best = scored[0]
        best = {"best_name": name_best, "ks_D": float(D_best), "ks_p": float(p_best), "alpha": float(alpha)}
    return fits_for_plots, best


def plot_prior_from_window(
    rm: Any,
    *,
    start_label: Optional[str] = None,
    end_label: str,
    alpha: float = 0.05,
    save_fig: bool = True,
    images_dir: Optional[str] = None,
    title_prefix: Optional[str] = None,
    plot_ecdf: bool = True,
    return_table: bool = False,
) -> Dict[str, float]:
    """
    Plot the empirical prior sample in a window ending at end_label.

    By default the window is the rolling calibration window as in the models.
    If start_label is provided, the window is forced to be the fixed period
    start_label..end_label (useful for checking the entire SEED_PERIOD).

    The function delegates distribution fitting and plotting to the mechanisms used
    in prior_test.py (histogram with PDF overlays and ECDF with CDF overlays).

    Parameters
    ----------
    rm:
        A BFMRollingModel or BERMRollingModel instance.
    start_label:
        Optional left endpoint of a fixed window. If None, uses the rolling
        calibration window implied by rm.cal_window_years.
    end_label:
        Right endpoint of the window.
    alpha:
        KS significance level (reported for context).
    save_fig:
        If True, saves the prior plots into images_dir with deterministic names.
    images_dir:
        Directory where figures are saved. If None, uses ./images/<agg>.
    title_prefix:
        Optional prefix for plot titles.

    plot_ecdf:
        If False, suppresses the ECDF+CDF diagnostic panel.

    return_table:
        If True, enriches the returned dictionary with a formatted KS table and
        the pass/reject sets.

    Returns
    -------
    diag:
        Dictionary with the best candidate diagnostic (name, D, p).
    """
    agg = getattr(rm, "agg")
    images_dir = ensure_images_dir(agg) if images_dir is None else images_dir

    # Collect window samples by reusing the rolling model's own caching methods.
    if hasattr(rm, "_fit_prior_from_window"):
        # Use the rolling window logic: extract raw samples (z or p) from caches.
        # The required private fields/methods exist in both rolling model classes.
        mod = BFM_solver if hasattr(rm, "_get_z_samples_observed") else BERM_solver

        if start_label is None:
            labels_cal = mod._labels_cal_window(agg, end_label, rm.seed_start, rm.cal_window_years)  # type: ignore[attr-defined]
        else:
            labels_cal = mod.labels_in_period(agg, start_label, end_label)  # type: ignore[attr-defined]

        vals: List[float] = []
        if hasattr(rm, "_get_z_samples_observed"):
            kind = "z"
            for lbl in labels_cal:
                if rm.update == "observed":
                    zs = rm._get_z_samples_observed(lbl)  # type: ignore[attr-defined]
                else:
                    zs = rm._z_pool_by_label.get(lbl, [])  # type: ignore[attr-defined]
                if zs:
                    vals.extend([float(z) for z in zs])
            dists = ["gamma", "lognormal", "normal"]
        else:
            kind = "p"
            for lbl in labels_cal:
                if rm.update == "observed":
                    ps = rm._get_p_samples_observed(lbl)  # type: ignore[attr-defined]
                else:
                    ps = rm._p_pool_by_label.get(lbl, [])  # type: ignore[attr-defined]
                if ps:
                    vals.extend([float(p) for p in ps])
            dists = ["beta", "normal"]

        x = np.asarray(vals, dtype=float)
    else:
        raise RuntimeError("Provided object does not look like a rolling model instance.")

    if kind == "z":
        x = x[np.isfinite(x)]
        x = x[x > 0.0]
    else:
        x = x[np.isfinite(x)]
        x = x[(x > 0.0) & (x < 1.0)]

    if x.size < 2:
        return {"best_name": "none", "ks_D": np.nan, "ks_p": np.nan, "alpha": float(alpha)}

    fits_for_plots, diag = _fit_candidates_for_plot(x, kind=kind, dists=dists, alpha=float(alpha))

    # Delegate to prior_test plotting helpers for consistent visuals.
    prefix = "" if title_prefix is None else f"{title_prefix} - "
    title_hist = f"{prefix}Prior window @ {end_label} (kind={kind}, n={x.size})"
    if save_fig:
        save_prefix = os.path.join(images_dir, f"{agg}_prior_window_{end_label}".replace("/", "_"))
        save_hist = f"{save_prefix}_hist.pdf"
        save_cdf  = f"{save_prefix}_cdf.pdf"
    else:
        save_hist = None
        save_cdf = None

    # Configure prior_test plotting policy for this call. This avoids exposing
    # additional plotting flags in each call site while allowing notebooks to
    # suppress the ECDF diagnostic panel.
    prev_plot_ecdf = getattr(prior_test, "PLOT_ECDF_DEFAULT", True)
    if hasattr(prior_test, "set_plot_options"):
        prior_test.set_plot_options(plot_ecdf=bool(plot_ecdf))  # type: ignore[attr-defined]

    prior_test._plot_hist_with_pdfs(  # type: ignore[attr-defined]
        x=x,
        fits=fits_for_plots,
        kind=kind,  # type: ignore[arg-type]
        title=title_hist,
        save_path=save_hist,
    )
    prior_test._plot_ecdf_with_cdfs(  # type: ignore[attr-defined]
        x=x,
        fits=fits_for_plots,
        title=f"{prefix}Prior window ECDF @ {end_label}",
        save_path=save_cdf,
    )

    # Restore default plotting policy.
    if hasattr(prior_test, "set_plot_options"):
        prior_test.set_plot_options(plot_ecdf=bool(prev_plot_ecdf))  # type: ignore[attr-defined]

    if return_table:
        # Recompute KS summary in a consistent printable form.
        # The candidate list is fixed by kind.
        _, rows, _ = prior_test.fit_prior_candidates(
            x,
            kind=kind,  # type: ignore[arg-type]
            dists=dists,
            alpha=float(alpha),
            clip_eps=1e-12,
            verbose=False,
        )
        table = prior_test.format_results_table(rows, alpha=float(alpha))
        passed = [r.name for r in rows if r.passed]
        diag = dict(diag)
        diag["table"] = table
        diag["passed"] = passed
        diag["rejected"] = [r.name for r in rows if not r.passed]
        diag["best_name"] = rows[0].name if rows else diag.get("best_name", "none")

    return diag


# =============================================================================
# Plots: EXACT style and definitions as requested
# =============================================================================
def plot_L_true_vs_Lhat(
    agg: str,
    steps: Sequence[Mapping[str, Any]],
    *,
    save_fig: bool = True,
    images_dir: Optional[str] = None,
) -> None:
    """
    Plot time series of L_true and <L> with the exact style of the reference cell.
    """
    P = build_plot_inputs(agg, steps)
    images_dir = ensure_images_dir(agg) if images_dir is None else images_dir

    dates = P.dates
    major_loc, major_fmt = _date_axis_formatters(dates)

    L_true_ts = np.asarray(P.L_true_list, dtype=float)
    L_hat_ts  = np.asarray(P.L_hat_list,  dtype=float)

    if len(dates) != len(L_true_ts) or len(dates) != len(L_hat_ts):
        raise RuntimeError("Lengths of dates, L_true_list, and L_hat_list do not match.")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.plot(dates, L_true_ts, marker="o", linewidth=2.0, markersize=4, label=r"$L_{\mathrm{true}}$")
    ax.plot(dates, L_hat_ts,  marker="s", linewidth=2.0, markersize=4, label=r"$\langle L\rangle$")
    ax.xaxis.set_major_locator(major_loc)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.set_xlim(min(dates), max(dates))
    ax.set_xlabel("Month", fontsize=18)
    ax.set_ylabel(r"Number of links", fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=18, frameon=True, loc="best")
    fig.autofmt_xdate(rotation=45, ha="right")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{images_dir}/{agg}_L_true_vs_Lhat.pdf", dpi=300)
    plt.show()


def plot_parity_L_and_k(
    agg: str,
    steps: Sequence[Mapping[str, Any]],
    *,
    mode: str = "pooled",
    label_single: Optional[str] = "2002-W45",
    labels_subset: Optional[Sequence[str]] = None,
    save_fig: bool = True,
    images_dir: Optional[str] = None,
) -> None:
    """
    Combined parity plot: L (top-x vs left-y) + degrees (bottom-x vs right-y).

    Parameters mirror the notebook variables:
    - mode: "single", "multi", or "pooled"
    - label_single: used only when mode="single"
    - labels_subset: used for "multi" or as a subset for "pooled"
    """
    P = build_plot_inputs(agg, steps)
    images_dir = ensure_images_dir(agg) if images_dir is None else images_dir

    t1_labels = [m.get("label_t1") for m in P.pair_metrics]
    degrees_by_label_t1 = P.degrees_by_label_t1

    # Ensure parity data is available:
    L_true_arr = np.asarray(P.L_true_list, dtype=float)
    L_hat_arr  = np.asarray(P.L_hat_list,  dtype=float)

    L_max = float(max(L_true_arr.max(), L_hat_arr.max())) if L_true_arr.size and L_hat_arr.size else 1.0
    L_lim = max(1.0, L_max) * 1.05

    def _collect_single(label: Optional[str]):
        if (label is None) or (label not in degrees_by_label_t1):
            label = _middle_label([str(x) for x in t1_labels if x is not None])
        kt, kh = degrees_by_label_t1[str(label)]
        return [str(label)], [np.asarray(kt, float)], [np.asarray(kh, float)]

    def _collect_multi(labels: Optional[Sequence[str]]):
        all_labs = [str(x) for x in t1_labels if x is not None]
        if not labels:
            chosen = _pick_three(all_labs)
        else:
            chosen = [lab for lab in labels if lab in degrees_by_label_t1]
            if not chosen:
                chosen = _pick_three(all_labs)
        kts, khs = [], []
        for lab in chosen:
            kt, kh = degrees_by_label_t1[lab]
            kts.append(np.asarray(kt, float))
            khs.append(np.asarray(kh, float))
        return chosen, kts, khs

    def _collect_pooled(labels_subset: Optional[Sequence[str]]):
        if labels_subset:
            use = [lab for lab in labels_subset if lab in degrees_by_label_t1]
        else:
            use = list(degrees_by_label_t1.keys())
        if not use:
            raise RuntimeError("No labels available for pooled degree parity plot.")
        kts, khs = [], []
        for lab in use:
            kt, kh = degrees_by_label_t1[lab]
            kts.append(np.asarray(kt, float))
            khs.append(np.asarray(kh, float))
        kt_all = np.concatenate(kts)
        kh_all = np.concatenate(khs)
        return use, [kt_all], [kh_all]

    if mode == "single":
        chosen_labels, ktrues_list, khats_list = _collect_single(label_single)
        save_suffix = "single"
    elif mode == "multi":
        chosen_labels, ktrues_list, khats_list = _collect_multi(labels_subset)
        save_suffix = "multi"
    elif mode == "pooled":
        chosen_labels, ktrues_list, khats_list = _collect_pooled(labels_subset)
        save_suffix = f"pooled_{len(chosen_labels)}snap"
    else:
        raise ValueError("mode must be 'single', 'multi', or 'pooled'.")

    if not chosen_labels:
        raise RuntimeError("No labels available for the degree parity plot.")

    deg_x_lim = 1.05 * max(1.0, max((float(np.max(kt)) for kt in ktrues_list if kt.size), default=1.0))
    deg_y_lim = 1.05 * max(1.0, max((float(np.max(kh)) for kh in khats_list if kh.size), default=1.0))
    deg_lim = max(deg_x_lim, deg_y_lim)

    fig, ax_base = plt.subplots(figsize=(10, 8), dpi=300)

    ax_base.set_xlim(0, deg_lim)
    ax_base.set_xlabel(r"$k_i^{\mathrm{true}}$", fontsize=20, color="tab:blue")
    ax_base.tick_params(axis="x", labelsize=20, colors="tab:blue")

    ax_base.set_ylim(0, L_lim)
    ax_base.set_ylabel(r"$\langle L\rangle$", fontsize=20, color="tab:red")
    ax_base.tick_params(axis="y", labelsize=20, colors="tab:red")

    ax_right = ax_base.twinx()
    ax_right.set_ylim(0, deg_lim)
    ax_right.set_ylabel(r"$\langle k_i\rangle$", fontsize=20, color="tab:blue")
    ax_right.tick_params(axis="y", labelsize=20, colors="tab:blue")

    ax_top = ax_base.twiny()
    ax_top.set_xlim(0, L_lim)
    ax_top.set_xlabel(r"$L_{\mathrm{true}}$", fontsize=20, color="tab:red")
    ax_top.tick_params(axis="x", labelsize=20, colors="tab:red")

    sL = ax_top.scatter(L_true_arr, L_hat_arr, s=28, alpha=0.85,
                        edgecolor="black", linewidth=0.6, color="tab:red",
                        label="L", rasterized=True)

    if len(ktrues_list) == 1:
        kt, kh = ktrues_list[0], khats_list[0]
        sK = ax_right.scatter(kt, kh, s=16, alpha=0.6, edgecolor="black", linewidth=0.4,
                              color="tab:blue", marker="o", label="k", rasterized=True)
    else:
        markers = ["o", "s", "^"]
        handles_tmp = []
        for idx, (lab, kt, kh) in enumerate(zip(chosen_labels, ktrues_list, khats_list)):
            h = ax_right.scatter(kt, kh, s=22, alpha=0.85, edgecolor="black", linewidth=0.6,
                                 color="tab:blue", marker=markers[idx % len(markers)], label=lab, rasterized=True)
            handles_tmp.append(h)
        sK = handles_tmp[0]

    ax_top.plot([0, L_lim], [0, L_lim], ls="--", lw=1.5, color="k", zorder=1)
    ax_right.plot([0, deg_lim], [0, deg_lim], ls="--", lw=1.5, color="k", zorder=1)

    yx_handle = Line2D([0], [0], ls="--", lw=1.5, color="k")
    handles = [sL, sK, yx_handle]
    labels  = [r"$L$", r"$k$", r"$y=x$"]
    ax_base.legend(handles, labels, fontsize=20, frameon=True, loc="lower right")

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{images_dir}/{agg}_parity_L_and_k.pdf", dpi=300)
    plt.show()


def plot_errors_RE_L_and_MRE_k(
    agg: str,
    steps: Sequence[Mapping[str, Any]],
    *,
    save_fig: bool = True,
    images_dir: Optional[str] = None,
) -> None:
    """
    Plot time series: RE_L (left, red) and MRE_k (right, blue), with symmetric rolling mean.
    """
    P = build_plot_inputs(agg, steps)
    images_dir = ensure_images_dir(agg) if images_dir is None else images_dir

    dates = P.dates
    major_loc, major_fmt = _date_axis_formatters(dates)

    RE_L_ts  = np.asarray(P.RE_L_list,  dtype=float)
    MRE_k_ts = np.asarray(P.MRE_k_list, dtype=float)

    XWIN = 5
    RE_L_smooth  = BFM_solver.rolling_mean_sym(RE_L_ts,  x=XWIN)
    MRE_k_smooth = BFM_solver.rolling_mean_sym(MRE_k_ts, x=XWIN)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    l1, = ax.plot(dates, RE_L_smooth, marker="^", linewidth=2.0, markersize=4,
                  label=r"$\mathrm{RE}_L$", color="tab:red")
    ax.set_xlabel("Month", fontsize=20)
    ax.set_ylabel(r"$\mathrm{RE}_L$", fontsize=20, color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax.xaxis.set_major_locator(major_loc)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.set_xlim(min(dates), max(dates))

    ax2 = ax.twinx()
    l2, = ax2.plot(dates, MRE_k_smooth, marker="d", linewidth=2.0, markersize=4,
                   label=r"$\mathrm{MRE}_k$", color="tab:blue")
    ax2.set_ylabel(r"$\mathrm{MRE}_k$", fontsize=20, color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    y1_min, y1_max = np.nanmin(RE_L_ts),  np.nanmax(RE_L_ts)
    y2_min, y2_max = np.nanmin(MRE_k_ts), np.nanmax(MRE_k_ts)
    if np.isfinite(y1_min) and np.isfinite(y1_max) and y1_min != y1_max:
        pad = 0.05 * (y1_max - y1_min)
        ax.set_ylim(y1_min - pad, y1_max + pad)
    if np.isfinite(y2_min) and np.isfinite(y2_max) and y2_min != y2_max:
        pad = 0.05 * (y2_max - y2_min)
        ax2.set_ylim(y2_min - pad, y2_max + pad)

    ax.tick_params(axis="both", labelsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, fontsize=20, frameon=True, loc="best")

    fig.autofmt_xdate(rotation=45, ha="right")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{images_dir}/{agg}_RE_L_and_MRE_k_timeseries.pdf", dpi=300)
    plt.show()


def plot_PI_JI_ACC(
    agg: str,
    steps: Sequence[Mapping[str, Any]],
    *,
    save_fig: bool = True,
    images_dir: Optional[str] = None,
) -> None:
    """
    Plot time series: PI, JI, and <ACC> on the same axes with identical style.
    """
    P = build_plot_inputs(agg, steps)
    images_dir = ensure_images_dir(agg) if images_dir is None else images_dir

    dates = P.dates
    major_loc, major_fmt = _date_axis_formatters(dates)

    PI_ts       = np.asarray(P.PI_list,       dtype=float)
    JI_ts       = np.asarray(P.JI_list,       dtype=float)
    ACC_soft_ts = np.asarray(P.ACC_soft_list, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    ax.plot(dates, PI_ts, marker="s", linewidth=2.0, markersize=4,
            label=r"$\mathrm{PI}$", color="tab:orange")
    ax.plot(dates, JI_ts, marker="^", linewidth=2.0, markersize=4,
            label=r"$\mathrm{JI}$", color="tab:green")
    ax.plot(dates, ACC_soft_ts, marker="D", linewidth=2.0, markersize=4,
            label=r"$\langle \mathrm{ACC}\rangle$", color="tab:red")

    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(major_loc)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.set_xlim(min(dates), max(dates))
    ax.set_xlabel("Month", fontsize=18)
    ax.set_ylabel("Score", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=18, frameon=True, loc="best")
    fig.autofmt_xdate(rotation=45, ha="right")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{images_dir}/{agg}_PI_JI_ACC_timeseries.pdf", dpi=300)
    plt.show()


def plot_degree_and_L_histograms(
    agg: str,
    steps: Sequence[Mapping[str, Any]],
    *,
    deg_hist_mode: str = "pooled",
    deg_hist_label: Optional[str] = "2002-W45",
    deg_hist_labels: Optional[Sequence[str]] = None,
    save_fig: bool = True,
    images_dir: Optional[str] = None,
) -> None:
    """
    Histogram of L and degrees (distribution). Kept for completeness.
    """
    P = build_plot_inputs(agg, steps)
    images_dir = ensure_images_dir(agg) if images_dir is None else images_dir

    degrees_by_label_t1 = P.degrees_by_label_t1
    t1_labels = P.t1_labels

    def _collect_degrees_single(label: Optional[str]):
        if label is None or label not in degrees_by_label_t1:
            label = _middle_label(t1_labels)
        k_true_vec, k_hat_vec = degrees_by_label_t1[str(label)]
        return np.asarray(k_true_vec, float), np.asarray(k_hat_vec, float), str(label)

    def _collect_degrees_pooled(labels_to_use: Optional[Sequence[str]]):
        if not labels_to_use:
            labels_to_use = list(degrees_by_label_t1.keys())
        labels_to_use = [lab for lab in labels_to_use if lab in degrees_by_label_t1]
        if not labels_to_use:
            raise RuntimeError("No labels available for pooled degree histogram.")
        k_true_list, k_hat_list = [], []
        for lab in labels_to_use:
            kt, kh = degrees_by_label_t1[lab]
            k_true_list.append(np.asarray(kt, float))
            k_hat_list.append(np.asarray(kh, float))
        return np.concatenate(k_true_list), np.concatenate(k_hat_list), labels_to_use

    if deg_hist_mode == "single":
        k_true_vec, k_hat_vec, _chosen = _collect_degrees_single(deg_hist_label)
        save_stem  = f"{agg}_degree_hist"
    elif deg_hist_mode == "pooled":
        k_true_vec, k_hat_vec, _used_labels = _collect_degrees_pooled(deg_hist_labels)
        save_stem  = f"{agg}_degree_hist_pooled"
    else:
        raise ValueError("deg_hist_mode must be 'single' or 'pooled'.")

    L_true_arr = np.asarray(P.L_true_list, dtype=float)
    L_hat_arr  = np.asarray(P.L_hat_list,  dtype=float)

    k_all = np.concatenate([np.asarray(k_true_vec, float), np.asarray(k_hat_vec, float)])
    k_all = k_all[np.isfinite(k_all)]
    if k_all.size == 0:
        raise RuntimeError("No finite degree values to bin.")
    kmin = int(np.floor(k_all.min()))
    kmax = int(np.ceil (k_all.max()))
    bins_k = np.arange(kmin - 0.5, kmax + 1.5, 1)

    L_all = np.concatenate([L_true_arr, L_hat_arr])
    L_all = L_all[np.isfinite(L_all)]
    if L_all.size == 0:
        raise RuntimeError("No finite L values to bin.")
    Lmin = float(np.floor(L_all.min()))
    Lmax = float(np.ceil (L_all.max()))
    n_bins_L = max(10, int(np.sqrt(L_all.size)))
    bins_L = np.linspace(Lmin, Lmax, n_bins_L)

    LIGHT_BLUE  = "#add8e6"
    DARK_BLUE   = "tab:blue"
    LIGHT_RED   = "#ffb3b3"
    DARK_RED    = "tab:red"

    fig, ax_left = plt.subplots(figsize=(10, 8), dpi=300)

    ax_top   = ax_left.twiny()
    ax_right = ax_left.twinx()

    ax_left.set_xlim(kmin - 0.5, kmax + 0.5)
    ax_top.set_xlim(Lmin, Lmax)

    ax_left.set_xlabel("k", fontsize=20, color=DARK_BLUE)
    ax_left.tick_params(axis="x", labelsize=20, colors=DARK_BLUE)

    ax_left.tick_params(axis="y", labelsize=20, colors=DARK_RED)

    ax_top.set_xlabel(r"$L$", fontsize=20, color=DARK_RED)
    ax_top.tick_params(axis="x", labelsize=20, colors=DARK_RED)

    ax_right.tick_params(axis="y", labelsize=20, colors=DARK_BLUE)

    ax_top.hist(L_true_arr, bins=bins_L, density=True, alpha=0.55, color=LIGHT_RED)
    ax_top.hist(L_hat_arr,  bins=bins_L, density=True, alpha=0.55, color=DARK_RED)

    ax_right.hist(k_true_vec, bins=bins_k, density=True, alpha=0.55, color=LIGHT_BLUE)
    ax_right.hist(k_hat_vec,  bins=bins_k, density=True, alpha=0.55, color=DARK_BLUE)

    legend_handles = [
        Patch(facecolor=LIGHT_BLUE, edgecolor="none", label=r"ePDF $(k)$"),
        Patch(facecolor=DARK_BLUE,  edgecolor="none", label=r"PDF $(k)$"),
        Patch(facecolor=LIGHT_RED,  edgecolor="none", label=r"ePDF $(L)$"),
        Patch(facecolor=DARK_RED,   edgecolor="none", label=r"PDF $(L)$")
    ]
    ax_left.legend(handles=legend_handles, fontsize=20, frameon=True, loc="best")

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{images_dir}/{save_stem}.pdf", dpi=300)
    plt.show()


# =============================================================================
# Pretty printing
# =============================================================================
def _format_kv(k: str, v: Any, width: int = 22) -> str:
    """Format a key-value pair with aligned keys."""
    return f"{k:<{width}}: {v}"


def print_run_report(
    agg: str,
    steps: Sequence[Mapping[str, Any]],
    *,
    model: str,
    mode: str,
    update: str,
    seed_period: Tuple[Any, Any],
    analysis_period: Tuple[Any, Any],
    ks_alpha: float = 0.05,
    avg_window: Optional[Tuple[datetime, datetime]] = (datetime(2002, 2, 1), datetime(2002, 10, 31)),
) -> None:
    """
    Print a compact, visually structured report of the rolling run.

    The report includes:
    - run metadata and span,
    - average errors in a user-specified calendar window (if present),
    - KS summary for Gamma prior (BFM) when available in step dicts,
    - contiguous failure spans for quick inspection.
    """
    P = build_plot_inputs(agg, steps)

    print("\n" + "=" * 78)
    print("ORBIT rolling reconstruction report")
    print("=" * 78)
    print(_format_kv("Model", model))
    print(_format_kv("Aggregation", agg))
    print(_format_kv("Mode", mode))
    print(_format_kv("Update", update))
    print(_format_kv("Seed period", f"{seed_period[0]} → {seed_period[1]}"))
    print(_format_kv("Analysis period", f"{analysis_period[0]} → {analysis_period[1]}"))
    print(_format_kv("n steps", len(P.pair_metrics)))
    if P.t1_labels:
        print(_format_kv("first step", f"{P.pair_metrics[0].get('label_t')} → {P.pair_metrics[0].get('label_t1')}"))
        print(_format_kv("last step",  f"{P.pair_metrics[-1].get('label_t')} → {P.pair_metrics[-1].get('label_t1')}"))

    # Average errors in a date window
    if avg_window is not None and P.dates:
        start_window, end_window = avg_window
        dates_arr = np.asarray(P.dates)
        RE_L_arr  = np.asarray(P.RE_L_list, dtype=float)
        MRE_k_arr = np.asarray(P.MRE_k_list, dtype=float)
        mask = (dates_arr >= start_window) & (dates_arr <= end_window)
        RE_L_sub  = RE_L_arr[mask]
        MRE_k_sub = MRE_k_arr[mask]
        RE_L_avg  = float(np.nanmean(RE_L_sub)) if RE_L_sub.size else float("nan")
        MRE_k_avg = float(np.nanmean(MRE_k_sub)) if MRE_k_sub.size else float("nan")
        print("-" * 78)
        print("Average errors in window")
        print(_format_kv("window", f"{start_window.date()} → {end_window.date()}"))
        print(_format_kv("Mean RE_L",  f"{RE_L_avg:.4f}"))
        print(_format_kv("Mean MRE_k", f"{MRE_k_avg:.4f}"))
        print(_format_kv("n points", int(mask.sum())))

    # KS summary (Gamma prior) if present in step dicts.
    ks_ps = np.array([m.get("KS_gamma_p", np.nan) for m in P.pair_metrics], dtype=float)
    ks_Ds = np.array([m.get("KS_gamma_D", np.nan) for m in P.pair_metrics], dtype=float)
    if np.isfinite(ks_ps).any():
        share_ok = float(np.nanmean(ks_ps > float(ks_alpha)))
        print("-" * 78)
        print("KS test summary (Gamma prior)")
        print(_format_kv("alpha", ks_alpha))
        print(_format_kv("mean D", f"{np.nanmean(ks_Ds):.3f}"))
        print(_format_kv("share p>alpha", f"{share_ok:.2%}"))

        # Failures and contiguous spans
        t1_seq = [str(m.get("label_t1")) for m in P.pair_metrics]
        idx_of = {lab: i for i, lab in enumerate(t1_seq)}
        tested = [m for m in P.pair_metrics if np.isfinite(m.get("KS_gamma_p", np.nan))]
        fails  = [m for m in tested if float(m.get("KS_gamma_p", np.nan)) <= float(ks_alpha)]

        print(_format_kv("failures", f"{len(fails)}/{len(tested)} steps"))

        fails.sort(key=lambda m: idx_of[str(m.get("label_t1"))])
        runs = []
        for m in fails:
            idx = idx_of[str(m.get("label_t1"))]
            if not runs or idx != runs[-1]["end_idx"] + 1:
                runs.append({
                    "start": m.get("label_t"),
                    "end": m.get("label_t1"),
                    "start_idx": idx,
                    "end_idx": idx,
                    "count": 1,
                    "min_p": float(m.get("KS_gamma_p")),
                    "min_p_lab": str(m.get("label_t1")),
                })
            else:
                r = runs[-1]
                r["end"] = m.get("label_t1")
                r["end_idx"] = idx
                r["count"] += 1
                pval = float(m.get("KS_gamma_p"))
                if pval < r["min_p"]:
                    r["min_p"] = pval
                    r["min_p_lab"] = str(m.get("label_t1"))

        if runs:
            print("Failure spans (sorted by length):")
            runs.sort(key=lambda r: r["count"], reverse=True)
            for r in runs[:20]:
                print(f"  • {r['start']}→{r['end']}  (len={r['count']}, min p={r['min_p']:.3g} @ {r['min_p_lab']})")
            if len(runs) > 20:
                print(f"  … {len(runs) - 20} more spans suppressed.")

        by_year = Counter([str(m.get("label_t1")).split("-W")[0] for m in fails])
        if by_year:
            summary = ", ".join([f"{y}:{c}" for y, c in sorted(by_year.items())])
            print("  by year:", summary)

    print("=" * 78 + "\n")
