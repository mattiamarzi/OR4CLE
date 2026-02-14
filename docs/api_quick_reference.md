# API quick reference

This page is a compact index of the objects you typically interact with.

## Top-level imports

```python
from or4cle import BFM_solver, BERM_solver, prior_test, utils
```

## BFM

### Rolling model

```python
rm = BFM_solver.BFMRollingModel(
    agg,
    seed_period,
    analysis_period,
    loader,
    update="observed",           # or "counterfactual"
    cal_window_years=1,
    prior_family="auto",
    rng=None,
    return_degrees=False,
)
res = rm.run()
```

Output:

- `res.steps`: list of dicts, one per rolling step.
  Common keys:
  - `label_t`, `label_t1`
  - `L_true`, `L_hat`
  - `PI`, `JI`, `ACC`
  - (optional) degree diagnostics when `return_degrees=True`

## BERM

### Rolling model

```python
rm = BERM_solver.BERMRollingModel(
    agg,
    seed_period,
    analysis_period,
    loader,
    update="observed",
    cal_window_years=1,
    prior_family="beta",
)
res = rm.run()
```

## Prior fitting helpers

### Fit and KS diagnostics

```python
x, results, fits = prior_test.fit_prior_candidates(
    x_raw,
    kind="z",                # "z" or "p"
    dists=["gamma","lognormal","normal"],
    alpha=0.05,
)
print(prior_test.format_results_table(results, alpha=0.05))
```

### Histogram plot

```python
prior_test.plot_prior_histogram(
    x,
    fits,
    kind="z",
    xlabel_tex=r"$z$",
)
```

## Plot utilities

All plotting helpers live in `or4cle.utils`. The most commonly used are:

- `plot_prior_from_window(rm, start_label, end_label, ...)`
- `plot_L_true_vs_Lhat(agg, steps, ...)`
- `plot_errors_RE_L_and_MRE_k(agg, steps, ...)`
- `plot_PI_JI_ACC(agg, steps, ...)`
- `plot_parity_L_and_k(agg, steps, ...)`
