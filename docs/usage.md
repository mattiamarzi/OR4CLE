# Usage

OR4CLE is designed around a *rolling* evaluation protocol:

1. choose a **seed period** to estimate an empirical prior,
2. choose an **analysis period** to run a sequence of one-step-ahead reconstructions,
3. optionally restrict the prior estimation to a finite **calibration window** (rolling prior).

The repository ships with a ready-to-use synthetic dataset under `datasets/synthetic_networks/`.

## Concepts

### Seed period

The *seed period* is a label interval (inclusive) used to collect historical samples for the prior.

### Calibration window

During the rolling run, the solver may refit the prior at each step using only the last `cal_window_years` years of available seed history.

- `cal_window_years=None` uses all available seed history.
- Smaller values make the prior more local in time.

### Analysis period

The *analysis period* is the label interval over which the rolling experiment is executed.
Each step typically uses $t$ to predict $t+1$.

### Observed vs self-sustained

The rolling update can be performed in two modes:

- `update="observed"`: the next snapshot in the dataset is taken as observation.
- `update="counterfactual"`: the model uses only its own previous reconstruction as the new state (self-sustained / counterfactual).

## Minimal example (BFM)

```python
import json
import numpy as np
from pathlib import Path

from or4cle import BFM_solver, utils

DATA_ROOT = Path("datasets/synthetic_networks")
meta = json.load(open(DATA_ROOT / "metadata.json", "r"))
snap_index = {d["label"]: d for d in meta["snapshots"]}

def load_snapshot(label: str) -> BFM_solver.Snapshot:
    f = DATA_ROOT / "snapshots" / snap_index[label]["file"]
    z = np.load(f)
    return BFM_solver.Snapshot(
        A=z["A"].astype(float),
        s=z["s"].astype(float),
        node_ids=z["node_ids"].astype(np.int64),
    )

rm = BFM_solver.BFMRollingModel(
    agg="monthly",
    seed_period=("2000-01", "2002-12"),
    analysis_period=("2002-12", "2005-12"),
    loader=load_snapshot,
    update="observed",
    cal_window_years=3,
)

res = rm.run()
steps = res.steps

utils.plot_L_true_vs_Lhat("monthly", steps, save_fig=False)
```

## Minimal example (BERM)

```python
import json
import numpy as np
from pathlib import Path

from or4cle import BERM_solver

DATA_ROOT = Path("datasets/synthetic_networks")
meta = json.load(open(DATA_ROOT / "metadata.json", "r"))
snap_index = {d["label"]: d for d in meta["snapshots"]}

def load_snapshot(label: str) -> BERM_solver.Snapshot:
    f = DATA_ROOT / "snapshots" / snap_index[label]["file"]
    z = np.load(f)
    return BERM_solver.Snapshot(
        A=z["A"].astype(float),
        node_ids=z["node_ids"].astype(np.int64),
    )

rm = BERM_solver.BERMRollingModel(
    agg="monthly",
    seed_period=("2000-01", "2002-12"),
    analysis_period=("2002-12", "2005-12"),
    loader=load_snapshot,
    update="observed",
    cal_window_years=3,
)

res = rm.run()
```

## Prior diagnostic with KS test

The plotting utilities expose a notebook-friendly wrapper:

```python
from or4cle import utils

ks = utils.plot_prior_from_window(
    rm,
    start_label="2000-01",
    end_label="2003-12",
    alpha=0.05,
    save_fig=False,
    plot_ecdf=False,
    return_table=True,
)

print(ks["table"])      # formatted table
print(ks["passed"])     # list of families that pass
print(ks["best_name"])  # best (by KS p-value)
```