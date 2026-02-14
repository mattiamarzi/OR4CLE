# OR4CLE

OR4CLE - **O**ut-of-sample bayesian **R**ecostruction **4** **C**omp**L**ex n**E**tworks.

This repository provides a reproducible reference implementation of rolling, out-of-sample reconstruction workflows.

## Installation

```bash
pip install or4cle
```

For development:

```bash
pip install -e .
```

## Bundled example dataset

The repository ships a synthetic monthly dataset in `datasets/synthetic_networks/` (already generated). It is used by the notebook in `notebooks/` and by the automated tests.

## Quickstart: BFM rolling experiment

```python
import json
import numpy as np
from pathlib import Path

from or4cle import BFM_solver

DATA_ROOT = Path("datasets/synthetic_networks")
META_PATH = DATA_ROOT / "metadata.json"
SNAP_DIR  = DATA_ROOT / "snapshots"

meta = json.load(open(META_PATH, "r"))
snap_index = {d["label"]: d for d in meta["snapshots"]}

def load_snapshot(label: str) -> BFM_solver.Snapshot:
    z = np.load(SNAP_DIR / snap_index[label]["file"], allow_pickle=False)
    return BFM_solver.Snapshot(
        A=z["A"].astype(float),
        s=z["s"].astype(float),
        node_ids=z["node_ids"].astype(np.int64),
    )

rm = BFM_solver.BFMRollingModel(
    agg="monthly",
    seed_period=("2000-01", "2002-12"),
    analysis_period=("2002-12", "2008-04"),
    loader=load_snapshot,
    update="observed",          # or "counterfactual" for self-sustained
    cal_window_years=3,          # rolling calibration window length
    prior_family="auto",        # choose among supported families
)

res = rm.run()
print(len(res.steps), res.steps[0]["label_t"], "->", res.steps[0]["label_t1"])
```

## Quickstart: BERM rolling experiment

```python
import json
import numpy as np
from pathlib import Path

from or4cle import BERM_solver

DATA_ROOT = Path("datasets/synthetic_networks")
META_PATH = DATA_ROOT / "metadata.json"
SNAP_DIR  = DATA_ROOT / "snapshots"

meta = json.load(open(META_PATH, "r"))
snap_index = {d["label"]: d for d in meta["snapshots"]}

def load_snapshot(label: str) -> BERM_solver.Snapshot:
    z = np.load(SNAP_DIR / snap_index[label]["file"], allow_pickle=False)
    return BERM_solver.Snapshot(
        A=z["A"].astype(float),
        node_ids=z["node_ids"].astype(np.int64),
    )

rm = BERM_solver.BERMRollingModel(
    agg="monthly",
    seed_period=("2000-01", "2002-12"),
    analysis_period=("2002-12", "2008-04"),
    loader=load_snapshot,
    update="observed",
    cal_window_years=3,
    prior_family="auto",
)

res = rm.run()
print(len(res.steps), res.steps[0]["label_t"], "->", res.steps[0]["label_t1"])
```

## Prior diagnostics (fit + KS test) on a seed window

The rolling models fit priors internally. For diagnostics on a chosen seed window you can use `utils.plot_prior_from_window`:

```python
from or4cle import utils

ks_info = utils.plot_prior_from_window(
    rm,
    start_label="2000-01",
    end_label="2003-12",
    alpha=0.05,
    save_fig=False,
    plot_ecdf=False,
    return_table=True,
)

print(ks_info["table"])      # formatted table string
print(ks_info["passed"])     # list of families passing the KS threshold
print(ks_info["best_name"])  # best family by p-value (tie-broken by D)
```

## Documentation

See `docs/`:

- `docs/usage.md`
- `docs/api_quick_reference.md`
- `docs/math.md`

## License

MIT (see `LICENSE`).
