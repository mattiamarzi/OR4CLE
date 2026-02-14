"""Minimal smoke tests for the OR4CLE rolling solvers.

The tests intentionally run on the *bundled* synthetic dataset shipped with the
repository and do not regenerate any data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    """Return the repository root path from the location of this file."""
    return Path(__file__).resolve().parents[1]


def _synthetic_paths() -> tuple[Path, Path]:
    """Return (metadata_path, snapshots_dir) for the bundled synthetic dataset."""
    root = _repo_root()
    data_root = root / "datasets" / "synthetic_networks"
    meta_path = data_root / "metadata.json"
    snap_dir = data_root / "snapshots"
    assert meta_path.exists(), f"Missing dataset metadata: {meta_path}"
    assert snap_dir.exists(), f"Missing dataset snapshots dir: {snap_dir}"
    return meta_path, snap_dir


def _load_index() -> tuple[dict, list[str]]:
    """Load synthetic dataset metadata and return (index_by_label, labels)."""
    meta_path, _ = _synthetic_paths()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    labels = [d["label"] for d in meta["snapshots"]]
    index = {d["label"]: d for d in meta["snapshots"]}
    return index, labels


def test_bfm_rolling_gh_runs() -> None:
    """BFM rolling model should run and produce at least one step."""
    from or4cle import BFM_solver

    index, labels = _load_index()
    _, snap_dir = _synthetic_paths()

    def loader(label: str) -> BFM_solver.Snapshot:
        z = np.load(snap_dir / index[label]["file"], allow_pickle=False)
        return BFM_solver.Snapshot(
            A=z["A"].astype(np.float64),
            s=z["s"].astype(np.float64),
            node_ids=z["node_ids"].astype(np.int64),
        )

    seed_period = (labels[0], labels[24])
    analysis_period = (labels[24], labels[30])

    rm = BFM_solver.BFMRollingModel(
        agg="monthly",
        seed_period=seed_period,
        analysis_period=analysis_period,
        loader=loader,
        prior_family="auto",
        update="observed",
        cal_window_years=1,
        rng=np.random.default_rng(0),
        # Keep runtime bounded.
        gh_nodes=15,
        gh_max_iter=40,
        gh_tol=1e-10,
        return_Q=False,
        return_degrees=True,
    )

    res = rm.run()
    assert hasattr(res, "steps")
    assert len(res.steps) > 0
    assert "label_t" in res.steps[0]
    assert "label_t1" in res.steps[0]


def test_berm_rolling_gh_runs() -> None:
    """BERM rolling model should run and produce at least one step."""
    from or4cle import BERM_solver

    index, labels = _load_index()
    _, snap_dir = _synthetic_paths()

    def loader(label: str) -> BERM_solver.Snapshot:
        z = np.load(snap_dir / index[label]["file"], allow_pickle=False)
        return BERM_solver.Snapshot(
            A=z["A"].astype(np.float64),
            node_ids=z["node_ids"].astype(np.int64),
        )

    seed_period = (labels[0], labels[24])
    analysis_period = (labels[24], labels[30])

    rm = BERM_solver.BERMRollingModel(
        agg="monthly",
        seed_period=seed_period,
        analysis_period=analysis_period,
        loader=loader,
        prior_family="auto",
        update="observed",
        cal_window_years=1,
        rng=np.random.default_rng(0),
        # Keep runtime bounded.
        gh_nodes=15,
        gh_max_iter=40,
        gh_tol=1e-10,
    )

    res = rm.run()
    assert hasattr(res, "steps")
    assert len(res.steps) > 0
    assert "label_t" in res.steps[0]
    assert "label_t1" in res.steps[0]
