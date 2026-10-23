"""Predeclared software measurements; no real Dataset admission or ranking."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from tools.tier_d_nonrl.tier_d_protocol import (
    REPO,
    file_sha256,
    market_protocol,
    validate_committed_source,
)
from trade_rl.data.market import MarketDataset

PACKAGE = Path(__file__).resolve().parent
CASES = (
    ("n100", 100, False),
    ("n400", 400, False),
    ("n1600", 1600, False),
    ("n1600-profile", 1600, True),
)


def profile_source_pins() -> dict[str, str]:
    return {p.name: file_sha256(p) for p in sorted(PACKAGE.glob("*.py"))}


def validate_profile_source(protocol: dict[str, Any]) -> None:
    """Require committed driver and measurement bytes before artificial fits."""
    validate_committed_source(protocol)
    names = set(profile_source_pins())
    tracked = {
        Path(name.decode()).name
        for name in subprocess.check_output(
            ["git", "ls-files", "-z", "--", "tools/tier_d_scaling/*.py"], cwd=REPO
        ).split(b"\0")
        if name
    }
    if not names or names != tracked:
        raise ValueError("all scaling Python sources must be committed")
    for name in names:
        path = PACKAGE / name
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:tools/tier_d_scaling/{name}"], cwd=REPO
        )
        if path.is_symlink() or hashlib.sha256(committed).hexdigest() != file_sha256(
            path
        ):
            raise ValueError("scaling source differs from committed bytes")


def profile_protocol(dataset: MarketDataset, dataset_root: Path) -> dict[str, Any]:
    if (
        (dataset.n_bars, dataset.n_symbols, dataset.n_features) != (9504, 2, 6)
        or dataset.symbols != ("ARTIFICIAL_A", "ARTIFICIAL_B")
        or dataset.timestamps[0] != np.datetime64("2000-01-01", "ns")
        or not dataset.identity_verified
    ):
        raise ValueError("requires the full artificial Dataset")
    base = market_protocol(max_workers=1)

    def timestamp(index: int) -> str:
        return str(dataset.timestamps[index])

    return {
        "schema": "issue888_artificial_scaling_v3",
        "scope": "ARTIFICIAL_SOFTWARE_PROFILE_ONLY",
        "source_head": base["source_head"],
        "concurrency": {"max_workers": 1},
        "progress_policy": base["progress_policy"],
        "dataset": {
            "root": str(dataset_root.resolve()),
            "dataset_id": dataset.dataset_id,
            "input_sha256": {
                n: file_sha256(dataset_root / n)
                for n in ("manifest.json", "arrays.npz")
            },
            "dimensions": [9504, 2, 6],
            "feature_names": list(dataset.feature_names),
            "fit_symbols": list(dataset.symbols),
            "development_start": timestamp(0),
            "development_stop_exclusive": str(
                dataset.timestamps[-1] + np.timedelta64(1, "h")
            ),
            "unused_status": "ARTIFICIAL_ONLY",
        },
        "configurations": [
            {"id": "H72_RIDGE_A10", "horizon_hours": 72, "ridge_alpha": 10.0}
        ],
        "accounts": ["ARTIFICIAL_H72_A10_A"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
        "capital": base["capital"],
        "actor": {
            "start": 2903,
            "stop": 9430,
            "rows": 6527,
            "vintage_starts": [2903, 5087, 7295],
        },
        "terminal": {"stop_inclusive": 9502, "advances": 72},
        "economics": base["economics"],
        "funding": {
            "source": "artificial eight-hour events",
            "marks": "rate times artificial close",
            "coverage_bounds": base["funding"]["coverage_bounds"],
        },
        "policy": base["policy"],
        "budget": {"native_arms": 2, "ridge_vintages": 3, "advances_per_arm": 6599},
        "measurement": {
            "execution_order": ["n1600-profile", "n100", "n400", "n1600"],
            "cases": [
                {"name": name, "target": target, "profile": profile}
                for name, target, profile in CASES
            ],
            "limits": {
                "wall_seconds": 10_800,
                "rss_bytes": 2 * 1024**3,
                "disk_bytes": 4 * 1024**3,
                "sample_seconds": 10,
            },
            "financial_authority": False,
            "profile_checkpoint_first": 1,
            "profile_checkpoint_every": 100,
            "profile_checkpoint_semantics": "atomic exclusive cProfile snapshots without disabling active contexts; unfinished enclosing frames may be absent; profile IO overhead included in profiled timing only",
            "semantics": "intentional failed native prefixes of one identical full plan; no reset/step substitute",
            "coverage": "all measured advances are actor/vintage one; no reserve or vintage transition measurement",
            "repetitions": 1,
            "limitations": "single observations cannot establish complexity order, variance, full-matrix feasibility, or a real-data bound",
            "oracle": "same-HEAD same-plan profiled/unprofiled n1600 partial receipt canonical bytes; historical release equivalence unestablished",
        },
    }


def case_by_name(protocol: dict[str, Any], name: object) -> dict[str, Any]:
    if type(name) is str:
        for case in protocol["measurement"]["cases"]:
            if case["name"] == name:
                return dict(case)
    raise ValueError("measurement case is not predeclared")
