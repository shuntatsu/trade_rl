from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Candidate
from trade_rl.workflows.universal_causal_alpha_v10_stage_entry import (
    _execution_no_trade_band,
    _path,
    _require_execution_no_trade_band,
    _training_rows,
)


def test_v10_training_rows_bind_fast_and_slow_labels_without_symbol_features() -> None:
    decisions = np.arange(10, 90, 10, dtype=np.int64)
    local = SimpleNamespace(
        feature_names=("return_4h", "volatility"),
        values=np.column_stack((decisions, decisions + 1.0)),
        available=np.ones((len(decisions), 2), dtype=np.bool_),
    )
    global_context = SimpleNamespace(
        feature_names=("market_return",),
        values=decisions[:, None] + 2.0,
        available=np.ones((len(decisions), 1), dtype=np.bool_),
    )
    sample = SimpleNamespace(
        decision_indices=decisions,
        label_end_indices_4h=decisions + 2,
        label_end_indices_72h=decisions + 20,
        labels_4h=decisions / 1_000.0,
        labels_72h=decisions / 100.0,
        local_context=local,
        global_context=global_context,
    )
    prepared = SimpleNamespace(
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": sample},
    )

    rows = _training_rows(prepared)["AAAUSDT"]

    assert rows.fast_label_end_indices.tolist() == (decisions + 2).tolist()
    assert rows.slow_label_end_indices.tolist() == (decisions + 20).tolist()
    assert rows.fast_labels.tolist() == (decisions / 1_000.0).tolist()
    assert rows.slow_labels.tolist() == (decisions / 100.0).tolist()
    assert rows.feature_names == (
        "local:return_4h",
        "local:volatility",
        "global:market_return",
    )
    assert all("symbol" not in name for name in rows.feature_names)


def test_v10_resolves_execution_band_from_replay_environment_and_closes_it() -> None:
    closed: list[bool] = []

    class Environment:
        pre_trade_risk = SimpleNamespace(
            config=SimpleNamespace(no_trade_band=0.05)
        )

        def close(self) -> None:
            closed.append(True)

    prepared = SimpleNamespace(
        prepared_v3=SimpleNamespace(
            environment_factories={"BTCUSDT": Environment},
        )
    )

    assert _execution_no_trade_band(prepared, "BTCUSDT") == 0.05
    assert closed == [True]


def test_v10_replay_rejects_execution_band_drift() -> None:
    environment = SimpleNamespace(
        pre_trade_risk=SimpleNamespace(
            config=SimpleNamespace(no_trade_band=0.04)
        )
    )

    with pytest.raises(ValueError, match="execution no-trade band drifted"):
        _require_execution_no_trade_band(environment, 0.05)


def test_v10_leaf_paths_are_candidate_symbol_episode_scoped() -> None:
    assert _path(CausalAlphaV10Candidate.HIERARCHICAL_WAVE, "BTCUSDT", 8).as_posix() == (
        "selection/replays/08/BTCUSDT/hierarchical_wave.json"
    )
