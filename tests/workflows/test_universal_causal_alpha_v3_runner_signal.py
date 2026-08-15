from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3SignalGate
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
    non_overlapping_causal_alpha_v3_rows,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    evaluate_causal_alpha_v3_signal_gate_clustered,
)


def _metric(
    *,
    rank: float,
    spread: float,
    direction: float,
    episode: int,
    symbol: str = "BTCUSDT",
    contract_start: int | None = None,
    contract_stop: int | None = None,
) -> CausalAlphaV3SignalScopeMetric:
    start = episode * 100 if contract_start is None else contract_start
    stop = start + 97 if contract_stop is None else contract_stop
    kwargs: dict[str, Any] = {
        "fit_config_digest": "1" * 64,
        "symbol": symbol,
        "episode_index": episode,
        "contract_start": start,
        "contract_stop": stop,
        "contract_digest": (f"{episode + 2:x}" * 64)[:64],
        "fit_digest": "3" * 64,
        "forecast_digest": "4" * 64,
        "sample_count": 8,
        "rank_correlation": rank,
        "direction_accuracy": direction,
        "top_bottom_realized_spread": spread,
        "cohort_indices": tuple(range(8)),
    }
    return CausalAlphaV3SignalScopeMetric(**kwargs)


def _gate(*, minimum_independent_episode_count: int = 2) -> CausalAlphaV3SignalGate:
    return CausalAlphaV3SignalGate(
        minimum_independent_episode_count=minimum_independent_episode_count,
        minimum_raw_scope_coverage=1.0,
        minimum_rank_ic_lower_ci=-1.0,
        minimum_top_bottom_spread_lower_ci=-1.0,
        minimum_direction_accuracy_excess_lower_ci=-1.0,
        bootstrap_resamples=200,
        bootstrap_seed=5,
        bootstrap_block_size=1,
    )


def test_non_overlapping_signal_rows_use_label_interval_endpoints() -> None:
    selected = non_overlapping_causal_alpha_v3_rows(
        decision_indices=np.asarray([0, 1, 2, 3, 6], dtype=np.int64),
        label_end_indices=np.asarray([2, 3, 4, 5, 8], dtype=np.int64),
        eligible_mask=np.asarray([True, True, True, True, True]),
    )

    assert selected.tolist() == [0, 3, 4]


def test_signal_gate_v2_exposes_raw_and_independent_episode_units() -> None:
    metrics = (
        _metric(rank=0.20, spread=0.010, direction=0.65, episode=0),
        _metric(rank=0.25, spread=0.012, direction=0.62, episode=1),
        _metric(rank=0.18, spread=0.008, direction=0.60, episode=2),
    )

    evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
        metrics,
        expected_raw_scope_count=3,
        expected_independent_episode_count=3,
        gate=_gate(minimum_independent_episode_count=2),
    )

    assert evidence.passed is True
    assert evidence.raw_scope_count == 3
    assert evidence.expected_raw_scope_count == 3
    assert evidence.raw_scope_coverage == pytest.approx(1.0)
    assert evidence.independent_episode_count == 3
    assert evidence.expected_independent_episode_count == 3
    assert evidence.independence_unit == "chronological_episode"
    assert evidence.aggregation_mode == "cross_symbol_episode_mean"
    assert evidence.promotion_eligible is False


def test_signal_gate_clusters_by_contract_interval_not_local_episode_index() -> None:
    metrics = (
        _metric(
            rank=0.2,
            spread=0.01,
            direction=0.6,
            episode=0,
            symbol="BTCUSDT",
            contract_start=100,
            contract_stop=197,
        ),
        _metric(
            rank=0.3,
            spread=0.02,
            direction=0.7,
            episode=0,
            symbol="ETHUSDT",
            contract_start=200,
            contract_stop=297,
        ),
    )

    evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
        metrics,
        expected_raw_scope_count=2,
        expected_independent_episode_count=2,
        gate=_gate(minimum_independent_episode_count=2),
    )

    assert evidence.passed is True
    assert evidence.independent_episode_count == 2


def test_signal_gate_symbol_duplicates_increase_raw_not_independent_count() -> None:
    metrics = tuple(
        _metric(
            rank=0.2,
            spread=0.01,
            direction=0.6,
            episode=0,
            symbol=symbol,
            contract_start=100,
            contract_stop=197,
        )
        for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
    )

    evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
        metrics,
        expected_raw_scope_count=4,
        expected_independent_episode_count=1,
        gate=_gate(minimum_independent_episode_count=2),
    )

    assert evidence.raw_scope_count == 4
    assert evidence.raw_scope_coverage == pytest.approx(1.0)
    assert evidence.independent_episode_count == 1
    assert evidence.passed is False
    assert "independent_episode_count" in evidence.rejection_reasons


def test_signal_gate_fails_closed_on_missing_raw_scope() -> None:
    metric = _metric(rank=0.2, spread=0.01, direction=0.6, episode=0)
    evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
        (metric,),
        expected_raw_scope_count=2,
        expected_independent_episode_count=1,
        gate=_gate(minimum_independent_episode_count=1),
    )

    assert evidence.passed is False
    assert evidence.raw_scope_coverage == pytest.approx(0.5)
    assert "raw_scope_coverage" in evidence.rejection_reasons


def test_signal_scope_metric_requires_valid_contract_interval() -> None:
    with pytest.raises(ValueError, match="contract interval"):
        _metric(
            rank=0.2,
            spread=0.01,
            direction=0.6,
            episode=0,
            contract_start=100,
            contract_stop=100,
        )


def test_signal_scope_metric_rejects_undefined_rank_ic() -> None:
    kwargs: dict[str, Any] = {
        "fit_config_digest": "1" * 64,
        "symbol": "BTCUSDT",
        "episode_index": 0,
        "contract_start": 0,
        "contract_stop": 97,
        "contract_digest": "2" * 64,
        "fit_digest": "3" * 64,
        "forecast_digest": "4" * 64,
        "sample_count": 8,
        "rank_correlation": None,
        "direction_accuracy": 0.6,
        "top_bottom_realized_spread": 0.01,
        "cohort_indices": tuple(range(8)),
    }
    with pytest.raises(ValueError, match="rank"):
        CausalAlphaV3SignalScopeMetric(**kwargs)
