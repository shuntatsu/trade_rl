from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation import (
    RidgeEconomicGateCostAuthority,
    RidgeEconomicGateEvaluation,
    RidgeEconomicGateEvaluationSpec,
    RidgeEconomicGateSymbolResult,
    canonical_ridge_economic_gate_evaluation_spec,
    evaluate_ridge_economic_gate,
    evaluation_return_sha256,
    research_status_from_counts,
)
from trade_rl.strategies.forecasts.ridge import RidgeForecastModel, RidgeForecastStrategy
from trade_rl.strategies.forecasts.ridge_economic_gate import RidgeEconomicGateStrategy

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_FEATURE_NAMES = (
    "1h__log_return_1bar",
    "1h__log_return_4bar",
    "1h__log_return_24bar",
    "unused_3",
    "1h__realized_volatility_24bar",
    "1h__volume_zscore_24bar",
    "1h__funding_bps",
    "1h__rsi_14bar",
    "unused_8",
    "unused_9",
    "1h__macd_histogram_12_26_9",
) + tuple(f"unused_{index}" for index in range(11, 60)) + (
    "4h__log_return_4bar",
    "unused_61",
    "unused_62",
    "4h__realized_volatility_24bar",
) + tuple(f"unused_{index}" for index in range(64, 114)) + (
    "1d__log_return_1bar",
    "unused_115",
    "unused_116",
    "unused_117",
    "1d__realized_volatility_24bar",
)


def _dataset() -> MarketDataset:
    n_bars = 3
    n_symbols = len(_SYMBOLS)
    n_features = len(_FEATURE_NAMES)
    close = np.full((n_bars, n_symbols), 100.0, dtype=np.float64)
    timestamps = np.asarray(
        [
            "2023-01-01T00:00:00",
            "2025-01-01T00:00:00",
            "2025-01-01T01:00:00",
        ],
        dtype="datetime64[ns]",
    )
    return MarketDataset(
        dataset_id="6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
        symbols=_SYMBOLS,
        timestamps=timestamps,
        features=np.zeros((n_bars, n_symbols, n_features), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, n_symbols), 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones(
            (n_bars, n_symbols, n_features), dtype=np.bool_
        ),
        feature_names=_FEATURE_NAMES,
        global_feature_names=("regime",),
        periods_per_year=8_760,
        calendar_kind="session_calendar",
        fee_rate=np.full((n_bars, n_symbols), 0.0005, dtype=np.float64),
        taker_fee_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        spread_rate=np.full((n_bars, n_symbols), 0.0002, dtype=np.float64),
    )


def _model() -> RidgeForecastModel:
    indices = (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)
    width = len(indices)
    return RidgeForecastModel(
        feature_indices=indices,
        feature_mean=np.zeros(width, dtype=np.float64),
        feature_scale=np.ones(width, dtype=np.float64),
        coefficients=np.full(width, 0.001, dtype=np.float64),
        intercept=0.0,
        horizon_hours=24,
        alpha=1.0,
        n_samples=100,
        fit_cutoff=np.datetime64("2023-01-01T00:00:00", "ns"),
    )


def _cost_authority(
    spec: RidgeEconomicGateEvaluationSpec,
) -> RidgeEconomicGateCostAuthority:
    return RidgeEconomicGateCostAuthority(
        source_run_id=1,
        source_artifact_id=2,
        source_artifact_api_digest="a" * 64,
        dataset_id=spec.dataset_id,
        dataset_artifact_digest=spec.dataset_artifact_digest,
        evaluation_start=spec.evaluation_start,
        evaluation_stop_exclusive=spec.evaluation_stop_exclusive,
        fee_rate=spec.fee_rate,
        taker_fee_rate=spec.taker_fee_rate,
        spread_rate=spec.spread_rate,
        one_way_explicit_cost=spec.one_way_explicit_cost,
        verified=True,
    )


def _entry(
    name: str,
    *,
    total_return: float,
    total_cost: float,
    turnover: float,
    max_drawdown: float,
    returns: tuple[float, ...],
    termination_reasons: tuple[str, ...] = (),
) -> Any:
    return SimpleNamespace(
        name=name,
        metrics=SimpleNamespace(
            total_return=total_return,
            total_cost=total_cost,
            turnover_total=turnover,
            max_drawdown=max_drawdown,
            termination_count=len(termination_reasons),
            n_periods=len(returns),
        ),
        replay=SimpleNamespace(
            returns=SimpleNamespace(values=returns),
            diagnostics=SimpleNamespace(termination_reasons=termination_reasons),
        ),
    )


def _comparison() -> Any:
    by_symbol = []
    for index, symbol in enumerate(_SYMBOLS):
        baseline_returns = (0.0, 0.0)
        candidate_returns = (0.01 + index * 0.001, 0.0)
        baseline = _entry(
            "baseline",
            total_return=0.0,
            total_cost=10.0,
            turnover=4.0,
            max_drawdown=0.20,
            returns=baseline_returns,
        )
        candidate = _entry(
            "candidate",
            total_return=(1.0 + candidate_returns[0]) * (1.0 + candidate_returns[1])
            - 1.0,
            total_cost=5.0,
            turnover=2.0,
            max_drawdown=0.10,
            returns=candidate_returns,
        )
        by_symbol.append(
            SimpleNamespace(
                symbol_index=index,
                symbol=symbol,
                comparison=SimpleNamespace(entries=(baseline, candidate)),
            )
        )
    return SimpleNamespace(by_symbol=tuple(by_symbol))


def _symbol_result(
    symbol: str,
    *,
    baseline_return: float = -0.10,
    candidate_return: float = 0.05,
    baseline_cost: float = 10.0,
    candidate_cost: float = 5.0,
    baseline_turnover: float = 4.0,
    candidate_turnover: float = 2.0,
    baseline_drawdown: float = 0.20,
    candidate_drawdown: float = 0.10,
    baseline_reasons: tuple[str, ...] = (),
    candidate_reasons: tuple[str, ...] = (),
) -> RidgeEconomicGateSymbolResult:
    return RidgeEconomicGateSymbolResult(
        symbol=symbol,
        baseline_total_return=baseline_return,
        candidate_total_return=candidate_return,
        excess_total_return=candidate_return - baseline_return,
        baseline_total_cost=baseline_cost,
        candidate_total_cost=candidate_cost,
        baseline_turnover_total=baseline_turnover,
        candidate_turnover_total=candidate_turnover,
        baseline_max_drawdown=baseline_drawdown,
        candidate_max_drawdown=candidate_drawdown,
        baseline_termination_count=len(baseline_reasons),
        candidate_termination_count=len(candidate_reasons),
        baseline_termination_reasons=baseline_reasons,
        candidate_termination_reasons=candidate_reasons,
        new_termination=bool(candidate_reasons and candidate_reasons != baseline_reasons),
        baseline_n_periods=2,
        candidate_n_periods=2,
        baseline_return_sha256="b" * 64,
        candidate_return_sha256="c" * 64,
    )


def test_canonical_spec_binds_sealed_prereg_and_successor_authorities() -> None:
    spec = canonical_ridge_economic_gate_evaluation_spec()

    assert spec.issue_number == 618
    assert spec.protocol_issue_number == 616
    assert spec.protocol_head_sha == "75999e53c70224c31a62b106e4a8d2caa4b920ac"
    assert spec.protocol_module_blob == "72d532351238223a1c467d54addb8b6a8de829f5"
    assert (
        spec.protocol_digest
        == "167af235eeba44b9ade780c95ef7bab11c7b211d81018b82ece5d7044f7c4eb3"
    )
    assert spec.protocol_seal_run_id == 35126253392
    assert spec.protocol_seal_artifact_id == 10459431804
    assert spec.protocol_fresh_artifact_id == 10458759061
    assert spec.dataset_id == _dataset().dataset_id
    assert spec.symbols == _SYMBOLS
    assert spec.feature_indices == (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)
    assert spec.ridge_horizon_hours == 24
    assert spec.ridge_alpha == 1.0
    assert spec.forecast_entry_threshold == 0.0025
    assert spec.forecast_exit_threshold == 0.0005
    assert spec.one_way_explicit_cost == 0.0007
    assert spec.cost_constancy_authority_required is True
    assert spec.evaluation_execution_authorized is False
    assert spec.final_test_accessed is False
    assert spec.shared_cash_profitability_established is False
    assert spec.production_eligible is False
    assert spec.live_trading_authorized is False
    assert spec.merge_authorized is False

    with pytest.raises(ValueError, match="frozen evaluation field gross_budget"):
        replace(spec, gross_budget=0.6)


def test_cost_authority_is_separate_strict_pre_pnl_evidence() -> None:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    authority = _cost_authority(spec)

    assert authority.verified is True
    assert authority.one_way_explicit_cost == pytest.approx(0.0007)
    assert len(authority.digest) == 64

    with pytest.raises(ValueError, match="verified"):
        replace(authority, verified=False)
    with pytest.raises(ValueError, match="dataset"):
        replace(authority, dataset_id="f" * 64)
    with pytest.raises(ValueError, match="one_way_explicit_cost"):
        replace(authority, one_way_explicit_cost=0.0008)
    with pytest.raises(ValueError, match="API digest"):
        replace(authority, source_artifact_api_digest="not-a-digest")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                positive_effect_symbols=4,
                median_excess_total_return=0.01,
                cost_reduction_symbols=4,
                turnover_reduction_symbols=4,
                drawdown_nonworse_symbols=4,
                new_termination_symbols=0,
            ),
            "PROMOTE_RESEARCH_REFERENCE",
        ),
        (
            dict(
                positive_effect_symbols=3,
                median_excess_total_return=0.01,
                cost_reduction_symbols=3,
                turnover_reduction_symbols=4,
                drawdown_nonworse_symbols=4,
                new_termination_symbols=0,
            ),
            "INCONCLUSIVE",
        ),
        (
            dict(
                positive_effect_symbols=2,
                median_excess_total_return=0.01,
                cost_reduction_symbols=5,
                turnover_reduction_symbols=5,
                drawdown_nonworse_symbols=5,
                new_termination_symbols=0,
            ),
            "REJECT_MECHANISM",
        ),
        (
            dict(
                positive_effect_symbols=5,
                median_excess_total_return=0.0,
                cost_reduction_symbols=5,
                turnover_reduction_symbols=5,
                drawdown_nonworse_symbols=5,
                new_termination_symbols=0,
            ),
            "REJECT_MECHANISM",
        ),
        (
            dict(
                positive_effect_symbols=5,
                median_excess_total_return=0.01,
                cost_reduction_symbols=5,
                turnover_reduction_symbols=5,
                drawdown_nonworse_symbols=5,
                new_termination_symbols=1,
            ),
            "REJECT_MECHANISM",
        ),
    ],
)
def test_research_status_replays_frozen_rule(
    kwargs: dict[str, int | float], expected: str
) -> None:
    assert research_status_from_counts(**kwargs) == expected


def test_research_status_rejects_bool_count_alias() -> None:
    with pytest.raises(ValueError, match="counts"):
        research_status_from_counts(
            positive_effect_symbols=True,  # type: ignore[arg-type]
            median_excess_total_return=0.01,
            cost_reduction_symbols=5,
            turnover_reduction_symbols=5,
            drawdown_nonworse_symbols=5,
            new_termination_symbols=0,
        )


def test_return_digest_is_deterministic_and_fail_closed() -> None:
    values = (0.01, -0.02, 0.03)
    assert evaluation_return_sha256(values) == evaluation_return_sha256(values)
    assert evaluation_return_sha256(values) != evaluation_return_sha256((0.01, -0.02))
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluation_return_sha256(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        evaluation_return_sha256((0.0, float("nan")))


def test_pairwise_evaluator_uses_same_frozen_model_and_common_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation as module

    dataset = _dataset()
    spec = canonical_ridge_economic_gate_evaluation_spec()
    cost_authority = _cost_authority(spec)
    model = _model()
    fit_calls: list[dict[str, object]] = []
    compare_calls: list[dict[str, object]] = []

    def fit_ridge(dataset_arg: MarketDataset, **kwargs: object) -> RidgeForecastModel:
        assert dataset_arg is dataset
        fit_calls.append(dict(kwargs))
        return model

    def compare(dataset_arg: MarketDataset, strategies: dict[str, object], **kwargs: object) -> Any:
        assert dataset_arg is dataset
        baseline = strategies["baseline"]
        candidate = strategies["candidate"]
        assert isinstance(baseline, RidgeForecastStrategy)
        assert isinstance(candidate, RidgeEconomicGateStrategy)
        assert baseline.model is model
        assert candidate.model is model
        assert candidate.config.model is model
        execution_cost = kwargs["execution_cost"]
        assert getattr(execution_cost, "processing_bar_volume_capacity") is False
        compare_calls.append({"strategies": strategies, **kwargs})
        return _comparison()

    monkeypatch.setattr(module, "fit_ridge_forecast", fit_ridge)
    monkeypatch.setattr(module, "compare_strategies_by_symbol", compare)

    result = evaluate_ridge_economic_gate(dataset, spec, cost_authority)

    assert len(fit_calls) == 1
    assert fit_calls[0] == {
        "feature_indices": spec.feature_indices,
        "fit_symbol_indices": (0, 1, 2, 3, 4),
        "fit_cutoff": np.datetime64("2023-01-01T00:00:00", "ns"),
        "horizon_hours": 24,
        "alpha": 1.0,
    }
    assert len(compare_calls) == 1
    assert compare_calls[0]["start_index"] == 0
    assert compare_calls[0]["stop_index"] == 1
    assert compare_calls[0]["gross_budget"] == 0.5
    assert compare_calls[0]["initial_capital"] == 100_000.0
    assert result.research_status == "PROMOTE_RESEARCH_REFERENCE"
    assert result.positive_effect_symbols == 5
    assert result.cost_reduction_symbols == 5
    assert result.turnover_reduction_symbols == 5
    assert result.drawdown_nonworse_symbols == 5
    assert result.new_termination_symbols == 0
    assert result.candidate_positive_total_return_symbols == 5
    assert result.cost_authority_digest == cost_authority.digest
    assert result.production_eligible is False
    assert result.final_test_accessed is False
    assert result.shared_cash_profitability_established is False


def test_evaluator_requires_exact_verified_cost_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation as module

    spec = canonical_ridge_economic_gate_evaluation_spec()
    dataset = _dataset()
    model = _model()
    monkeypatch.setattr(module, "fit_ridge_forecast", lambda *args, **kwargs: model)
    monkeypatch.setattr(module, "compare_strategies_by_symbol", lambda *args, **kwargs: _comparison())

    with pytest.raises(ValueError, match="cost authority"):
        evaluate_ridge_economic_gate(dataset, spec, None)  # type: ignore[arg-type]

    bad = replace(_cost_authority(spec), dataset_artifact_digest="d" * 64)
    with pytest.raises(ValueError, match="cost authority"):
        evaluate_ridge_economic_gate(dataset, spec, bad)


def test_symbol_result_and_aggregate_fail_closed_on_arithmetic() -> None:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    cost = _cost_authority(spec)
    rows = tuple(_symbol_result(symbol) for symbol in _SYMBOLS)
    result = RidgeEconomicGateEvaluation(
        spec_digest=spec.digest,
        cost_authority_digest=cost.digest,
        dataset_id=spec.dataset_id,
        symbols=spec.symbols,
        by_symbol=rows,
        positive_effect_symbols=5,
        median_excess_total_return=0.15,
        cost_reduction_symbols=5,
        turnover_reduction_symbols=5,
        drawdown_nonworse_symbols=5,
        new_termination_symbols=0,
        candidate_positive_total_return_symbols=5,
        research_status="PROMOTE_RESEARCH_REFERENCE",
    )
    assert len(result.digest) == 64

    with pytest.raises(ValueError, match="excess_total_return"):
        replace(rows[0], excess_total_return=123.0)
    with pytest.raises(ValueError, match="positive_effect_symbols"):
        replace(result, positive_effect_symbols=4)
    with pytest.raises(ValueError, match="research_status"):
        replace(result, research_status="REJECT_MECHANISM")
    with pytest.raises(ValueError, match="production"):
        replace(result, production_eligible=True)
