from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data import (
    MarketCalendarKind,
    MarketDataset,
    publish_market_dataset_artifact,
)
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.runs import (
    RidgeArtifactComparisonConfig,
    compare_ridge_artifact_shared_cash,
    load_ridge_comparison_artifact,
    publish_ridge_comparison_artifact,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.forecasts import (
    RidgeForecastArtifactBinding,
    RidgeForecastModel,
    publish_ridge_forecast_artifact,
)

_DIGEST_DOMAIN = b"trade_rl.ridge_shared_cash_comparison_artifact_v1\0"
_DIGEST_DOMAIN_V2 = b"trade_rl.ridge_shared_cash_comparison_artifact_v2\0"


def _dataset(*, cash_rate: np.ndarray | None = None) -> MarketDataset:
    offsets_minutes = np.asarray([0, 30, 120, 180, 360, 390], dtype=np.int64)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        offsets_minutes * np.timedelta64(1, "m")
    )
    signal = np.asarray([-0.20, -0.15, 0.20, 0.25, -0.20, -0.25], dtype=np.float32)
    secondary = np.asarray([0.4, 0.3, 0.2, 0.1, -0.1, -0.2], dtype=np.float32)
    row_features = np.stack((signal, secondary), axis=1)
    features = np.repeat(row_features[:, None, :], 2, axis=1)
    close = np.asarray(
        [
            [100.0, 200.0],
            [101.0, 198.0],
            [102.0, 196.0],
            [103.0, 194.0],
            [104.0, 192.0],
            [105.0, 190.0],
        ],
        dtype=np.float64,
    )
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((6, 2), 100_000.0, dtype=np.float64),
        funding_rate=np.zeros((6, 2), dtype=np.float64),
        cash_rate=cash_rate,
        tradable=np.ones((6, 2), dtype=np.bool_),
        feature_available=np.ones((6, 2, 2), dtype=np.bool_),
        feature_names=("momentum", "secondary"),
        global_feature_names=("regime",),
        periods_per_year=252,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
    )
    return dataset.with_content_identity({"fixture": "ridge-comparison-artifact"})


def _model(dataset: MarketDataset) -> RidgeForecastModel:
    return RidgeForecastModel(
        feature_indices=(1, 0),
        feature_mean=np.asarray([0.0, 0.0], dtype=np.float64),
        feature_scale=np.asarray([2.0, 1.0], dtype=np.float64),
        coefficients=np.asarray([0.0, 1.0], dtype=np.float64),
        intercept=0.0,
        horizon_hours=2,
        alpha=0.5,
        n_samples=4,
        fit_cutoff=dataset.timestamps[2].astype("datetime64[ns]"),
    )


def _binding(
    dataset: MarketDataset,
    schema_version: str,
    dataset_digest: str,
    model: RidgeForecastModel,
) -> RidgeForecastArtifactBinding:
    symbol_indices = (1, 0)
    return RidgeForecastArtifactBinding(
        dataset_id=dataset.dataset_id,
        dataset_schema_version=schema_version,
        dataset_artifact_sha256=dataset_digest,
        fit_symbol_indices=symbol_indices,
        fit_symbol_names=tuple(dataset.symbols[index] for index in symbol_indices),
        feature_names=dataset.feature_names,
        feature_indices=model.feature_indices,
        fit_cutoff_ns=int(model.fit_cutoff.astype("int64")),
        horizon_hours=model.horizon_hours,
        alpha_hex=float(model.alpha).hex(),
        n_samples=model.n_samples,
        vintage_id="literal-comparison-artifact-fixture",
        source_commit="a" * 40,
        implementation_sha256="b" * 64,
        runtime_sha256="c" * 64,
    )


def _comparison(
    tmp_path: Path,
    *,
    market_order_profile=None,
    cash_rate: np.ndarray | None = None,
    include_forecast_diagnostics: bool = False,
):
    dataset = _dataset(cash_rate=cash_rate)
    dataset_root = tmp_path / "dataset"
    dataset_artifact = publish_market_dataset_artifact(dataset_root, dataset)
    model = _model(dataset)
    binding = _binding(
        dataset,
        dataset_artifact.schema_version,
        dataset_artifact.artifact_digest,
        model,
    )
    ridge_root = tmp_path / "ridge-model"
    ridge_digest = publish_ridge_forecast_artifact(ridge_root, model, binding)
    config = RidgeArtifactComparisonConfig(
        start_index=2,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=10_000.0,
        entry_threshold=0.10,
        exit_threshold=0.02,
        one_way_switch_cost=0.01,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.001,
            spread_rate=0.0005,
            impact_rate=0.0001,
            max_participation_rate=1.0,
        ),
        risk=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        ),
        market_order_profile=market_order_profile,
        minimum_hold_bars=0,
        include_forecast_diagnostics=include_forecast_diagnostics,
    )
    comparison = compare_ridge_artifact_shared_cash(
        dataset_root=dataset_root,
        ridge_artifact_root=ridge_root,
        expected_ridge_digest=ridge_digest,
        expected_binding=binding,
        config=config,
    )
    return comparison, build_candidate_run_provenance()


def _market_profile(dataset: MarketDataset):
    from tests.integrations.test_binance_forward_rules import symbol
    from trade_rl.integrations.binance.market_order_profile import (
        build_usdm_market_order_profile,
    )
    from trade_rl.integrations.binance.metadata import BinanceExchangeInfoSnapshot

    payload = {"symbols": [symbol("perpetual", name) for name in dataset.symbols]}
    raw = json.dumps(payload).encode()
    snapshot = BinanceExchangeInfoSnapshot(
        payload=payload,
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return build_usdm_market_order_profile(
        dataset,
        snapshot,
        selected_symbols=dataset.symbols,
        account_mode="one_way",
        reduce_only_exits=True,
    )


def _digest(raw: bytes) -> str:
    domain = (
        _DIGEST_DOMAIN_V2
        if b'"schema_version":"ridge_shared_cash_comparison_v2"' in raw
        else _DIGEST_DOMAIN
    )
    return hashlib.sha256(domain + raw).hexdigest()


def _mutable_payload(loaded) -> dict[str, object]:
    return json.loads(canonical_json_bytes(loaded.payload))


def _reseal(root: Path, payload: dict[str, object]) -> str:
    raw = canonical_json_bytes(payload)
    (root / "comparison.json").write_bytes(raw)
    return _digest(raw)


def _refresh_arm_metrics(arm: dict[str, object]) -> None:
    returns = cast(dict[str, object], arm["returns"])
    diagnostics = cast(dict[str, object], arm["diagnostics"])
    elapsed_value = returns["elapsed_years"]
    series = ReturnSeries(
        values=tuple(float(value) for value in cast(list[object], returns["values"])),
        kind=ReturnKind(cast(str, returns["kind"])),
        periods_per_year=int(cast(int, returns["periods_per_year"])),
        elapsed_years=(None if elapsed_value is None else float(elapsed_value)),
    )
    metrics = evaluate_performance(
        series,
        turnover_total=float(cast(float, diagnostics["turnover_total"])),
        total_cost=float(cast(float, diagnostics["total_cost"])),
        funding_pnl=float(cast(float, diagnostics["funding_pnl"])),
        borrow_cost=float(cast(float, diagnostics["borrow_cost"])),
        n_trades=int(cast(int, diagnostics["n_trades"])),
        rebalance_events=int(cast(int, diagnostics["rebalance_events"])),
        termination_count=len(cast(list[object], diagnostics["termination_reasons"])),
    )
    metrics_payload = asdict(metrics)
    metrics_payload["return_kind"] = metrics.return_kind.value
    arm["metrics"] = metrics_payload


def _set_cash_arm_movements(
    payload: dict[str, object],
    movements: tuple[float, ...],
    *,
    record_as_cash_interest: bool,
) -> None:
    arm = cast(dict[str, object], cast(list[object], payload["arms"])[0])
    ledger = cast(dict[str, object], arm["ledger"])
    intervals = cast(list[dict[str, object]], ledger["intervals"])
    initial_capital = float(
        cast(dict[str, object], payload["config"])["initial_capital"]
    )
    cash = initial_capital
    peak = initial_capital
    maximum_drawdown = 0.0
    returns: list[float] = []
    assert len(intervals) == len(movements)
    for interval, movement in zip(intervals, movements, strict=True):
        interval["cash_before"] = cash
        interval["portfolio_value_before"] = cash
        interval["max_drawdown_before"] = maximum_drawdown
        interval["interval_cash_interest"] = (
            movement if record_as_cash_interest else 0.0
        )
        next_cash = cash + movement
        interval["cash_after"] = next_cash
        interval["portfolio_value_after"] = next_cash
        interval["interval_net_return"] = next_cash / cash - 1.0
        peak = max(peak, next_cash)
        maximum_drawdown = max(maximum_drawdown, 1.0 - next_cash / peak)
        interval["max_drawdown_after"] = maximum_drawdown
        returns.append(float(interval["interval_net_return"]))
        cash = next_cash
    ledger["final_cash"] = cash
    ledger["final_portfolio_value"] = cash
    ledger["final_max_drawdown"] = maximum_drawdown
    cast(dict[str, object], arm["returns"])["values"] = returns
    _refresh_arm_metrics(arm)


def _semantic_mutations() -> tuple[tuple[str, Mutation], ...]:
    def decision_step_returns(payload):
        arm = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        cast(dict[str, object], arm["returns"])["kind"] = ReturnKind.DECISION_STEP.value
        _refresh_arm_metrics(arm)

    def elapsed_years_override(payload):
        arm = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        cast(dict[str, object], arm["returns"])["elapsed_years"] = 1.0
        _refresh_arm_metrics(arm)

    def mismatched_periods_per_year(payload):
        arm = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        cast(dict[str, object], arm["returns"])["periods_per_year"] = 365
        _refresh_arm_metrics(arm)

    def cash_gain_without_flow(payload):
        _set_cash_arm_movements(
            payload,
            (25.0, 25.0, 25.0),
            record_as_cash_interest=False,
        )

    def cash_arm_contains_order(payload):
        arms = cast(list[object], payload["arms"])
        cash = cast(dict[str, object], arms[0])
        traded = cast(dict[str, object], arms[1])
        cash_interval = cast(
            dict[str, object],
            cast(list[object], cast(dict[str, object], cash["ledger"])["intervals"])[0],
        )
        traded_interval = cast(
            dict[str, object],
            cast(list[object], cast(dict[str, object], traded["ledger"])["intervals"])[
                0
            ],
        )
        event = cast(list[object], traded_interval["order_events"])[0]
        cash_interval["order_events"] = [json.loads(canonical_json_bytes(event))]

    def cash_arm_requests_position(payload):
        cash = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        decisions = cast(
            list[object], cast(dict[str, object], cash["ledger"])["decisions"]
        )
        decision = cast(dict[str, object], decisions[0])
        decision["intents"] = [1, 0]
        decision["effective_intents"] = [1, 0]
        decision["proposal_weights"] = [0.25, 0.0]
        decision["target_weights"] = [0.25, 0.0]

    def cash_arm_has_position(payload):
        cash = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        ledger = cast(dict[str, object], cash["ledger"])
        intervals = cast(list[object], ledger["intervals"])
        first = cast(dict[str, object], intervals[0])
        second = cast(dict[str, object], intervals[1])
        terminal_interval = cast(dict[str, object], intervals[2])
        first["exact_quantities_after"] = ["1", "0"]
        second["exact_quantities_before"] = ["1", "0"]
        second["exact_quantities_after"] = ["1", "0"]
        terminal_interval["exact_quantities_before"] = ["1", "0"]
        decisions = cast(list[object], ledger["decisions"])
        first_decision = cast(dict[str, object], decisions[0])
        second_decision = cast(dict[str, object], decisions[1])
        first_decision["position_quantity_after"] = [1.0, 0.0]
        second_decision["position_quantity_before"] = [1.0, 0.0]
        second_decision["position_quantity_after"] = [1.0, 0.0]

    def cash_arm_claims_trades(payload):
        cash = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        diagnostics = cast(dict[str, object], cash["diagnostics"])
        diagnostics["n_trades"] = 1
        _refresh_arm_metrics(cash)

    def cash_arm_claims_rebalance(payload):
        cash = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        diagnostics = cast(dict[str, object], cash["diagnostics"])
        diagnostics["rebalance_events"] = 1
        _refresh_arm_metrics(cash)

    def cash_arm_claims_open_position(payload):
        cash = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        diagnostics = cast(dict[str, object], cash["diagnostics"])
        cast(dict[str, object], diagnostics["closed_trades"])["open_positions"] = 1
        _refresh_arm_metrics(cash)

    def cash_arm_claims_closed_trade(payload, *, realized_pnl: float):
        cash = cast(dict[str, object], cast(list[object], payload["arms"])[0])
        diagnostics = cast(dict[str, object], cash["diagnostics"])
        closed = cast(dict[str, object], diagnostics["closed_trades"])
        is_win = realized_pnl > 0.0
        closed.update(
            {
                "average_realized_pnl": realized_pnl,
                "breakeven_trades": 0,
                "closed_trades": 1,
                "gross_loss": max(-realized_pnl, 0.0),
                "gross_profit": max(realized_pnl, 0.0),
                "losing_trades": int(not is_win),
                "net_realized_pnl": realized_pnl,
                "open_positions": 0,
                "profit_factor": (
                    max(realized_pnl, 0.0) / max(-realized_pnl, 0.0)
                    if realized_pnl < 0.0
                    else None
                ),
                "win_rate": float(is_win),
                "winning_trades": int(is_win),
            }
        )
        _refresh_arm_metrics(cash)

    def cash_arm_claims_winning_trade(payload):
        cash_arm_claims_closed_trade(payload, realized_pnl=1.0)

    def cash_arm_claims_losing_trade(payload):
        cash_arm_claims_closed_trade(payload, realized_pnl=-1.0)

    def funding_event_not_joined_to_interval(payload):
        arms = cast(list[object], payload["arms"])
        arm = cast(dict[str, object], arms[1])
        intervals = cast(
            list[object], cast(dict[str, object], arm["ledger"])["intervals"]
        )
        interval = cast(dict[str, object], intervals[0])
        event = cast(
            dict[str, object], cast(list[object], interval["funding_events"])[0]
        )
        due = cast(list[bool], event["funding_due"])
        quantities = cast(list[float], event["signed_quantities"])
        marks = cast(list[float], event["mark_prices"])
        multipliers = cast(list[float], event["contract_multipliers"])
        rates = cast(list[float], event["funding_rates"])
        symbol_index = next(
            index
            for index, (is_due, quantity) in enumerate(
                zip(due, quantities, strict=True)
            )
            if is_due and quantity != 0.0
        )
        rates[symbol_index] = 0.001
        amount = -math.fsum(
            quantity * mark * multiplier * rate
            for is_due, quantity, mark, multiplier, rate in zip(
                due, quantities, marks, multipliers, rates, strict=True
            )
            if is_due
        )
        event["funding_rates"] = rates
        event["funding_amount"] = amount
        event["equity_after_funding"] = (
            cast(float, event["equity_before_funding"]) + amount
        )

    def ledger_drawdown_below_interval_returns(payload):
        arm = cast(dict[str, object], cast(list[object], payload["arms"])[1])
        ledger = cast(dict[str, object], arm["ledger"])
        for interval_value in cast(list[object], ledger["intervals"]):
            interval = cast(dict[str, object], interval_value)
            interval["max_drawdown_before"] = 0.0
            interval["max_drawdown_after"] = 0.0
        ledger["final_max_drawdown"] = 0.0

    return (
        ("non-producer return kind", decision_step_returns),
        ("elapsed-years override", elapsed_years_override),
        ("arm annualization mismatch", mismatched_periods_per_year),
        ("flat cash gain without a recorded flow", cash_gain_without_flow),
        ("cash arm order event", cash_arm_contains_order),
        ("cash arm non-flat intent", cash_arm_requests_position),
        ("cash arm non-flat quantity", cash_arm_has_position),
        ("cash arm trade diagnostic with recomputed metric", cash_arm_claims_trades),
        (
            "cash arm rebalance diagnostic with recomputed metric",
            cash_arm_claims_rebalance,
        ),
        ("cash arm open position diagnostic", cash_arm_claims_open_position),
        ("cash arm winning trade diagnostics", cash_arm_claims_winning_trade),
        ("cash arm losing trade diagnostics", cash_arm_claims_losing_trade),
        ("funding event not joined to interval", funding_event_not_joined_to_interval),
        (
            "ledger drawdown below return drawdown",
            ledger_drawdown_below_interval_returns,
        ),
    )


def test_publish_load_preserves_complete_four_arm_context_and_ledgers(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"

    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(root, expected_digest=digest)

    payload = loaded.payload
    assert loaded.root == root
    assert loaded.artifact_digest == digest
    assert payload["schema_version"] == "ridge_shared_cash_comparison_v1"
    dataset = payload["dataset"]
    assert dataset["dataset_id"] == comparison.dataset_id
    assert dataset["schema_version"] == comparison.dataset_artifact.schema_version
    assert dataset["artifact_digest"] == comparison.dataset_artifact.artifact_digest
    assert dataset["symbols"] == ("BTCUSDT", "ETHUSDT")
    assert dataset["evaluation_start_index"] == 2
    assert dataset["evaluation_stop_index"] == 5
    assert dataset["evaluation_start_ns"] == comparison.evaluation_start_ns
    assert dataset["evaluation_stop_ns"] == comparison.evaluation_stop_ns
    assert payload["model"]["artifact_digest"] == comparison.ridge_artifact_digest
    assert payload["model"]["binding"]["feature_names"] == (
        "momentum",
        "secondary",
    )
    assert payload["config"]["gross_budget"] == 0.5
    assert payload["config"]["execution_cost"]["fee_rate"] == 0.001
    assert canonical_json_bytes(payload["provenance"]) == canonical_json_bytes(
        provenance
    )

    arms = payload["arms"]
    assert tuple(arm["name"] for arm in arms) == (
        "cash",
        "legacy_hysteresis",
        "position_aware_zero_cost",
        "position_aware_cost",
    )
    for arm, entry in zip(arms, comparison.entries, strict=True):
        ledger = arm["ledger"]
        expected_ledger = json.loads(
            canonical_json_bytes(entry.replay.ledger_evidence.to_mapping())
        )
        assert canonical_json_bytes(ledger) == canonical_json_bytes(expected_ledger)
        assert ledger["schema_version"] == "shared_cash_replay_ledger_v2"
        assert ledger["dataset_id"] == comparison.dataset_id
        assert len(ledger["intervals"]) == 3
        assert len(ledger["decisions"]) == 2
        assert ledger["terminal_exact_quantities"] == ("0", "0")
        assert ledger["active_order_remainders"] == ()
        assert len(arm["returns"]["values"]) == 3
        assert arm["metrics"]["n_periods"] == 3
        assert arm["diagnostics"]["termination_reasons"] == ()

    with pytest.raises(TypeError):
        payload["schema_version"] = "changed"
    with pytest.raises(TypeError):
        dataset["dataset_id"] = "changed"


def test_opt_in_forecast_diagnostics_use_versioned_artifact_schema(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(
        tmp_path,
        include_forecast_diagnostics=True,
    )
    root = tmp_path / "comparison"

    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(root, expected_digest=digest)

    assert loaded.payload["schema_version"] == "ridge_shared_cash_comparison_v2"
    assert loaded.payload["config"]["include_forecast_diagnostics"] is True
    diagnostics = loaded.payload["forecast_diagnostics"]
    assert diagnostics["horizon_hours"] == comparison.binding.horizon_hours
    assert diagnostics["fit_cutoff_ns"] == comparison.binding.fit_cutoff_ns
    assert diagnostics["evaluation_start_index"] == 2
    assert diagnostics["decision_stop_index"] == 4
    assert diagnostics["evaluation_stop_index"] == 5
    assert diagnostics["scoring_as_of_ns"] == comparison.evaluation_stop_ns
    assert diagnostics["candidate_count"] == 4
    assert diagnostics["eligible_count"] == 0
    assert diagnostics["pooled"]["pair_count"] == 0
    assert diagnostics["pooled"]["mae"] is None
    assert tuple(item["symbol"] for item in diagnostics["by_symbol"]) == (
        "BTCUSDT",
        "ETHUSDT",
    )


def test_v2_loader_rejects_resealed_forecast_scope_and_count_changes(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(
        tmp_path,
        include_forecast_diagnostics=True,
    )
    root = tmp_path / "comparison"
    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    original = load_ridge_comparison_artifact(root, expected_digest=digest)

    def remove_opt_in(payload: dict[str, object]) -> None:
        cast(dict[str, object], payload["config"]).pop("include_forecast_diagnostics")

    def change_scoring_as_of(payload: dict[str, object]) -> None:
        diagnostics = cast(dict[str, object], payload["forecast_diagnostics"])
        diagnostics["scoring_as_of_ns"] = cast(int, diagnostics["scoring_as_of_ns"]) + 1

    def change_symbol_pair_count(payload: dict[str, object]) -> None:
        diagnostics = cast(dict[str, object], payload["forecast_diagnostics"])
        pooled = cast(dict[str, object], diagnostics["pooled"])
        pooled["pair_count"] = 1

    for mutate in (
        remove_opt_in,
        change_scoring_as_of,
        change_symbol_pair_count,
    ):
        payload = _mutable_payload(original)
        mutate(payload)
        digest = _reseal(root, payload)
        with pytest.raises((TypeError, ValueError), match="."):
            load_ridge_comparison_artifact(root, expected_digest=digest)


def test_v2_loader_rejects_resealed_symbol_candidate_redistribution(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(
        tmp_path,
        include_forecast_diagnostics=True,
    )
    root = tmp_path / "comparison"
    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    original = load_ridge_comparison_artifact(root, expected_digest=digest)
    payload = _mutable_payload(original)
    diagnostics = cast(dict[str, object], payload["forecast_diagnostics"])
    symbols = cast(list[dict[str, object]], diagnostics["by_symbol"])
    for item, candidate_count in zip(symbols, (3, 1), strict=True):
        item["candidate_count"] = candidate_count
        exclusions = cast(list[list[object]], item["excluded_counts"])
        for exclusion in exclusions:
            if exclusion[0] == "target_not_mature_by_scoring_as_of":
                exclusion[1] = candidate_count
                break
    digest = _reseal(root, payload)

    with pytest.raises(ValueError, match="candidate.*scope|scope.*candidate"):
        load_ridge_comparison_artifact(root, expected_digest=digest)


def test_v2_loader_rejects_resealed_pooled_metrics_that_disagree_with_symbols(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(
        tmp_path,
        include_forecast_diagnostics=True,
    )
    root = tmp_path / "comparison"
    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    original = load_ridge_comparison_artifact(root, expected_digest=digest)
    payload = _mutable_payload(original)
    diagnostics = cast(dict[str, object], payload["forecast_diagnostics"])
    symbols = cast(list[dict[str, object]], diagnostics["by_symbol"])
    symbol_metrics = (
        (1.0, 1.0, 2.0, 2.0, 1.0, 0.75),
        (3.0, 3.0, 4.0, 4.0, 1.0, 0.4375),
    )
    for item, values in zip(symbols, symbol_metrics, strict=True):
        item["eligible_count"] = 1
        exclusions = cast(list[list[object]], item["excluded_counts"])
        for exclusion in exclusions:
            if exclusion[0] == "target_not_mature_by_scoring_as_of":
                exclusion[1] = 1
                break
        metrics = cast(dict[str, object], item["metrics"])
        metrics["pair_count"] = 1
        (
            metrics["mae"],
            metrics["rmse"],
            metrics["zero_mae"],
            metrics["zero_rmse"],
            metrics["mae_improvement"],
            metrics["mse_skill"],
        ) = values
    diagnostics["eligible_count"] = 2
    exclusions = cast(list[list[object]], diagnostics["excluded_counts"])
    for exclusion in exclusions:
        if exclusion[0] == "target_not_mature_by_scoring_as_of":
            exclusion[1] = 2
            break
    cast(dict[str, object], diagnostics["pooled"]).update(
        {
            "pair_count": 2,
            "mae": 2.5,
            "rmse": 2.5,
            "zero_mae": 3.0,
            "zero_rmse": 3.0,
            "mae_improvement": 0.5,
            "mse_skill": 0.3055555555555556,
        }
    )
    digest = _reseal(root, payload)

    with pytest.raises(ValueError, match="pooled.*symbol|symbol.*pooled"):
        load_ridge_comparison_artifact(root, expected_digest=digest)


@pytest.mark.parametrize(
    ("case", "mutate"),
    _semantic_mutations(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_loader_rejects_unbound_return_and_cash_accounting_mutations(
    tmp_path: Path,
    case: str,
    mutate: Mutation,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(
        root,
        expected_digest=_digest((root / "comparison.json").read_bytes()),
    )
    payload = _mutable_payload(loaded)
    mutate(payload)
    recomputed_digest = _reseal(root, payload)

    with pytest.raises((TypeError, ValueError), match="."):
        load_ridge_comparison_artifact(root, expected_digest=recomputed_digest)


@pytest.mark.parametrize("annual_rate", (-0.05, 0.05))
def test_signed_cash_interest_is_preserved_for_flat_cash_arm(
    tmp_path: Path,
    annual_rate: float,
) -> None:
    comparison, provenance = _comparison(
        tmp_path,
        cash_rate=np.full((6,), annual_rate, dtype=np.float64),
    )
    root = tmp_path / "comparison"

    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(root, expected_digest=digest)

    cash_ledger = loaded.payload["arms"][0]["ledger"]
    negative_interest = [
        interval["interval_cash_interest"] for interval in cash_ledger["intervals"]
    ]
    assert negative_interest
    assert all(amount * annual_rate > 0.0 for amount in negative_interest)


def test_complete_losing_comparison_is_published_without_profit_filter(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"

    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(root, expected_digest=digest)

    arms = loaded.payload["arms"]
    losing_names = tuple(
        arm["name"] for arm in arms if arm["metrics"]["total_return"] < 0.0
    )
    assert losing_names
    assert tuple(arm["name"] for arm in arms) == (
        "cash",
        "legacy_hysteresis",
        "position_aware_zero_cost",
        "position_aware_cost",
    )


@pytest.mark.parametrize("case", ("early termination", "nonflat terminal"))
def test_publisher_rejects_incomplete_comparison_before_creating_destination(
    tmp_path: Path,
    case: str,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    entry = comparison.entries[1]
    ledger = entry.replay.ledger_evidence
    assert ledger is not None
    if case == "early termination":
        invalid_ledger = replace(ledger, termination_reason="insolvent")
    else:
        invalid_ledger = replace(ledger, terminal_exact_quantities=("1", "0"))
    invalid_replay = replace(entry.replay, ledger_evidence=invalid_ledger)
    invalid_entry = replace(entry, replay=invalid_replay)
    entries = list(comparison.entries)
    entries[1] = invalid_entry
    invalid_comparison = replace(comparison, entries=tuple(entries))
    root = tmp_path / "incomplete-comparison"

    with pytest.raises(ValueError, match="."):
        publish_ridge_comparison_artifact(
            root,
            invalid_comparison,
            provenance=provenance,
        )

    assert not root.exists()


def test_profile_remains_a_descriptive_declaration_after_reload(tmp_path: Path) -> None:
    profile = _market_profile(_dataset())
    comparison, provenance = _comparison(tmp_path, market_order_profile=profile)
    root = tmp_path / "comparison"

    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(root, expected_digest=digest)

    saved = loaded.payload["config"]["market_order_profile"]
    assert saved["digest"] == profile.digest
    assert canonical_json_bytes(saved["declaration"]) == canonical_json_bytes(
        profile.canonical_payload()
    )
    assert not isinstance(saved, type(profile))


Mutation = Callable[[dict[str, object]], None]


def _mutations() -> tuple[tuple[str, Mutation], ...]:
    def unknown_field(payload):
        payload["unexpected"] = 1

    def reordered_arms(payload):
        payload["arms"].reverse()

    def boolean_number(payload):
        payload["config"]["gross_budget"] = True

    def malformed_dataset_id(payload):
        payload["dataset"]["dataset_id"] = "not-a-sha"

    def mismatched_model_dataset(payload):
        payload["model"]["binding"]["dataset_id"] = "e" * 64

    def changed_return(payload):
        payload["arms"][0]["returns"]["values"][0] += 0.01

    def changed_metrics(payload):
        payload["arms"][0]["metrics"]["total_return"] += 0.01

    def broken_cash_chain(payload):
        payload["arms"][0]["ledger"]["intervals"][0]["cash_after"] += 1.0

    def incomplete_coverage(payload):
        payload["arms"][0]["ledger"]["intervals"].pop()

    def nonflat_terminal(payload):
        payload["arms"][0]["ledger"]["terminal_exact_quantities"][0] = "1"

    def bad_event_symbol(payload):
        event = payload["arms"][1]["ledger"]["intervals"][0]["order_events"][0]
        event["symbol_index"] = True

    def changed_implementation_digest(payload):
        payload["provenance"]["implementation_digest"] = "f" * 64

    def changed_policy_digest(payload):
        payload["arms"][0]["ledger"]["execution_policy_digest"] = "d" * 64

    def changed_decision_quantity(payload):
        payload["arms"][0]["ledger"]["decisions"][0]["position_quantity_after"][0] += (
            1.0
        )

    return (
        ("unknown field", unknown_field),
        ("arm order", reordered_arms),
        ("boolean numeric", boolean_number),
        ("dataset identity", malformed_dataset_id),
        ("model Dataset binding", mismatched_model_dataset),
        ("return consistency", changed_return),
        ("metric consistency", changed_metrics),
        ("ledger cash chain", broken_cash_chain),
        ("coverage", incomplete_coverage),
        ("terminal position", nonflat_terminal),
        ("nested event", bad_event_symbol),
        ("implementation provenance", changed_implementation_digest),
        ("execution policy binding", changed_policy_digest),
        ("decision quantity", changed_decision_quantity),
    )


@pytest.mark.parametrize(
    ("case", "mutate"),
    _mutations(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_loader_rejects_semantic_mutations_even_with_recomputed_digest(
    tmp_path: Path,
    case: str,
    mutate: Mutation,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    loaded = load_ridge_comparison_artifact(
        root,
        expected_digest=_digest((root / "comparison.json").read_bytes()),
    )
    payload = _mutable_payload(loaded)
    mutate(payload)
    recomputed_digest = _reseal(root, payload)

    with pytest.raises((TypeError, ValueError), match="."):
        load_ridge_comparison_artifact(root, expected_digest=recomputed_digest)


def test_loader_rejects_wrong_external_digest_raw_tampering_and_extra_files(
    tmp_path: Path,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)

    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest="f" * 64)

    raw = (root / "comparison.json").read_bytes()
    marker = b'"schema_version":"ridge_shared_cash_comparison_v1"'
    assert marker in raw
    (root / "comparison.json").write_bytes(
        raw.replace(marker, b'"schema_version":"x"', 1)
    )
    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest=digest)

    (root / "comparison.json").write_bytes(raw)
    (root / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest=digest)


def test_loader_rejects_duplicate_keys_and_noncanonical_json(tmp_path: Path) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    raw = (root / "comparison.json").read_bytes()
    duplicate = raw[:-1] + b',"schema_version":"ridge_shared_cash_comparison_v1"}'
    (root / "comparison.json").write_bytes(duplicate)
    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest=_digest(duplicate))

    (root / "comparison.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        load_ridge_comparison_artifact(root, expected_digest=_digest(raw + b" "))


def test_publish_is_write_once_and_preserves_existing_artifact(tmp_path: Path) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    digest = publish_ridge_comparison_artifact(root, comparison, provenance=provenance)
    before = (root / "comparison.json").read_bytes()

    with pytest.raises(FileExistsError):
        publish_ridge_comparison_artifact(root, comparison, provenance=provenance)

    assert (root / "comparison.json").read_bytes() == before
    assert (
        load_ridge_comparison_artifact(root, expected_digest=digest).artifact_digest
        == digest
    )


@pytest.mark.parametrize("failure_point", ["write", "rename"])
def test_publish_failure_cleans_only_owned_staging_and_leaves_no_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    comparison, provenance = _comparison(tmp_path)
    root = tmp_path / "comparison"
    unowned_stage = tmp_path / ".comparison.staging-unowned"
    unowned_stage.mkdir()
    sentinel = unowned_stage / "sentinel"
    sentinel.write_bytes(b"preserve")
    if failure_point == "write":

        def fail(*_args, **_kwargs):
            raise OSError("injected write failure")

        monkeypatch.setattr(
            "trade_rl.evaluation.runs.ridge_comparison_artifact.atomic_write_bytes",
            fail,
        )
    else:

        def fail(*_args, **_kwargs):
            raise OSError("injected rename failure")

        monkeypatch.setattr(
            "trade_rl.evaluation.runs.ridge_comparison_artifact.atomic_rename_directory",
            fail,
        )

    with pytest.raises(OSError):
        publish_ridge_comparison_artifact(root, comparison, provenance=provenance)

    assert not root.exists()
    assert sentinel.read_bytes() == b"preserve"
    assert tuple(tmp_path.glob(".comparison.staging-*")) == (unowned_stage,)
