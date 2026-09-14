from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.loss_attribution import analyze_deterministic_loss_attribution
from trade_rl.evaluation.runs.artifact import LoadedCandidateRun


def _market() -> MarketDataset:
    timestamps = np.asarray(
        [
            "2025-12-31T23:00:00",
            "2026-01-01T00:00:00",
            "2026-02-01T00:00:00",
            "2027-01-01T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    mark = np.asarray([[100.0], [110.0], [99.0], [118.8]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.zeros((4, 1, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=mark.copy(),
        high=mark.copy(),
        low=mark.copy(),
        close=mark.copy(),
        volume=np.full((4, 1), 1_000.0),
        funding_rate=np.zeros((4, 1)),
        tradable=np.ones((4, 1), dtype=np.bool_),
        feature_available=np.ones((4, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=365,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=24.0,
        mark_price=mark,
    )


def _candidate_run(
    returns: np.ndarray,
    *,
    seed: int,
    total_return: float | None = None,
    total_cost: float = 15.0,
    funding_pnl: float = 3.0,
    borrow_cost: float = 2.0,
) -> LoadedCandidateRun:
    values = np.asarray(returns, dtype=np.float64)
    reconstructed = float(np.prod(1.0 + values) - 1.0)
    resolved_return = reconstructed if total_return is None else total_return
    metrics = {
        "total_return": resolved_return,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.1,
        "turnover_total": 2.5,
        "total_cost": total_cost,
        "funding_pnl": funding_pnl,
        "borrow_cost": borrow_cost,
        "n_trades": 4,
        "rebalance_events": 4,
        "termination_count": 0,
        "n_periods": len(values),
        "return_kind": "base_bar",
        "periods_per_year": 365,
    }
    diagnostics = {
        "turnover_total": 2.5,
        "total_cost": total_cost,
        "funding_pnl": funding_pnl,
        "borrow_cost": borrow_cost,
        "n_trades": 4,
        "rebalance_events": 4,
        "termination_reasons": [],
    }
    summary: dict[str, object] = {
        "schema_version": "lean_candidate_result_v2",
        "dataset_id": "a" * 64,
        "symbols": ["BTCUSDT"],
        "candidate_config": {"ppo_seed": seed},
        "evaluation": {
            "start": "2025-12-31T23:00:00.000000000",
            "stop_exclusive": "2027-01-01T00:00:00.000000000",
            "gross_budget": 0.5,
            "initial_capital": 1_000.0,
            "execution_overlay": "zero_overlay_dataset_fields_authoritative",
        },
        "by_symbol": [
            {
                "symbol_index": 0,
                "symbol": "BTCUSDT",
                "strategies": [
                    {
                        "name": "trend",
                        "return_key": "symbol_0_strategy_0",
                        "metrics": metrics,
                        "diagnostics": diagnostics,
                        "final_portfolio_value": 1_000.0 * (1.0 + resolved_return),
                        "fill_count": 4,
                    }
                ],
            }
        ],
    }
    return LoadedCandidateRun(
        root=Path(f"seed-{seed}"),
        summary=summary,
        returns={"symbol_0_strategy_0": values},
        provenance={},
    )


def test_loss_attribution_uses_realized_path_accounting_and_compounding() -> None:
    returns = np.asarray([0.10, -0.10, 0.20], dtype=np.float64)
    runs = tuple(_candidate_run(returns, seed=seed) for seed in range(5))

    report = analyze_deterministic_loss_attribution(
        runs,
        _market(),
        deterministic_strategies=("trend",),
    )

    assert report.dataset_id == "a" * 64
    assert report.symbols == ("BTCUSDT",)
    assert report.strategies == ("trend",)
    assert len(report.cells) == 1
    cell = report.cells[0]
    expected_total_return = float(np.prod(1.0 + returns) - 1.0)
    expected_net_pnl = 1_000.0 * expected_total_return
    expected_residual = expected_net_pnl + 15.0 - 3.0 + 2.0

    assert cell.net_total_return == pytest.approx(expected_total_return)
    assert cell.reconstructed_total_return == pytest.approx(expected_total_return)
    assert cell.net_pnl == pytest.approx(expected_net_pnl)
    assert cell.explicit_execution_cost == 15.0
    assert cell.funding_pnl == 3.0
    assert cell.borrow_cost == 2.0
    assert cell.residual_pnl == pytest.approx(expected_residual)
    assert cell.net_pnl == pytest.approx(
        cell.residual_pnl
        - cell.explicit_execution_cost
        + cell.funding_pnl
        - cell.borrow_cost
    )
    assert cell.explicit_cost_flip_on_realized_path is False
    assert cell.year_returns == (
        (2026, pytest.approx(-0.01)),
        (2027, pytest.approx(0.20)),
    )
    assert cell.positive_months == 2
    assert cell.month_count == 3
    assert cell.underlying_mark_correlation == pytest.approx(1.0)
    assert cell.underlying_mark_beta == pytest.approx(1.0)
    assert report.digest == report.digest
    assert "counterfactual" not in report.to_payload()


def test_loss_attribution_identifies_explicit_cost_flip_without_zero_cost_claim() -> (
    None
):
    returns = np.asarray([-0.01, 0.0, 0.0], dtype=np.float64)
    runs = tuple(
        _candidate_run(
            returns,
            seed=seed,
            total_cost=20.0,
            funding_pnl=0.0,
            borrow_cost=0.0,
        )
        for seed in range(5)
    )

    cell = analyze_deterministic_loss_attribution(
        runs,
        _market(),
        deterministic_strategies=("trend",),
    ).cells[0]

    assert cell.net_pnl == pytest.approx(-10.0)
    assert cell.residual_pnl == pytest.approx(10.0)
    assert cell.explicit_cost_flip_on_realized_path is True


def test_loss_attribution_rejects_deterministic_seed_drift() -> None:
    returns = np.asarray([0.01, 0.02, -0.01], dtype=np.float64)
    runs = [_candidate_run(returns, seed=seed) for seed in range(5)]
    runs[4] = _candidate_run(
        np.asarray([0.01, 0.021, -0.01], dtype=np.float64),
        seed=4,
    )

    with pytest.raises(ValueError, match="seed invariant"):
        analyze_deterministic_loss_attribution(
            tuple(runs),
            _market(),
            deterministic_strategies=("trend",),
        )


def test_loss_attribution_rejects_metric_or_clock_mismatch() -> None:
    returns = np.asarray([0.01, 0.02, -0.01], dtype=np.float64)
    bad_metric = tuple(
        _candidate_run(returns, seed=seed, total_return=0.5) for seed in range(5)
    )
    with pytest.raises(ValueError, match="total return"):
        analyze_deterministic_loss_attribution(
            bad_metric,
            _market(),
            deterministic_strategies=("trend",),
        )

    short = tuple(
        _candidate_run(np.asarray([0.01, 0.02]), seed=seed) for seed in range(5)
    )
    with pytest.raises(ValueError, match="return length"):
        analyze_deterministic_loss_attribution(
            short,
            _market(),
            deterministic_strategies=("trend",),
        )


def test_loss_attribution_rejects_nonfinite_or_internal_diagnostic_drift() -> None:
    returns = np.asarray([0.01, 0.02, -0.01], dtype=np.float64)
    runs = list(_candidate_run(returns, seed=seed) for seed in range(5))
    summary = dict(runs[0].summary)
    by_symbol = list(summary["by_symbol"])
    symbol = dict(by_symbol[0])
    strategies = list(symbol["strategies"])
    strategy = dict(strategies[0])
    metrics = dict(strategy["metrics"])
    metrics["total_cost"] = float("nan")
    strategy["metrics"] = metrics
    strategies[0] = strategy
    symbol["strategies"] = strategies
    by_symbol[0] = symbol
    summary["by_symbol"] = by_symbol
    runs[0] = LoadedCandidateRun(
        root=runs[0].root,
        summary=summary,
        returns=runs[0].returns,
        provenance=runs[0].provenance,
    )
    with pytest.raises(ValueError, match="finite"):
        analyze_deterministic_loss_attribution(
            tuple(runs),
            _market(),
            deterministic_strategies=("trend",),
        )

    runs = list(_candidate_run(returns, seed=seed) for seed in range(5))
    summary = dict(runs[0].summary)
    by_symbol = list(summary["by_symbol"])
    symbol = dict(by_symbol[0])
    strategies = list(symbol["strategies"])
    strategy = dict(strategies[0])
    diagnostics = dict(strategy["diagnostics"])
    diagnostics["total_cost"] = 999.0
    strategy["diagnostics"] = diagnostics
    strategies[0] = strategy
    symbol["strategies"] = strategies
    by_symbol[0] = symbol
    summary["by_symbol"] = by_symbol
    runs[0] = LoadedCandidateRun(
        root=runs[0].root,
        summary=summary,
        returns=runs[0].returns,
        provenance=runs[0].provenance,
    )
    with pytest.raises(ValueError, match="diagnostic"):
        analyze_deterministic_loss_attribution(
            tuple(runs),
            _market(),
            deterministic_strategies=("trend",),
        )
