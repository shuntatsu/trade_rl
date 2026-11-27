from __future__ import annotations

import math

import pytest

from trade_rl.evaluation.metrics import compound_return, evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries


def test_performance_metrics_use_explicit_series_annualization() -> None:
    returns = ReturnSeries(
        values=(0.10, -0.05, 0.02),
        kind=ReturnKind.BASE_BAR,
        periods_per_year=365,
    )

    result = evaluate_performance(
        returns,
        turnover_total=4.5,
        total_cost=0.03,
        funding_pnl=0.01,
        n_trades=7,
    )

    mean = sum(returns.values) / len(returns.values)
    variance = sum((value - mean) ** 2 for value in returns.values) / len(
        returns.values
    )
    downside_rms = math.sqrt(
        sum(min(value, 0.0) ** 2 for value in returns.values) / len(returns.values)
    )
    assert result.total_return == pytest.approx(1.10 * 0.95 * 1.02 - 1.0)
    assert result.sharpe == pytest.approx(mean / math.sqrt(variance) * math.sqrt(365))
    assert result.sortino == pytest.approx(mean / downside_rms * math.sqrt(365))
    assert result.max_drawdown == pytest.approx(0.05)
    assert result.turnover_total == 4.5
    assert result.total_cost == 0.03
    assert result.funding_pnl == 0.01
    assert result.n_trades == 7
    assert result.n_periods == 3
    assert result.return_kind is ReturnKind.BASE_BAR
    assert result.periods_per_year == 365


def test_constant_return_series_has_zero_ratio_metrics() -> None:
    returns = ReturnSeries(
        values=(0.0, 0.0, 0.0),
        kind=ReturnKind.DECISION_STEP,
        periods_per_year=2_190,
    )

    result = evaluate_performance(returns)

    assert result.sharpe == 0.0
    assert result.sortino == 0.0
    assert result.max_drawdown == 0.0


@pytest.mark.parametrize(
    "values",
    [(), (math.nan,), (math.inf,), (-1.0,)],
)
def test_return_series_rejects_invalid_financial_values(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        ReturnSeries(
            values=values,
            kind=ReturnKind.BASE_BAR,
            periods_per_year=8_760,
        )


def test_performance_inputs_reject_negative_accounting_totals() -> None:
    returns = ReturnSeries(
        values=(0.01,),
        kind=ReturnKind.BASE_BAR,
        periods_per_year=8_760,
    )

    with pytest.raises(ValueError, match="turnover"):
        evaluate_performance(returns, turnover_total=-0.1)


@pytest.mark.parametrize("values", [(1e155, 1e155), (float("nan"),), (float("inf"),)])
def test_compound_return_rejects_nonfinite_wealth(values) -> None:
    with pytest.raises(ValueError, match="compound.*finite"):
        compound_return(values)


def test_metrics_cannot_publish_overflowed_total_return() -> None:
    returns = ReturnSeries((1e155, 1e155), ReturnKind.BASE_BAR, 365)
    with pytest.raises(ValueError, match="compound.*finite"):
        evaluate_performance(returns)


def test_metrics_rejects_unrepresentable_variance() -> None:
    returns = ReturnSeries((1e155, -0.5), ReturnKind.BASE_BAR, 365)
    with pytest.raises(ValueError, match="performance.*finite"):
        evaluate_performance(returns)


@pytest.mark.parametrize("periods, elapsed", [(365, 5e-324), (10**400, None)])
def test_metrics_rejects_unrepresentable_effective_annualization(
    periods, elapsed
) -> None:
    returns = ReturnSeries((0.01, 0.02), ReturnKind.BASE_BAR, periods, elapsed)
    with pytest.raises(ValueError, match="annualization.*finite"):
        evaluate_performance(returns)


def test_compound_return_preserves_empty_and_total_loss_conventions() -> None:
    assert compound_return(()) == 0.0
    assert compound_return((0.1, -1.0, 0.2)) == -1.0
    assert compound_return((0.1, -0.05, 0.02)) == 1.1 * 0.95 * 1.02 - 1.0


def test_metrics_rejects_nonfinite_annualized_ratio_with_finite_wealth() -> None:
    returns = ReturnSeries((1e150, -1e-150), ReturnKind.BASE_BAR, 365, 1e-300)
    with pytest.raises(ValueError, match="performance.*finite"):
        evaluate_performance(returns)
