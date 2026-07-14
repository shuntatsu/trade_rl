from __future__ import annotations

import math

from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries


def test_elapsed_years_controls_annualization_for_session_returns() -> None:
    series = ReturnSeries(
        values=(0.01, -0.01, 0.02, -0.005),
        kind=ReturnKind.BASE_BAR,
        periods_per_year=252,
        elapsed_years=4 / 365.0,
    )

    metrics = evaluate_performance(series)

    expected_frequency = 365.0
    mean = sum(series.values) / len(series.values)
    variance = sum((value - mean) ** 2 for value in series.values) / len(series.values)
    assert math.isclose(
        metrics.sharpe,
        mean / math.sqrt(variance) * math.sqrt(expected_frequency),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics.periods_per_year == 365
