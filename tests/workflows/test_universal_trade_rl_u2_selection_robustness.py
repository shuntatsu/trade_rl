from __future__ import annotations

import math

import pytest

from trade_rl.artifacts.hashing import content_digest

_D1 = "D1"
_D2 = "D2"
_D12 = "D1+D2"
_D1_WINDOW = "development_future_1"
_D2_WINDOW = "development_future_2"

_MEDIAN_REASON = "median_seed_symbol_balanced_net_wealth_not_above_cash"
_WORST_REASON = "worst_seed_symbol_balanced_net_wealth_below_cash"
_RISK_REASON = "all_seed_hard_risk_violation_count_nonzero"
_TURNOVER_REASON = "all_seed_turnover_p95_per_day_above_1_0"
_BOOTSTRAP_REASON = "bootstrap_lower_95_ci_not_above_zero"


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _leaf(
    *,
    training_seed: int,
    cell: str,
    concrete_symbol: str,
    net_log_growth: float,
    turnover_per_day: float = 0.5,
    hard_risk_violation_count: int = 0,
    tile_variant: str = "canonical",
):
    module = _module()
    return module.UniversalTradeRLU2SelectionLeafMetrics(
        training_seed=training_seed,
        cell=cell,
        concrete_symbol=concrete_symbol,
        tile_identity=content_digest(
            {
                "cell": cell,
                "symbol": concrete_symbol,
                "tile_variant": tile_variant,
            }
        ),
        replay_evidence_digest=content_digest(
            {
                "seed": training_seed,
                "cell": cell,
                "symbol": concrete_symbol,
                "tile_variant": tile_variant,
            }
        ),
        leaf_net_log_growth=net_log_growth,
        leaf_gross_log_growth=net_log_growth + 0.01,
        turnover_per_day=turnover_per_day,
        meaningful_execution=True,
        hard_risk_violation_count=hard_risk_violation_count,
        unexplained_execution_rejection_count=0,
    )


def _leaves(
    *,
    cells: tuple[str, ...],
    net_by_seed: dict[int, float] | None = None,
    turnover_by_seed: dict[int, float] | None = None,
    hard_risk_seed: int | None = None,
    omit_seed: int | None = None,
    tile_drift_seed: int | None = None,
) -> tuple[object, ...]:
    net_by_seed = net_by_seed or {0: 0.02, 1: 0.03, 2: 0.01}
    turnover_by_seed = turnover_by_seed or {0: 0.5, 1: 0.5, 2: 0.5}
    rows: list[object] = []
    for cell in cells:
        for seed in (0, 1, 2):
            if seed == omit_seed:
                continue
            for symbol in ("DEV_A", "DEV_B"):
                rows.append(
                    _leaf(
                        training_seed=seed,
                        cell=cell,
                        concrete_symbol=symbol,
                        net_log_growth=net_by_seed[seed],
                        turnover_per_day=turnover_by_seed[seed],
                        hard_risk_violation_count=(
                            1 if seed == hard_risk_seed and symbol == "DEV_A" else 0
                        ),
                        tile_variant=(
                            "drifted"
                            if seed == tile_drift_seed and symbol == "DEV_A"
                            else "canonical"
                        ),
                    )
                )
    return tuple(rows)


def _segment(source_window: str, *values: float):
    module = _module()
    resolved = values or (0.02, 0.01, 0.03, 0.02, 0.01, 0.02)
    return module.UniversalTradeRLU2ReducedBootstrapSegment(
        source_window=source_window,
        decision_timestamps_ns=tuple(range(100, 100 + len(resolved))),
        net_log_excess=tuple(resolved),
    )


def _evaluate(
    *,
    scope: str,
    leaves: tuple[object, ...] | None = None,
    segments: tuple[object, ...] | None = None,
):
    module = _module()
    if leaves is None:
        cells = (_D1,) if scope == _D1 else (_D2,) if scope == _D2 else (_D1, _D2)
        leaves = _leaves(cells=cells)
    if segments is None:
        segments = (
            (_segment(_D1_WINDOW),)
            if scope == _D1
            else (_segment(_D2_WINDOW),)
            if scope == _D2
            else (_segment(_D1_WINDOW), _segment(_D2_WINDOW))
        )
    return module.evaluate_universal_trade_rl_u2_development_seed_robustness(
        scope=scope,
        leaves=leaves,
        bootstrap_segments=segments,
    )


@pytest.mark.parametrize(
    ("scope", "cells", "windows"),
    (
        (_D1, (_D1,), (_D1_WINDOW,)),
        (_D2, (_D2,), (_D2_WINDOW,)),
        (_D12, (_D1, _D2), (_D1_WINDOW, _D2_WINDOW)),
    ),
)
def test_u2_seed_robustness_accepts_exact_fixed_seed_closure(
    scope: str,
    cells: tuple[str, ...],
    windows: tuple[str, ...],
) -> None:
    result = _evaluate(scope=scope)

    assert result.scope == scope
    assert result.training_seeds == (0, 1, 2)
    assert result.cells == cells
    assert result.bootstrap_source_windows == windows
    assert result.passed is True
    assert result.rejection_reasons == ()
    assert result.median_seed_symbol_balanced_net_wealth > 1.0
    assert result.worst_seed_symbol_balanced_net_wealth >= 1.0
    assert result.all_seed_hard_risk_violation_count == 0
    assert result.all_seed_turnover_p95_per_day <= 1.0
    assert result.bootstrap_lower_ci > 0.0
    assert len(result.digest) == 64


def test_u2_seed_robustness_d1_d2_aggregate_sums_sequential_log_growth() -> None:
    result = _evaluate(
        scope=_D12,
        leaves=_leaves(
            cells=(_D1, _D2),
            net_by_seed={0: 0.02, 1: 0.03, 2: 0.01},
        ),
    )

    observed = dict(result.seed_symbol_balanced_net_wealth)
    assert observed[0] == pytest.approx(math.exp(0.04))
    assert observed[1] == pytest.approx(math.exp(0.06))
    assert observed[2] == pytest.approx(math.exp(0.02))


@pytest.mark.parametrize("scope", (_D1, _D2, _D12))
def test_u2_seed_robustness_rejects_missing_fixed_seed(scope: str) -> None:
    cells = (_D1,) if scope == _D1 else (_D2,) if scope == _D2 else (_D1, _D2)
    with pytest.raises(ValueError, match="seed|closure|0.*1.*2|complete"):
        _evaluate(scope=scope, leaves=_leaves(cells=cells, omit_seed=2))


def test_u2_seed_robustness_rejects_tile_closure_drift_across_seeds() -> None:
    with pytest.raises(ValueError, match="tile|scope|closure|identity"):
        _evaluate(
            scope=_D1,
            leaves=_leaves(cells=(_D1,), tile_drift_seed=2),
        )


def test_u2_seed_robustness_rejects_wrong_cell_for_scope() -> None:
    with pytest.raises(ValueError, match="cell|scope|D1"):
        _evaluate(scope=_D1, leaves=_leaves(cells=(_D2,)))


def test_u2_seed_robustness_rejects_wrong_bootstrap_segment_for_scope() -> None:
    with pytest.raises(ValueError, match="bootstrap|segment|window|D1"):
        _evaluate(scope=_D1, segments=(_segment(_D2_WINDOW),))


def test_u2_seed_robustness_rejects_aggregate_missing_d2_segment() -> None:
    with pytest.raises(ValueError, match="bootstrap|segment|D1|D2|window"):
        _evaluate(scope=_D12, segments=(_segment(_D1_WINDOW),))


def test_u2_seed_robustness_fails_when_median_seed_wealth_equals_cash() -> None:
    result = _evaluate(
        scope=_D1,
        leaves=_leaves(
            cells=(_D1,),
            net_by_seed={0: 0.0, 1: 0.0, 2: 0.02},
        ),
    )

    assert result.median_seed_symbol_balanced_net_wealth == pytest.approx(1.0)
    assert result.worst_seed_symbol_balanced_net_wealth == pytest.approx(1.0)
    assert result.passed is False
    assert result.rejection_reasons == (_MEDIAN_REASON,)


def test_u2_seed_robustness_fails_when_worst_seed_wealth_is_below_cash() -> None:
    result = _evaluate(
        scope=_D1,
        leaves=_leaves(
            cells=(_D1,),
            net_by_seed={0: 0.02, 1: 0.03, 2: -0.01},
        ),
    )

    assert result.median_seed_symbol_balanced_net_wealth > 1.0
    assert result.worst_seed_symbol_balanced_net_wealth < 1.0
    assert result.passed is False
    assert result.rejection_reasons == (_WORST_REASON,)


def test_u2_seed_robustness_fails_on_any_hard_risk_violation() -> None:
    result = _evaluate(
        scope=_D2,
        leaves=_leaves(cells=(_D2,), hard_risk_seed=1),
    )

    assert result.all_seed_hard_risk_violation_count == 1
    assert result.passed is False
    assert result.rejection_reasons == (_RISK_REASON,)


def test_u2_seed_robustness_fails_on_any_seed_turnover_p95_above_one() -> None:
    result = _evaluate(
        scope=_D2,
        leaves=_leaves(
            cells=(_D2,),
            turnover_by_seed={0: 0.5, 1: 1.01, 2: 0.5},
        ),
    )

    assert result.all_seed_turnover_p95_per_day > 1.0
    assert result.passed is False
    assert result.rejection_reasons == (_TURNOVER_REASON,)


@pytest.mark.parametrize(
    ("scope", "segments"),
    (
        (_D1, (_segment(_D1_WINDOW, -0.02, -0.01, -0.03),)),
        (_D2, (_segment(_D2_WINDOW, -0.02, -0.01, -0.03),)),
        (
            _D12,
            (
                _segment(_D1_WINDOW, -0.02, -0.01, -0.03),
                _segment(_D2_WINDOW, -0.02, -0.01, -0.03),
            ),
        ),
    ),
)
def test_u2_seed_robustness_has_separate_fail_closed_bootstrap_gate(
    scope: str,
    segments: tuple[object, ...],
) -> None:
    result = _evaluate(scope=scope, segments=segments)

    assert result.bootstrap_lower_ci <= 0.0
    assert result.passed is False
    assert result.rejection_reasons == (_BOOTSTRAP_REASON,)
