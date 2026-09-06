from __future__ import annotations

import math
from dataclasses import replace

import pytest

from trade_rl.artifacts.hashing import content_digest

_MEDIAN_REASON = "median_seed_symbol_balanced_net_wealth_not_above_cash"
_WORST_REASON = "worst_seed_symbol_balanced_net_wealth_below_cash"
_RISK_REASON = "all_seed_hard_risk_violation_count_nonzero"
_TURNOVER_REASON = "all_seed_turnover_p95_per_day_above_1_0"
_BOOTSTRAP_REASON = "paired_excess_bootstrap_lower_ci_not_above_zero"


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _leaf(
    *,
    cell: str,
    training_seed: int,
    concrete_symbol: str,
    net_log_growth: float,
    tile_suffix: str = "",
    turnover_per_day: float = 0.5,
    hard_risk_violation_count: int = 0,
):
    module = _module()
    tile_label = f"{cell}:{concrete_symbol}:{tile_suffix}"
    return module.UniversalTradeRLU2SelectionLeafMetrics(
        training_seed=training_seed,
        cell=cell,
        concrete_symbol=concrete_symbol,
        tile_identity=content_digest({"tile": tile_label}),
        replay_evidence_digest=content_digest(
            {
                "replay": tile_label,
                "seed": training_seed,
            }
        ),
        leaf_net_log_growth=net_log_growth,
        leaf_gross_log_growth=net_log_growth + 0.05,
        turnover_per_day=turnover_per_day,
        meaningful_execution=True,
        hard_risk_violation_count=hard_risk_violation_count,
        unexplained_execution_rejection_count=0,
    )


def _scope_leaves(
    cell: str,
    *,
    seed_logs: dict[int, float] | None = None,
    turnover_seed: int | None = None,
    risk_seed: int | None = None,
    tile_drift_seed: int | None = None,
):
    logs = seed_logs or {0: 0.02, 1: 0.01, 2: 0.0}
    rows = []
    for seed in (0, 1, 2):
        for symbol in ("DEV_A", "DEV_B"):
            suffix = "drift" if seed == tile_drift_seed and symbol == "DEV_A" else ""
            rows.append(
                _leaf(
                    cell=cell,
                    training_seed=seed,
                    concrete_symbol=symbol,
                    net_log_growth=logs[seed],
                    tile_suffix=suffix,
                    turnover_per_day=1.000001 if seed == turnover_seed else 0.5,
                    hard_risk_violation_count=(
                        1 if seed == risk_seed and symbol == "DEV_A" else 0
                    ),
                )
            )
    return tuple(rows)


def _segments(
    *,
    d1_values: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02),
    d2_values: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02),
):
    module = _module()
    return (
        module.UniversalTradeRLU2ReducedBootstrapSegment(
            source_window="development_future_1",
            decision_timestamps_ns=tuple(range(100, 100 + len(d1_values))),
            net_log_excess=d1_values,
        ),
        module.UniversalTradeRLU2ReducedBootstrapSegment(
            source_window="development_future_2",
            decision_timestamps_ns=tuple(range(200, 200 + len(d2_values))),
            net_log_excess=d2_values,
        ),
    )


def _evaluate(*, scope: str, leaves, segments=None):
    module = _module()
    return module.evaluate_universal_trade_rl_u2_seed_robustness(
        scope=scope,
        leaves=tuple(leaves),
        segments=_segments() if segments is None else tuple(segments),
    )


@pytest.mark.parametrize(
    ("scope", "leaves", "expected_wealth"),
    (
        (
            "D1",
            _scope_leaves("D1"),
            (math.exp(0.02), math.exp(0.01), 1.0),
        ),
        (
            "D2",
            _scope_leaves("D2"),
            (math.exp(0.02), math.exp(0.01), 1.0),
        ),
        (
            "D1+D2",
            _scope_leaves("D1") + _scope_leaves("D2"),
            (math.exp(0.04), math.exp(0.02), 1.0),
        ),
    ),
)
def test_u2_seed_robustness_accepts_fixed_seed_boundary(
    scope: str,
    leaves,
    expected_wealth: tuple[float, float, float],
) -> None:
    result = _evaluate(scope=scope, leaves=leaves)

    assert result.scope == scope
    assert result.training_seeds == (0, 1, 2)
    assert result.seed_symbol_balanced_net_wealth == pytest.approx(expected_wealth)
    assert result.median_seed_symbol_balanced_net_wealth > 1.0
    assert result.worst_seed_symbol_balanced_net_wealth == pytest.approx(1.0)
    assert result.all_seed_hard_risk_violation_count == 0
    assert result.all_seed_turnover_p95_per_day <= 1.0
    assert result.bootstrap_lower_ci > 0.0
    assert result.passed is True
    assert result.rejection_reasons == ()


def test_u2_seed_robustness_rejects_missing_seed() -> None:
    leaves = tuple(leaf for leaf in _scope_leaves("D1") if leaf.training_seed != 2)

    with pytest.raises(ValueError, match="seed|closure|complete"):
        _evaluate(scope="D1", leaves=leaves)


def test_u2_seed_robustness_rejects_cross_seed_tile_drift() -> None:
    leaves = _scope_leaves("D1", tile_drift_seed=2)

    with pytest.raises(ValueError, match="tile|symbol|closure|seed"):
        _evaluate(scope="D1", leaves=leaves)


def test_u2_seed_robustness_rejects_unexpected_cell() -> None:
    leaves = _scope_leaves("D1") + (
        _leaf(
            cell="C1",
            training_seed=0,
            concrete_symbol="DEV_A",
            net_log_growth=0.02,
        ),
    )

    with pytest.raises(ValueError, match="cell|scope|D1"):
        _evaluate(scope="D1", leaves=leaves)


def test_u2_seed_robustness_aggregate_requires_both_d_cells() -> None:
    with pytest.raises(ValueError, match="D1|D2|cell|scope"):
        _evaluate(scope="D1+D2", leaves=_scope_leaves("D1"))


def test_u2_seed_robustness_rejects_duplicate_leaf_identity() -> None:
    leaves = _scope_leaves("D1")

    with pytest.raises(ValueError, match="duplicate|identity|leaf"):
        _evaluate(scope="D1", leaves=leaves + (leaves[0],))


def test_u2_seed_robustness_rejects_median_seed_wealth_equal_cash() -> None:
    leaves = _scope_leaves("D1", seed_logs={0: 0.0, 1: 0.0, 2: 0.02})

    result = _evaluate(scope="D1", leaves=leaves)

    assert result.median_seed_symbol_balanced_net_wealth == pytest.approx(1.0)
    assert result.worst_seed_symbol_balanced_net_wealth == pytest.approx(1.0)
    assert result.rejection_reasons == (_MEDIAN_REASON,)
    assert result.passed is False


def test_u2_seed_robustness_rejects_worst_seed_below_cash() -> None:
    leaves = _scope_leaves("D1", seed_logs={0: -0.01, 1: 0.01, 2: 0.02})

    result = _evaluate(scope="D1", leaves=leaves)

    assert result.median_seed_symbol_balanced_net_wealth > 1.0
    assert result.worst_seed_symbol_balanced_net_wealth < 1.0
    assert result.rejection_reasons == (_WORST_REASON,)
    assert result.passed is False


def test_u2_seed_robustness_rejects_any_hard_risk_violation() -> None:
    result = _evaluate(scope="D1", leaves=_scope_leaves("D1", risk_seed=1))

    assert result.all_seed_hard_risk_violation_count == 1
    assert result.rejection_reasons == (_RISK_REASON,)
    assert result.passed is False


def test_u2_seed_robustness_rejects_any_seed_turnover_p95_above_one() -> None:
    result = _evaluate(scope="D1", leaves=_scope_leaves("D1", turnover_seed=1))

    assert result.all_seed_turnover_p95_per_day > 1.0
    assert result.rejection_reasons == (_TURNOVER_REASON,)
    assert result.passed is False


def test_u2_seed_robustness_rejects_bootstrap_lower_ci_equal_zero() -> None:
    result = _evaluate(
        scope="D1",
        leaves=_scope_leaves("D1"),
        segments=_segments(d1_values=(0.0, 0.0, 0.0, 0.0)),
    )

    assert result.bootstrap_lower_ci == pytest.approx(0.0)
    assert result.rejection_reasons == (_BOOTSTRAP_REASON,)
    assert result.passed is False


def test_u2_seed_robustness_d1_bootstrap_can_fail_while_d2_passes() -> None:
    segments = _segments(d1_values=(0.0, 0.0, 0.0, 0.0))

    d1 = _evaluate(scope="D1", leaves=_scope_leaves("D1"), segments=segments)
    d2 = _evaluate(scope="D2", leaves=_scope_leaves("D2"), segments=segments)

    assert d1.passed is False
    assert d1.rejection_reasons == (_BOOTSTRAP_REASON,)
    assert d2.passed is True


def test_u2_seed_robustness_d2_bootstrap_can_fail_while_d1_passes() -> None:
    segments = _segments(d2_values=(0.0, 0.0, 0.0, 0.0))

    d1 = _evaluate(scope="D1", leaves=_scope_leaves("D1"), segments=segments)
    d2 = _evaluate(scope="D2", leaves=_scope_leaves("D2"), segments=segments)

    assert d1.passed is True
    assert d2.passed is False
    assert d2.rejection_reasons == (_BOOTSTRAP_REASON,)


def test_u2_seed_robustness_aggregate_binds_both_bootstrap_segments() -> None:
    segments = _segments()
    result = _evaluate(
        scope="D1+D2",
        leaves=_scope_leaves("D1") + _scope_leaves("D2"),
        segments=segments,
    )

    assert result.bootstrap_segment_digests == tuple(
        segment.digest for segment in segments
    )
    assert result.bootstrap_lower_ci > 0.0
    assert result.passed is True


def test_u2_seed_robustness_evidence_rejects_pass_state_tampering() -> None:
    result = _evaluate(scope="D1", leaves=_scope_leaves("D1"))

    with pytest.raises(ValueError, match="pass|reason|consistent"):
        replace(result, passed=False, digest="")


def test_u2_seed_robustness_evidence_rejects_nonpositive_wealth_tampering() -> None:
    result = _evaluate(scope="D1", leaves=_scope_leaves("D1"))
    transformed = (
        math.exp(-1.0),
        result.seed_symbol_balanced_net_wealth[1],
        result.seed_symbol_balanced_net_wealth[2],
    )

    with pytest.raises(ValueError, match="wealth|positive|closure"):
        replace(
            result,
            seed_symbol_balanced_net_wealth=(
                -1.0,
                result.seed_symbol_balanced_net_wealth[1],
                result.seed_symbol_balanced_net_wealth[2],
            ),
            median_seed_symbol_balanced_net_wealth=float(sorted(transformed)[1]),
            worst_seed_symbol_balanced_net_wealth=float(min(transformed)),
            passed=False,
            rejection_reasons=(_WORST_REASON,),
            digest="",
        )
