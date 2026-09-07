from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.artifacts.hashing import content_digest

_MANDATORY_CELLS = ("B", "C1", "C2", "D1", "D2")

_GROSS_REASON = "symbol_balanced_gross_wealth_not_above_cash"
_NET_REASON = "symbol_balanced_net_wealth_not_above_cash"
_MEDIAN_REASON = "median_symbol_net_wealth_below_cash"
_MINIMUM_REASON = "minimum_symbol_net_wealth_below_cash"
_POSITIVE_FRACTION_REASON = "positive_net_scope_fraction_below_0_50"
_CVAR_REASON = "scope_net_return_cvar10_below_minus_0_01"
_TURNOVER_REASON = "turnover_p95_per_day_above_1_0"
_MEANINGFUL_REASON = "meaningful_execution_symbol_fraction_not_complete"
_HARD_RISK_REASON = "hard_risk_violation_count_nonzero"
_REJECTION_REASON = "unexplained_execution_rejection_count_nonzero"
_RETENTION_REASON = "positive_gross_log_growth_retention_below_0_50"


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _leaf(
    *,
    label: str,
    cell: str = "B",
    training_seed: int = 0,
    concrete_symbol: str = "DEV_A",
    net_log_growth: float,
    gross_log_growth: float,
    turnover_per_day: float = 1.0,
    meaningful_execution: bool = True,
    hard_risk_violation_count: int = 0,
    unexplained_execution_rejection_count: int = 0,
):
    module = _module()
    return module.UniversalTradeRLU2SelectionLeafMetrics(
        training_seed=training_seed,
        cell=cell,
        concrete_symbol=concrete_symbol,
        tile_identity=content_digest({"tile": label}),
        replay_evidence_digest=content_digest({"replay": label}),
        leaf_net_log_growth=net_log_growth,
        leaf_gross_log_growth=gross_log_growth,
        turnover_per_day=turnover_per_day,
        meaningful_execution=meaningful_execution,
        hard_risk_violation_count=hard_risk_violation_count,
        unexplained_execution_rejection_count=(unexplained_execution_rejection_count),
    )


def _summary(*leaves):
    module = _module()
    return module.summarize_universal_trade_rl_u2_selection_metrics(
        leaves=tuple(leaves)
    )


def _inclusive_boundary_summary(*, cell: str = "B", training_seed: int = 0):
    return _summary(
        _leaf(
            label=f"{cell}-boundary-negative",
            cell=cell,
            training_seed=training_seed,
            net_log_growth=-0.01,
            gross_log_growth=0.03125,
        ),
        _leaf(
            label=f"{cell}-boundary-positive",
            cell=cell,
            training_seed=training_seed,
            net_log_growth=0.04125,
            gross_log_growth=0.03125,
        ),
    )


def _median_and_minimum_equality_summary():
    leaves = []
    for symbol in ("DEV_A", "DEV_B"):
        leaves.extend(
            (
                _leaf(
                    label=f"{symbol}-zero-positive",
                    concrete_symbol=symbol,
                    net_log_growth=0.005,
                    gross_log_growth=0.01,
                ),
                _leaf(
                    label=f"{symbol}-zero-negative",
                    concrete_symbol=symbol,
                    net_log_growth=-0.005,
                    gross_log_growth=0.01,
                ),
            )
        )
    leaves.extend(
        (
            _leaf(
                label="DEV_C-positive-0",
                concrete_symbol="DEV_C",
                net_log_growth=0.015,
                gross_log_growth=0.01,
            ),
            _leaf(
                label="DEV_C-positive-1",
                concrete_symbol="DEV_C",
                net_log_growth=0.015,
                gross_log_growth=0.01,
            ),
        )
    )
    return _summary(*leaves)


@pytest.mark.parametrize("cell", _MANDATORY_CELLS)
def test_u2_primary_cell_gate_accepts_all_inclusive_boundaries(cell: str) -> None:
    module = _module()
    summary = _inclusive_boundary_summary(cell=cell)

    assert summary.positive_net_scope_fraction == 0.50
    assert summary.scope_net_return_cvar10 == -0.01
    assert summary.turnover_per_day_p95 == 1.0
    assert summary.positive_gross_log_growth_retention == 0.50

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.cell == cell
    assert result.training_seed == 0
    assert result.summary_digest == summary.digest
    assert result.passed is True
    assert result.rejection_reasons == ()


def test_u2_primary_cell_gate_accepts_median_and_minimum_wealth_equal_cash() -> None:
    module = _module()
    summary = _median_and_minimum_equality_summary()

    assert summary.symbol_balanced_net_wealth > 1.0
    assert summary.median_symbol_net_wealth == pytest.approx(1.0)
    assert summary.minimum_symbol_net_wealth == pytest.approx(1.0)
    assert summary.positive_net_scope_fraction >= 0.50
    assert summary.scope_net_return_cvar10 >= -0.01
    assert summary.positive_gross_log_growth_retention == pytest.approx(0.50)

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is True
    assert result.rejection_reasons == ()


def test_u2_primary_cell_gate_rejects_balanced_gross_wealth_equal_cash() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="gross-equality-0",
            net_log_growth=0.01,
            gross_log_growth=0.0,
        ),
        _leaf(
            label="gross-equality-1",
            net_log_growth=0.01,
            gross_log_growth=0.0,
        ),
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert _GROSS_REASON in result.rejection_reasons


def test_u2_primary_cell_gate_rejects_balanced_net_wealth_equal_cash() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="net-equality-positive",
            net_log_growth=0.005,
            gross_log_growth=0.01,
        ),
        _leaf(
            label="net-equality-negative",
            net_log_growth=-0.005,
            gross_log_growth=0.01,
        ),
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert _NET_REASON in result.rejection_reasons


def test_u2_primary_cell_gate_rejects_median_symbol_wealth_below_cash() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="median-A-positive",
            concrete_symbol="DEV_A",
            net_log_growth=0.005,
            gross_log_growth=0.01,
        ),
        _leaf(
            label="median-A-negative",
            concrete_symbol="DEV_A",
            net_log_growth=-0.010,
            gross_log_growth=0.01,
        ),
        _leaf(
            label="median-B-positive",
            concrete_symbol="DEV_B",
            net_log_growth=0.005,
            gross_log_growth=0.01,
        ),
        _leaf(
            label="median-B-negative",
            concrete_symbol="DEV_B",
            net_log_growth=-0.006,
            gross_log_growth=0.01,
        ),
        _leaf(
            label="median-C-positive-0",
            concrete_symbol="DEV_C",
            net_log_growth=0.018,
            gross_log_growth=0.01,
        ),
        _leaf(
            label="median-C-positive-1",
            concrete_symbol="DEV_C",
            net_log_growth=0.018,
            gross_log_growth=0.01,
        ),
    )

    assert summary.symbol_balanced_net_wealth > 1.0
    assert summary.positive_net_scope_fraction >= 0.50
    assert summary.scope_net_return_cvar10 >= -0.01
    assert summary.positive_gross_log_growth_retention == pytest.approx(0.50)

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert _MEDIAN_REASON in result.rejection_reasons


def test_u2_primary_cell_gate_rejects_minimum_symbol_wealth_below_cash() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="minimum-loser",
            concrete_symbol="DEV_A",
            net_log_growth=-0.005,
            gross_log_growth=0.02,
        ),
        _leaf(
            label="minimum-winner",
            concrete_symbol="DEV_B",
            net_log_growth=0.025,
            gross_log_growth=0.02,
        ),
    )

    assert summary.symbol_balanced_net_wealth > 1.0
    assert summary.median_symbol_net_wealth > 1.0
    assert summary.positive_net_scope_fraction == pytest.approx(0.50)
    assert summary.scope_net_return_cvar10 >= -0.01
    assert summary.positive_gross_log_growth_retention == pytest.approx(0.50)

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert result.rejection_reasons == (_MINIMUM_REASON,)


def test_u2_primary_cell_gate_rejects_positive_scope_fraction_below_half() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="positive-fraction-negative",
            net_log_growth=-0.00390625,
            gross_log_growth=0.015625,
        ),
        _leaf(
            label="positive-fraction-zero",
            net_log_growth=0.0,
            gross_log_growth=0.015625,
        ),
        _leaf(
            label="positive-fraction-positive",
            net_log_growth=0.03515625,
            gross_log_growth=0.03125,
        ),
    )

    assert summary.symbol_balanced_net_wealth > 1.0
    assert summary.positive_net_scope_fraction < 0.50
    assert summary.scope_net_return_cvar10 >= -0.01
    assert summary.positive_gross_log_growth_retention == 0.50

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert result.rejection_reasons == (_POSITIVE_FRACTION_REASON,)


def test_u2_primary_cell_gate_rejects_cvar_below_minus_one_percent() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="cvar-negative",
            net_log_growth=-0.011,
            gross_log_growth=0.02,
        ),
        _leaf(
            label="cvar-positive",
            net_log_growth=0.031,
            gross_log_growth=0.02,
        ),
    )

    assert summary.symbol_balanced_net_wealth > 1.0
    assert summary.positive_net_scope_fraction == pytest.approx(0.50)
    assert summary.scope_net_return_cvar10 < -0.01
    assert summary.positive_gross_log_growth_retention == pytest.approx(0.50)

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert result.rejection_reasons == (_CVAR_REASON,)


def test_u2_primary_cell_gate_rejects_turnover_above_one_per_day() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="turnover-negative",
            net_log_growth=-0.01,
            gross_log_growth=0.03125,
            turnover_per_day=1.000001,
        ),
        _leaf(
            label="turnover-positive",
            net_log_growth=0.04125,
            gross_log_growth=0.03125,
            turnover_per_day=1.000001,
        ),
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert result.rejection_reasons == (_TURNOVER_REASON,)


def test_u2_primary_cell_gate_requires_meaningful_execution_for_every_symbol() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="meaningful-negative",
            net_log_growth=-0.01,
            gross_log_growth=0.03125,
            meaningful_execution=False,
        ),
        _leaf(
            label="meaningful-positive",
            net_log_growth=0.04125,
            gross_log_growth=0.03125,
            meaningful_execution=False,
        ),
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert result.rejection_reasons == (_MEANINGFUL_REASON,)


def test_u2_primary_cell_gate_rejects_any_hard_risk_violation() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="risk-negative",
            net_log_growth=-0.01,
            gross_log_growth=0.03125,
            hard_risk_violation_count=1,
        ),
        _leaf(
            label="risk-positive",
            net_log_growth=0.04125,
            gross_log_growth=0.03125,
        ),
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert result.rejection_reasons == (_HARD_RISK_REASON,)


def test_u2_primary_cell_gate_rejects_any_unexplained_execution_rejection() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="rejection-negative",
            net_log_growth=-0.01,
            gross_log_growth=0.03125,
            unexplained_execution_rejection_count=1,
        ),
        _leaf(
            label="rejection-positive",
            net_log_growth=0.04125,
            gross_log_growth=0.03125,
        ),
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert result.rejection_reasons == (_REJECTION_REASON,)


def test_u2_primary_cell_gate_rejects_positive_gross_retention_below_half() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="retention-negative",
            net_log_growth=-0.005,
            gross_log_growth=0.02,
        ),
        _leaf(
            label="retention-positive",
            net_log_growth=0.023,
            gross_log_growth=0.02,
        ),
    )

    assert summary.symbol_balanced_net_wealth > 1.0
    assert summary.positive_net_scope_fraction == pytest.approx(0.50)
    assert summary.scope_net_return_cvar10 >= -0.01
    assert summary.positive_gross_log_growth_retention < 0.50

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.passed is False
    assert result.rejection_reasons == (_RETENTION_REASON,)


def test_u2_primary_cell_gate_rejection_reason_order_is_deterministic() -> None:
    module = _module()
    summary = _summary(
        _leaf(
            label="multi-positive",
            net_log_growth=0.005,
            gross_log_growth=0.01,
            meaningful_execution=False,
            hard_risk_violation_count=1,
            unexplained_execution_rejection_count=1,
        ),
        _leaf(
            label="multi-negative",
            net_log_growth=-0.005,
            gross_log_growth=0.01,
            meaningful_execution=False,
        ),
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.rejection_reasons == (
        _NET_REASON,
        _MEANINGFUL_REASON,
        _HARD_RISK_REASON,
        _REJECTION_REASON,
        _RETENTION_REASON,
    )
    assert result.passed is False


@pytest.mark.parametrize("training_seed", (1, 2))
def test_u2_primary_cell_gate_rejects_non_primary_training_seed(
    training_seed: int,
) -> None:
    module = _module()
    summary = _inclusive_boundary_summary(training_seed=training_seed)

    with pytest.raises(ValueError, match="primary|seed|0"):
        module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)


@pytest.mark.parametrize("cell", ("A", "E", "unexpected"))
def test_u2_primary_cell_gate_rejects_nonmandatory_cell(cell: str) -> None:
    module = _module()
    summary = _inclusive_boundary_summary(cell=cell)

    with pytest.raises(ValueError, match="cell|mandatory|B|C1|D1"):
        module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)


def test_u2_primary_cell_gate_uses_preregistered_threshold_payload(
    monkeypatch,
) -> None:
    from trade_rl.workflows import universal_trade_rl_u2_contract

    module = _module()
    summary = _inclusive_boundary_summary()
    thresholds = dict(universal_trade_rl_u2_contract._selection_thresholds_payload())
    thresholds["turnover_p95_per_day_max_inclusive"] = 0.999999
    monkeypatch.setattr(
        universal_trade_rl_u2_contract,
        "_selection_thresholds_payload",
        lambda: thresholds,
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.selection_thresholds_digest == content_digest(thresholds)
    assert result.rejection_reasons == (_TURNOVER_REASON,)
    assert result.passed is False


def test_u2_primary_cell_gate_evidence_rejects_pass_state_tampering() -> None:
    module = _module()
    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(
        summary=_inclusive_boundary_summary()
    )

    with pytest.raises(ValueError, match="pass|reason|consistent"):
        replace(result, passed=False, digest="")
