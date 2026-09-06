from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.artifacts.hashing import content_digest

REASON_GROSS = "symbol_balanced_gross_wealth_not_above_minimum"
REASON_NET = "symbol_balanced_net_wealth_not_above_minimum"
REASON_MEDIAN = "median_symbol_net_wealth_below_minimum"
REASON_MINIMUM = "minimum_symbol_net_wealth_below_minimum"
REASON_POSITIVE = "positive_net_scope_fraction_below_minimum"
REASON_CVAR = "scope_net_return_cvar10_below_minimum"
REASON_TURNOVER = "turnover_per_day_p95_above_maximum"
REASON_EXECUTION = "meaningful_execution_symbol_fraction_mismatch"
REASON_RISK = "hard_risk_violation_count_mismatch"
REASON_REJECTION = "unexplained_execution_rejection_count_mismatch"
REASON_RETENTION = "positive_gross_log_growth_retention_below_minimum"


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _leaf(
    *,
    cell: str,
    training_seed: int,
    concrete_symbol: str,
    tile: str,
    net_log_growth: float,
    gross_log_growth: float,
    meaningful_execution: bool = True,
    hard_risk_violation_count: int = 0,
    unexplained_execution_rejection_count: int = 0,
):
    module = _module()
    return module.UniversalTradeRLU2SelectionLeafMetrics(
        training_seed=training_seed,
        cell=cell,
        concrete_symbol=concrete_symbol,
        tile_identity=content_digest(
            {
                "cell": cell,
                "seed": training_seed,
                "symbol": concrete_symbol,
                "tile": tile,
            }
        ),
        replay_evidence_digest=content_digest(
            {
                "replay": tile,
                "cell": cell,
                "seed": training_seed,
                "symbol": concrete_symbol,
            }
        ),
        leaf_net_log_growth=net_log_growth,
        leaf_gross_log_growth=gross_log_growth,
        turnover_per_day=1.0,
        meaningful_execution=meaningful_execution,
        hard_risk_violation_count=hard_risk_violation_count,
        unexplained_execution_rejection_count=(unexplained_execution_rejection_count),
    )


def _summarize(*, leaves):
    module = _module()
    return module.summarize_universal_trade_rl_u2_selection_metrics(
        leaves=tuple(leaves)
    )


def _base_summary(*, cell: str = "B", training_seed: int = 0):
    return _summarize(
        leaves=(
            _leaf(
                cell=cell,
                training_seed=training_seed,
                concrete_symbol="DEV_A",
                tile="A-worst",
                net_log_growth=-0.01,
                gross_log_growth=0.01,
            ),
            _leaf(
                cell=cell,
                training_seed=training_seed,
                concrete_symbol="DEV_A",
                tile="A-positive",
                net_log_growth=0.04,
                gross_log_growth=0.03,
            ),
            _leaf(
                cell=cell,
                training_seed=training_seed,
                concrete_symbol="DEV_B",
                tile="B-flat",
                net_log_growth=0.0,
                gross_log_growth=0.01,
            ),
            _leaf(
                cell=cell,
                training_seed=training_seed,
                concrete_symbol="DEV_B",
                tile="B-positive",
                net_log_growth=0.02,
                gross_log_growth=0.02,
            ),
        )
    )


def _summary_with_symbol_logs(
    *,
    net_logs: tuple[float, ...],
    gross_logs: tuple[float, ...],
    meaningful: tuple[bool, ...] | None = None,
    hard_risk_counts: tuple[int, ...] | None = None,
    rejection_counts: tuple[int, ...] | None = None,
):
    assert len(net_logs) == len(gross_logs)
    count = len(net_logs)
    resolved_meaningful = meaningful or (True,) * count
    resolved_risk = hard_risk_counts or (0,) * count
    resolved_rejections = rejection_counts or (0,) * count
    leaves = tuple(
        _leaf(
            cell="B",
            training_seed=0,
            concrete_symbol=f"DEV_{index}",
            tile=f"symbol-{index}",
            net_log_growth=net_log,
            gross_log_growth=gross_log,
            meaningful_execution=resolved_meaningful[index],
            hard_risk_violation_count=resolved_risk[index],
            unexplained_execution_rejection_count=resolved_rejections[index],
        )
        for index, (net_log, gross_log) in enumerate(
            zip(net_logs, gross_logs, strict=True)
        )
    )
    return _summarize(leaves=leaves)


@pytest.mark.parametrize("cell", ("B", "C1", "C2", "D1", "D2"))
def test_u2_primary_cell_gate_accepts_all_mandatory_cells_at_seed_zero(
    cell: str,
) -> None:
    module = _module()
    summary = _base_summary(cell=cell)

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert evidence.cell == cell
    assert evidence.training_seed == 0
    assert evidence.summary_digest == summary.digest
    assert evidence.rejection_reasons == ()
    assert evidence.passed is True


def test_u2_primary_cell_gate_binds_the_preregistered_threshold_payload() -> None:
    from trade_rl.workflows import universal_trade_rl_u2_contract

    module = _module()
    summary = _base_summary()

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    expected_threshold_digest = content_digest(
        universal_trade_rl_u2_contract._selection_thresholds_payload()
    )
    assert evidence.selection_thresholds_digest == expected_threshold_digest
    assert (
        evidence.digest
        == module.evaluate_universal_trade_rl_u2_primary_cell_gate(
            summary=summary
        ).digest
    )


@pytest.mark.parametrize("training_seed", (1, 2))
def test_u2_primary_cell_gate_rejects_nonprimary_training_seed(
    training_seed: int,
) -> None:
    module = _module()
    summary = _base_summary(training_seed=training_seed)

    with pytest.raises(ValueError, match="primary|seed|0"):
        module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)


@pytest.mark.parametrize("cell", ("A", "E", "unexpected"))
def test_u2_primary_cell_gate_rejects_nonmandatory_cell(cell: str) -> None:
    module = _module()
    summary = _base_summary(cell=cell)

    with pytest.raises(ValueError, match="mandatory|cell|B|C1|D1"):
        module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)


def test_u2_primary_cell_gate_balanced_gross_wealth_is_strict() -> None:
    module = _module()
    base = _base_summary()
    summary = _summary_with_symbol_logs(
        net_logs=(0.03, 0.02),
        gross_logs=(0.0, 0.0),
    )
    summary = replace(
        summary,
        positive_net_scope_fraction=base.positive_net_scope_fraction,
        scope_net_return_cvar10=base.scope_net_return_cvar10,
        digest="",
    )

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert summary.symbol_balanced_gross_wealth == 1.0
    assert REASON_GROSS in evidence.rejection_reasons
    assert evidence.passed is False


def test_u2_primary_cell_gate_balanced_net_wealth_is_strict() -> None:
    module = _module()
    base = _base_summary()
    summary = _summary_with_symbol_logs(
        net_logs=(0.0, 0.0),
        gross_logs=(0.02, 0.02),
    )
    summary = replace(
        summary,
        positive_net_scope_fraction=base.positive_net_scope_fraction,
        scope_net_return_cvar10=base.scope_net_return_cvar10,
        digest="",
    )

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert summary.symbol_balanced_net_wealth == 1.0
    assert REASON_NET in evidence.rejection_reasons
    assert evidence.passed is False


def test_u2_primary_cell_gate_median_symbol_wealth_below_one_rejects() -> None:
    module = _module()
    summary = _summary_with_symbol_logs(
        net_logs=(-0.02, -0.01, 0.10),
        gross_logs=(0.04, 0.04, 0.04),
    )
    summary = replace(
        summary,
        positive_net_scope_fraction=0.50,
        scope_net_return_cvar10=-0.01,
        digest="",
    )

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert summary.median_symbol_net_wealth < 1.0
    assert REASON_MEDIAN in evidence.rejection_reasons
    assert evidence.passed is False


def test_u2_primary_cell_gate_minimum_symbol_wealth_below_one_rejects() -> None:
    module = _module()
    summary = _summary_with_symbol_logs(
        net_logs=(-0.01, 0.02, 0.02),
        gross_logs=(0.02, 0.02, 0.02),
    )

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert summary.median_symbol_net_wealth > 1.0
    assert summary.minimum_symbol_net_wealth < 1.0
    assert evidence.rejection_reasons == (REASON_MINIMUM,)
    assert evidence.passed is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ({"positive_net_scope_fraction": 0.499999}, REASON_POSITIVE),
        ({"scope_net_return_cvar10": -0.010001}, REASON_CVAR),
        ({"turnover_per_day_p95": 1.000001}, REASON_TURNOVER),
    ),
)
def test_u2_primary_cell_gate_rejects_beyond_inclusive_numeric_boundaries(
    mutation: dict[str, float],
    reason: str,
) -> None:
    module = _module()
    summary = replace(_base_summary(), **mutation, digest="")

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert evidence.rejection_reasons == (reason,)
    assert evidence.passed is False


def test_u2_primary_cell_gate_requires_meaningful_execution_for_every_symbol() -> None:
    module = _module()
    summary = _summarize(
        leaves=(
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_A",
                tile="A-negative",
                net_log_growth=-0.01,
                gross_log_growth=0.01,
            ),
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_A",
                tile="A-positive",
                net_log_growth=0.04,
                gross_log_growth=0.03,
            ),
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_B",
                tile="B-flat",
                net_log_growth=0.0,
                gross_log_growth=0.01,
                meaningful_execution=False,
            ),
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_B",
                tile="B-positive",
                net_log_growth=0.02,
                gross_log_growth=0.02,
                meaningful_execution=False,
            ),
        )
    )

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert summary.meaningful_execution_symbol_fraction == 0.5
    assert evidence.rejection_reasons == (REASON_EXECUTION,)
    assert evidence.passed is False


def test_u2_primary_cell_gate_requires_zero_hard_risk_violations() -> None:
    module = _module()
    leaf = replace(
        _base_summary().symbol_metrics[0],
        hard_risk_violation_count=1,
        digest="",
    )
    summary = _base_summary()
    symbols = (leaf, *summary.symbol_metrics[1:])
    summary = replace(
        summary,
        symbol_metrics=symbols,
        hard_risk_violation_count=1,
        digest="",
    )

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert evidence.rejection_reasons == (REASON_RISK,)
    assert evidence.passed is False


def test_u2_primary_cell_gate_requires_zero_unexplained_rejections() -> None:
    module = _module()
    leaf = replace(
        _base_summary().symbol_metrics[0],
        unexplained_execution_rejection_count=1,
        digest="",
    )
    summary = _base_summary()
    symbols = (leaf, *summary.symbol_metrics[1:])
    summary = replace(
        summary,
        symbol_metrics=symbols,
        unexplained_execution_rejection_count=1,
        digest="",
    )

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert evidence.rejection_reasons == (REASON_REJECTION,)
    assert evidence.passed is False


def test_u2_primary_cell_gate_rejects_positive_gross_retention_below_half() -> None:
    module = _module()
    summary = _summarize(
        leaves=(
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_A",
                tile="A-negative",
                net_log_growth=-0.01,
                gross_log_growth=0.02,
            ),
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_A",
                tile="A-positive",
                net_log_growth=0.04,
                gross_log_growth=0.04,
            ),
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_B",
                tile="B-flat",
                net_log_growth=0.0,
                gross_log_growth=0.02,
            ),
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_B",
                tile="B-positive",
                net_log_growth=0.02,
                gross_log_growth=0.04,
            ),
        )
    )

    evidence = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert summary.positive_gross_log_growth_retention is not None
    assert summary.positive_gross_log_growth_retention < 0.50
    assert evidence.rejection_reasons == (REASON_RETENTION,)
    assert evidence.passed is False


def test_u2_primary_cell_gate_inclusive_boundaries_pass_exactly() -> None:
    module = _module()
    base = _base_summary()
    assert base.positive_net_scope_fraction == 0.50
    assert base.scope_net_return_cvar10 == -0.01
    assert base.turnover_per_day_p95 == 1.0
    assert module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=base).passed

    median_minimum_boundary = _summary_with_symbol_logs(
        net_logs=(0.0, 0.0, 0.03),
        gross_logs=(0.02, 0.02, 0.02),
    )
    median_minimum_boundary = replace(
        median_minimum_boundary,
        positive_net_scope_fraction=0.50,
        digest="",
    )
    assert median_minimum_boundary.median_symbol_net_wealth == 1.0
    assert median_minimum_boundary.minimum_symbol_net_wealth == 1.0
    assert median_minimum_boundary.positive_gross_log_growth_retention == 0.50
    assert module.evaluate_universal_trade_rl_u2_primary_cell_gate(
        summary=median_minimum_boundary
    ).passed


def test_u2_primary_cell_gate_reasons_are_canonical_and_deterministic() -> None:
    module = _module()
    summary = _summary_with_symbol_logs(
        net_logs=(0.0, 0.0),
        gross_logs=(0.0, 0.0),
        meaningful=(False, False),
        hard_risk_counts=(1, 0),
        rejection_counts=(0, 1),
    )
    summary = replace(
        summary,
        positive_net_scope_fraction=0.0,
        scope_net_return_cvar10=-0.02,
        turnover_per_day_p95=2.0,
        digest="",
    )

    first = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)
    second = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert first.rejection_reasons == (
        REASON_GROSS,
        REASON_NET,
        REASON_POSITIVE,
        REASON_CVAR,
        REASON_TURNOVER,
        REASON_EXECUTION,
        REASON_RISK,
        REASON_REJECTION,
    )
    assert first.rejection_reasons == second.rejection_reasons
    assert first.digest == second.digest
    assert first.passed is False
