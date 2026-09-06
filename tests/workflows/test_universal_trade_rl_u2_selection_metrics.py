from __future__ import annotations

import math
from dataclasses import replace
from statistics import median

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _leaf(
    *,
    training_seed: int,
    concrete_symbol: str,
    tile: str,
    net_log_growth: float,
    gross_log_growth: float,
    turnover_per_day: float,
    meaningful_execution: bool,
):
    module = _module()
    return module.UniversalTradeRLU2SelectionLeafMetrics(
        training_seed=training_seed,
        cell="B",
        concrete_symbol=concrete_symbol,
        tile_identity=content_digest({"tile": tile}),
        replay_evidence_digest=content_digest(
            {"replay": tile, "seed": training_seed, "symbol": concrete_symbol}
        ),
        leaf_net_log_growth=net_log_growth,
        leaf_gross_log_growth=gross_log_growth,
        turnover_per_day=turnover_per_day,
        meaningful_execution=meaningful_execution,
    )


def test_u2_selection_summary_executes_frozen_metric_formulas() -> None:
    module = _module()
    leaves = (
        _leaf(
            training_seed=0,
            concrete_symbol="DEV_A",
            tile="A0",
            net_log_growth=0.10,
            gross_log_growth=0.20,
            turnover_per_day=2.0,
            meaningful_execution=True,
        ),
        _leaf(
            training_seed=1,
            concrete_symbol="DEV_A",
            tile="A1",
            net_log_growth=-0.05,
            gross_log_growth=0.05,
            turnover_per_day=4.0,
            meaningful_execution=False,
        ),
        _leaf(
            training_seed=0,
            concrete_symbol="DEV_B",
            tile="B0",
            net_log_growth=0.02,
            gross_log_growth=0.04,
            turnover_per_day=6.0,
            meaningful_execution=False,
        ),
        _leaf(
            training_seed=1,
            concrete_symbol="DEV_B",
            tile="B1",
            net_log_growth=-0.10,
            gross_log_growth=0.01,
            turnover_per_day=8.0,
            meaningful_execution=True,
        ),
    )

    summary = module.summarize_universal_trade_rl_u2_selection_metrics(leaves=leaves)

    symbol_a_net_log = 0.10 - 0.05
    symbol_b_net_log = 0.02 - 0.10
    symbol_a_gross_log = 0.20 + 0.05
    symbol_b_gross_log = 0.04 + 0.01
    balanced_net_log = (symbol_a_net_log + symbol_b_net_log) / 2.0
    balanced_gross_log = (symbol_a_gross_log + symbol_b_gross_log) / 2.0
    symbol_net_wealth = (math.exp(symbol_a_net_log), math.exp(symbol_b_net_log))

    assert summary.leaf_count == 4
    assert summary.symbol_count == 2
    assert summary.symbol_balanced_net_log_growth == pytest.approx(balanced_net_log)
    assert summary.symbol_balanced_gross_log_growth == pytest.approx(
        balanced_gross_log
    )
    assert summary.symbol_balanced_net_wealth == pytest.approx(math.exp(balanced_net_log))
    assert summary.symbol_balanced_gross_wealth == pytest.approx(
        math.exp(balanced_gross_log)
    )
    assert summary.median_symbol_net_wealth == pytest.approx(median(symbol_net_wealth))
    assert summary.minimum_symbol_net_wealth == pytest.approx(min(symbol_net_wealth))
    assert summary.positive_net_scope_fraction == pytest.approx(0.5)
    assert summary.scope_net_return_cvar10 == pytest.approx(-0.10)
    assert summary.turnover_per_day_p95 == pytest.approx(
        float(np.quantile([2.0, 4.0, 6.0, 8.0], 0.95, method="linear"))
    )
    assert summary.meaningful_execution_symbol_fraction == pytest.approx(1.0)
    assert summary.positive_gross_log_growth_retention == pytest.approx(
        balanced_net_log / balanced_gross_log
    )

    by_symbol = {row.concrete_symbol: row for row in summary.symbol_metrics}
    assert by_symbol["DEV_A"].symbol_net_log_growth == pytest.approx(symbol_a_net_log)
    assert by_symbol["DEV_A"].symbol_gross_log_growth == pytest.approx(
        symbol_a_gross_log
    )
    assert by_symbol["DEV_A"].symbol_net_wealth == pytest.approx(
        math.exp(symbol_a_net_log)
    )
    assert by_symbol["DEV_A"].symbol_gross_wealth == pytest.approx(
        math.exp(symbol_a_gross_log)
    )
    assert by_symbol["DEV_A"].meaningful_execution is True
    assert by_symbol["DEV_B"].meaningful_execution is True


def test_u2_selection_retention_is_not_defined_for_nonpositive_gross_growth() -> None:
    module = _module()
    leaves = (
        _leaf(
            training_seed=0,
            concrete_symbol="DEV_A",
            tile="A0-negative-gross",
            net_log_growth=-0.20,
            gross_log_growth=-0.10,
            turnover_per_day=1.0,
            meaningful_execution=True,
        ),
        _leaf(
            training_seed=0,
            concrete_symbol="DEV_B",
            tile="B0-negative-gross",
            net_log_growth=-0.10,
            gross_log_growth=-0.05,
            turnover_per_day=1.0,
            meaningful_execution=True,
        ),
    )

    summary = module.summarize_universal_trade_rl_u2_selection_metrics(leaves=leaves)

    assert summary.symbol_balanced_gross_log_growth < 0.0
    assert summary.symbol_balanced_gross_wealth < 1.0
    assert summary.positive_gross_log_growth_retention is None


def test_u2_selection_summary_rejects_duplicate_leaf_identity() -> None:
    module = _module()
    leaf = _leaf(
        training_seed=0,
        concrete_symbol="DEV_A",
        tile="duplicate",
        net_log_growth=0.01,
        gross_log_growth=0.02,
        turnover_per_day=1.0,
        meaningful_execution=True,
    )

    with pytest.raises(ValueError, match="duplicate|identity|leaf"):
        module.summarize_universal_trade_rl_u2_selection_metrics(leaves=(leaf, leaf))


def test_u2_selection_summary_digest_binds_metric_values() -> None:
    module = _module()
    leaf = _leaf(
        training_seed=0,
        concrete_symbol="DEV_A",
        tile="digest",
        net_log_growth=0.01,
        gross_log_growth=0.02,
        turnover_per_day=1.0,
        meaningful_execution=True,
    )
    summary = module.summarize_universal_trade_rl_u2_selection_metrics(leaves=(leaf,))
    changed_leaf = replace(leaf, leaf_net_log_growth=0.02, digest="")
    changed = module.summarize_universal_trade_rl_u2_selection_metrics(
        leaves=(changed_leaf,)
    )

    assert changed.digest != summary.digest
