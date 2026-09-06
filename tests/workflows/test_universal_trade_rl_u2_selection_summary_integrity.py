from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tests.workflows.test_universal_trade_rl_u2_selection_metrics import _leaf
from trade_rl.artifacts.hashing import content_digest


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _summary():
    module = _module()
    leaves = (
        _leaf(
            training_seed=0,
            concrete_symbol="DEV_A",
            tile="integrity-A0",
            net_log_growth=0.10,
            gross_log_growth=0.20,
            turnover_per_day=2.0,
            meaningful_execution=True,
        ),
        _leaf(
            training_seed=1,
            concrete_symbol="DEV_A",
            tile="integrity-A1",
            net_log_growth=-0.05,
            gross_log_growth=0.05,
            turnover_per_day=4.0,
            meaningful_execution=False,
        ),
        _leaf(
            training_seed=0,
            concrete_symbol="DEV_B",
            tile="integrity-B0",
            net_log_growth=0.02,
            gross_log_growth=0.04,
            turnover_per_day=6.0,
            meaningful_execution=False,
        ),
        _leaf(
            training_seed=1,
            concrete_symbol="DEV_B",
            tile="integrity-B1",
            net_log_growth=-0.10,
            gross_log_growth=0.01,
            turnover_per_day=8.0,
            meaningful_execution=True,
        ),
    )
    return module.summarize_universal_trade_rl_u2_selection_metrics(leaves=leaves)


def test_u2_selection_summary_rejects_symbol_leaf_digest_closure_drift() -> None:
    summary = _summary()
    drifted_symbol = replace(
        summary.symbol_metrics[0],
        leaf_digests=(content_digest({"fixture": "unbound-selection-leaf"}),),
        digest="",
    )

    with pytest.raises(ValueError, match="leaf|closure|digest|symbol"):
        replace(
            summary,
            symbol_metrics=(drifted_symbol, *summary.symbol_metrics[1:]),
            digest="",
        )


def test_u2_selection_summary_rejects_duplicate_symbol_identity() -> None:
    summary = _summary()
    duplicate_symbol = replace(
        summary.symbol_metrics[1],
        concrete_symbol=summary.symbol_metrics[0].concrete_symbol,
        digest="",
    )

    with pytest.raises(ValueError, match="symbol|unique|duplicate|closure"):
        replace(
            summary,
            symbol_metrics=(summary.symbol_metrics[0], duplicate_symbol),
            digest="",
        )


def test_u2_selection_summary_rejects_symbol_derived_metric_drift() -> None:
    summary = _summary()
    bad_net_log = summary.symbol_balanced_net_log_growth + 0.125
    bad_gross_log = summary.symbol_balanced_gross_log_growth + 0.125

    mutations = (
        {
            "symbol_balanced_net_log_growth": bad_net_log,
            "symbol_balanced_net_wealth": math.exp(bad_net_log),
        },
        {
            "symbol_balanced_gross_log_growth": bad_gross_log,
            "symbol_balanced_gross_wealth": math.exp(bad_gross_log),
            "positive_gross_log_growth_retention": (
                summary.symbol_balanced_net_log_growth / bad_gross_log
            ),
        },
        {"median_symbol_net_wealth": summary.median_symbol_net_wealth + 0.1},
        {"minimum_symbol_net_wealth": summary.minimum_symbol_net_wealth + 0.1},
        {
            "meaningful_execution_symbol_fraction": (
                0.0 if summary.meaningful_execution_symbol_fraction != 0.0 else 1.0
            )
        },
        {
            "positive_gross_log_growth_retention": (
                summary.positive_gross_log_growth_retention + 0.1
                if summary.positive_gross_log_growth_retention is not None
                else 0.1
            )
        },
    )

    for mutation in mutations:
        with pytest.raises(
            ValueError, match="Selection|symbol|retention|wealth|consistent"
        ):
            replace(summary, **mutation, digest="")
