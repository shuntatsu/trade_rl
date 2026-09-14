from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_evaluation import (
    canonical_mean_reversion_economic_gate_evaluation_spec,
    research_status_from_counts,
)


def test_evaluation_spec_rejects_bool_integer_alias() -> None:
    spec = canonical_mean_reversion_economic_gate_evaluation_spec()

    with pytest.raises(ValueError, match="frozen evaluation"):
        replace(spec, production_eligible=0)
    with pytest.raises(ValueError, match="frozen evaluation"):
        replace(spec, final_test_authorized=0)


def test_research_status_rejects_counts_outside_frozen_roster() -> None:
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        research_status_from_counts(
            positive_effect_symbols=6,
            median_excess_total_return=0.01,
            cost_reduction_symbols=5,
            turnover_reduction_symbols=5,
            drawdown_nonworse_symbols=5,
            new_termination_symbols=0,
        )
