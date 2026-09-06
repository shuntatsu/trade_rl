from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tests.workflows.test_universal_trade_rl_u2_selection_robustness import (
    _evaluate,
    _scope_leaves,
)

_MEDIAN_REASON = "median_seed_symbol_balanced_net_wealth_not_above_cash"
_WORST_REASON = "worst_seed_symbol_balanced_net_wealth_below_cash"


def test_u2_seed_robustness_rejects_fully_consistent_nonpositive_wealth_tamper() -> None:
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
            rejection_reasons=(_MEDIAN_REASON, _WORST_REASON),
            digest="",
        )
