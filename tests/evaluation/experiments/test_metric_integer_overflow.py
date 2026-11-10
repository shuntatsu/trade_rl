"""M2 comparison readers must normalize unrepresentable integer metrics.

JSON permits huge integer tokens, but Python binary64 conversion can raise
OverflowError before the reader's finite-evidence admission check.
"""

from __future__ import annotations

import pytest

from trade_rl.evaluation.experiments.analysis import _require_metric_number
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.protocols import _selection_number


@pytest.mark.parametrize("value", (10**500, -(10**500)))
def test_candidate_metric_reader_rejects_huge_integer_as_integrity_error(
    value: int,
) -> None:
    with pytest.raises(
        ArtifactIntegrityError, match="candidate metric total_return is non-finite"
    ):
        _require_metric_number({"total_return": value}, "total_return")


@pytest.mark.parametrize("value", (10**500, -(10**500)))
def test_ppo_selection_reader_rejects_huge_integer_as_integrity_error(
    value: int,
) -> None:
    with pytest.raises(
        ArtifactIntegrityError,
        match=(
            "PPO holding-duration selection metric "
            "median_candidate_total_return is non-finite"
        ),
    ):
        _selection_number(
            {"median_candidate_total_return": value},
            "median_candidate_total_return",
        )


def test_representable_large_integer_metrics_keep_existing_float_semantics() -> None:
    finite = 10**308
    assert _require_metric_number({"total_return": finite}, "total_return") == float(
        finite
    )
    assert _selection_number(
        {"median_candidate_total_return": finite}, "median_candidate_total_return"
    ) == float(finite)
