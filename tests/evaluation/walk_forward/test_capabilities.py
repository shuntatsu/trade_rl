from __future__ import annotations

import pytest

from trade_rl.evaluation.walk_forward.capabilities import RangeCapability
from trade_rl.evaluation.walk_forward.folds import IndexRange


def test_range_capability_rejects_escape_and_hides_parent() -> None:
    capability = RangeCapability(
        dataset_id="a" * 64,
        stage="selection",
        allowed=IndexRange(10, 20),
    )
    assert capability.require(IndexRange(12, 18)) == IndexRange(12, 18)
    assert not hasattr(capability, "dataset")
    with pytest.raises(ValueError, match="outside"):
        capability.require(IndexRange(9, 18))
