from __future__ import annotations

import pytest

from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.evaluation.walk_forward.sealed_test import SealedTestLedger


def test_sealed_test_ledger_authorizes_each_plan_once() -> None:
    ledger = SealedTestLedger()
    record = ledger.authorize_once(
        experiment_plan_digest="1" * 64,
        dataset_id="2" * 64,
        fold_index=0,
        test_range=IndexRange(100, 120),
        selected_configuration="candidate",
        selected_policy_digest="3" * 64,
    )
    assert record.access_digest
    with pytest.raises(ValueError, match="already opened"):
        ledger.authorize_once(
            experiment_plan_digest="1" * 64,
            dataset_id="2" * 64,
            fold_index=0,
            test_range=IndexRange(100, 120),
            selected_configuration="candidate",
            selected_policy_digest="3" * 64,
        )
