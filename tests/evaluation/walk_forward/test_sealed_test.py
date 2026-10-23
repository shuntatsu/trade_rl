from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.robustness.walk_forward.folds import IndexRange
from trade_rl.evaluation.robustness.walk_forward.sealed_test import SealedTestLedger


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


def test_sealed_test_access_digest_is_stable_across_ledger_instances() -> None:
    first = SealedTestLedger().authorize_once(
        experiment_plan_digest="4" * 64,
        dataset_id="5" * 64,
        fold_index=2,
        test_range=IndexRange(200, 240),
        selected_configuration="candidate",
        selected_policy_digest="6" * 64,
    )
    second = SealedTestLedger().authorize_once(
        experiment_plan_digest="4" * 64,
        dataset_id="5" * 64,
        fold_index=2,
        test_range=IndexRange(200, 240),
        selected_configuration="candidate",
        selected_policy_digest="6" * 64,
    )
    assert first == second


def test_sealed_test_ledger_consumes_authorized_record_once() -> None:
    ledger = SealedTestLedger()
    record = ledger.authorize_once(
        experiment_plan_digest="7" * 64,
        dataset_id="8" * 64,
        fold_index=3,
        test_range=IndexRange(300, 340),
        selected_configuration="candidate",
        selected_policy_digest="9" * 64,
    )
    rebuilt = replace(record)
    assert rebuilt == record and rebuilt is not record
    assert ledger.consumed_access_digests == ()
    assert ledger.consume_once(rebuilt) == record
    assert ledger.consumed_access_digests == (record.access_digest,)
    with pytest.raises(ValueError, match="already consumed"):
        ledger.consume_once(record)


def test_sealed_test_ledger_rejects_consumption_without_issued_authority() -> None:
    foreign = SealedTestLedger().authorize_once(
        experiment_plan_digest="a" * 64,
        dataset_id="b" * 64,
        fold_index=4,
        test_range=IndexRange(400, 440),
        selected_configuration="candidate",
        selected_policy_digest="c" * 64,
    )
    ledger = SealedTestLedger()
    with pytest.raises(ValueError, match="not authorized"):
        ledger.consume_once(foreign)
    assert ledger.consumed_access_digests == ()


def test_sealed_test_batch_consumption_is_atomic_if_one_record_is_spent() -> None:
    ledger = SealedTestLedger()
    first = ledger.authorize_once(
        experiment_plan_digest="d" * 64,
        dataset_id="e" * 64,
        fold_index=5,
        test_range=IndexRange(500, 520),
        selected_configuration="first",
        selected_policy_digest="1" * 64,
    )
    second = ledger.authorize_once(
        experiment_plan_digest="d" * 64,
        dataset_id="e" * 64,
        fold_index=6,
        test_range=IndexRange(520, 540),
        selected_configuration="second",
        selected_policy_digest="2" * 64,
    )
    ledger.consume_once(first)
    with pytest.raises(ValueError, match="already consumed"):
        ledger.consume_all_once((second, first))
    assert ledger.consumed_access_digests == (first.access_digest,)
