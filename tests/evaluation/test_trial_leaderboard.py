from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from trade_rl.evaluation.comparison.leaderboard import (
    FamilyLeaderboard,
    VarianceSource,
    build_family_leaderboard,
)
from trade_rl.evaluation.comparison.multiple_testing import (
    SharpeMoments,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
)
from trade_rl.evaluation.comparison.trial_ledger import (
    TrialLedger,
    TrialLedgerError,
    TrialRecord,
    TrialStatus,
)

DIGEST = "b" * 64


def _completed(
    trial_id: str, sharpe: float, *, family: str = "fam", n_obs: int = 250
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        family=family,
        method=f"method-{trial_id}",
        config_digest=DIGEST,
        status=TrialStatus.COMPLETED,
        performance=SharpeMoments(sharpe, n_obs, -0.3, 4.0),
        evaluation_data_label="known-used dev window",
    )


def _failed(trial_id: str, *, family: str = "fam") -> TrialRecord:
    return TrialRecord(trial_id, family, "m", DIGEST, TrialStatus.FAILED, reason="nan")


def _abandoned(trial_id: str, *, family: str = "fam") -> TrialRecord:
    return TrialRecord(
        trial_id, family, "m", DIGEST, TrialStatus.ABANDONED, reason="stopped"
    )


def _plain_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def test_leaderboard_ranks_by_dsr_and_reports_the_inputs() -> None:
    ledger = TrialLedger(
        [
            _completed("low", 0.05),
            _completed("high", 0.15),
            _completed("mid", 0.10),
        ]
    )
    board = build_family_leaderboard(ledger, "fam")
    assert [item.trial_id for item in board.entries] == ["high", "mid", "low"]
    assert [item.rank for item in board.entries] == [1, 2, 3]
    expected_variance = _plain_variance([0.05, 0.15, 0.10])
    assert board.trial_sr_variance == pytest.approx(expected_variance, rel=1e-12)
    assert board.variance_source is VarianceSource.COMPLETED_TRIALS
    assert board.n_trials == 3 and board.n_completed == 3
    assert board.expected_max_sharpe == pytest.approx(
        expected_maximum_sharpe(3, expected_variance), rel=1e-12
    )
    assert board.ledger_digest == ledger.digest()
    top = board.entries[0]
    assert top.dsr == pytest.approx(
        deflated_sharpe_ratio(0.15, 250, -0.3, 4.0, 3, expected_variance), abs=1e-12
    )
    assert top.psr_zero == pytest.approx(
        probabilistic_sharpe_ratio(0.15, 250, -0.3, 4.0), abs=1e-12
    )
    assert top.evaluation_data_label == "known-used dev window"
    assert top.method == "method-high"
    assert all(item.dsr <= item.psr_zero for item in board.entries)


def test_failed_and_abandoned_trials_count_toward_multiplicity() -> None:
    completed = [_completed("a", 0.05), _completed("b", 0.12)]
    only_completed = build_family_leaderboard(TrialLedger(completed), "fam")
    with_failures = build_family_leaderboard(
        TrialLedger([*completed, _failed("f1"), _failed("f2"), _abandoned("x1")]),
        "fam",
    )
    assert only_completed.n_trials == 2
    assert with_failures.n_trials == 5
    assert (with_failures.n_completed, with_failures.n_failed) == (2, 2)
    assert with_failures.n_abandoned == 1
    # V comes from the two completed trials only, so it is unchanged...
    assert with_failures.trial_sr_variance == only_completed.trial_sr_variance
    # ...but E[max] and every DSR reflect the larger N.
    assert with_failures.expected_max_sharpe > only_completed.expected_max_sharpe
    for before, after in zip(
        only_completed.entries, with_failures.entries, strict=True
    ):
        assert before.trial_id == after.trial_id
        assert after.dsr < before.dsr
        assert after.psr_zero == before.psr_zero


def test_only_the_requested_family_is_counted() -> None:
    ledger = TrialLedger(
        [
            _completed("a", 0.05),
            _completed("b", 0.12),
            _completed("o1", 0.2, family="other"),
            _failed("o2", family="other"),
        ]
    )
    board = build_family_leaderboard(ledger, "fam")
    assert board.n_trials == 2
    assert {item.trial_id for item in board.entries} == {"a", "b"}
    assert (
        build_family_leaderboard(ledger, "other", trial_sr_variance=0.01).n_trials == 2
    )


def test_single_trial_family_with_supplied_variance_matches_psr() -> None:
    ledger = TrialLedger([_completed("only", 0.1)])
    board = build_family_leaderboard(ledger, "fam", trial_sr_variance=0.5)
    assert board.variance_source is VarianceSource.SUPPLIED
    assert board.n_trials == 1
    assert board.expected_max_sharpe == 0.0
    assert board.entries[0].dsr == board.entries[0].psr_zero


def test_failures_with_one_completed_trial_use_supplied_variance() -> None:
    ledger = TrialLedger([_completed("only", 0.1), _failed("f"), _failed("g")])
    board = build_family_leaderboard(ledger, "fam", trial_sr_variance=0.01)
    assert board.n_trials == 3
    assert board.expected_max_sharpe == pytest.approx(
        expected_maximum_sharpe(3, 0.01), rel=1e-12
    )


def test_no_completed_trials_gives_empty_ranking_with_supplied_variance() -> None:
    ledger = TrialLedger([_failed("f"), _abandoned("a")])
    board = build_family_leaderboard(ledger, "fam", trial_sr_variance=0.04)
    assert board.entries == ()
    assert (board.n_trials, board.n_completed) == (2, 0)


def test_ties_break_by_trial_id_deterministically() -> None:
    ledger = TrialLedger(
        [_completed("b", 0.1), _completed("a", 0.1), _completed("c", 0.2)]
    )
    board = build_family_leaderboard(ledger, "fam")
    assert [item.trial_id for item in board.entries] == ["c", "a", "b"]


def test_return_series_trials_are_scored_from_derived_moments() -> None:
    with_returns = TrialRecord(
        "r",
        "fam",
        "m",
        DIGEST,
        TrialStatus.COMPLETED,
        returns=(0.01, 0.02, -0.01, 0.015, 0.0, 0.03),
    )
    ledger = TrialLedger([with_returns, _completed("s", 0.05)])
    board = build_family_leaderboard(ledger, "fam")
    derived = with_returns.moments()
    row = next(item for item in board.entries if item.trial_id == "r")
    assert row.sharpe == derived.sharpe and row.n_obs == 6


def test_leaderboard_is_immutable_and_independent_of_append_order() -> None:
    records = [_completed("a", 0.05), _completed("b", 0.12), _failed("f")]
    forward = build_family_leaderboard(TrialLedger(records), "fam")
    backward = build_family_leaderboard(TrialLedger(reversed(records)), "fam")
    assert forward.entries == backward.entries
    assert forward.expected_max_sharpe == backward.expected_max_sharpe
    assert isinstance(forward.entries, tuple)
    with pytest.raises(FrozenInstanceError):
        forward.n_trials = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        forward.entries[0].dsr = 1.0  # type: ignore[misc]


def test_leaderboard_has_no_claim_fields() -> None:
    board = build_family_leaderboard(
        TrialLedger([_completed("a", 0.05), _completed("b", 0.1)]), "fam"
    )
    names = set(board.__slots__) | set(board.entries[0].__slots__)
    assert not {n for n in names if "pass" in n or "verdict" in n or "claim" in n}
    for documented in (FamilyLeaderboard, build_family_leaderboard):
        assert "not out-of-sample evidence" in (documented.__doc__ or "")


def test_leaderboard_refuses_unknown_family_and_bad_arguments() -> None:
    ledger = TrialLedger([_completed("a", 0.05), _completed("b", 0.12)])
    with pytest.raises(TrialLedgerError, match="no recorded trials"):
        build_family_leaderboard(ledger, "missing")
    with pytest.raises(TrialLedgerError, match="family must be a string"):
        build_family_leaderboard(ledger, 3)  # type: ignore[arg-type]
    with pytest.raises(TrialLedgerError, match="TrialLedger"):
        build_family_leaderboard([], "fam")  # type: ignore[arg-type]


def test_leaderboard_refuses_to_invent_variance() -> None:
    ledger = TrialLedger([_completed("only", 0.1), _failed("f")])
    with pytest.raises(TrialLedgerError, match="supply trial_sr_variance"):
        build_family_leaderboard(ledger, "fam")
    empty_completed = TrialLedger([_failed("f", family="z")])
    with pytest.raises(TrialLedgerError, match="supply trial_sr_variance"):
        build_family_leaderboard(empty_completed, "z")


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, True, "0.1"])
def test_leaderboard_refuses_inadmissible_supplied_variance(bad: object) -> None:
    ledger = TrialLedger([_completed("only", 0.1)])
    with pytest.raises(TrialLedgerError, match="trial_sr_variance"):
        build_family_leaderboard(
            ledger,
            "fam",
            trial_sr_variance=bad,  # type: ignore[arg-type]
        )


def test_leaderboard_refuses_supplied_variance_when_two_trials_completed() -> None:
    ledger = TrialLedger([_completed("a", 0.05), _completed("b", 0.12)])
    with pytest.raises(TrialLedgerError, match="fewer than two"):
        build_family_leaderboard(ledger, "fam", trial_sr_variance=0.01)


def test_leaderboard_refuses_trial_with_non_positive_psr_variance_term() -> None:
    degenerate = TrialRecord(
        "bad",
        "fam",
        "m",
        DIGEST,
        TrialStatus.COMPLETED,
        performance=SharpeMoments(1.0, 50, 2.0, 5.0),
    )
    ledger = TrialLedger([degenerate, _completed("ok", 0.1)])
    with pytest.raises(TrialLedgerError, match="'bad' cannot be scored"):
        build_family_leaderboard(ledger, "fam")


def test_saturated_dsr_trials_are_ordered_by_z_not_by_name() -> None:
    # Huge samples drive both DSRs to exactly 1.0; the id order is the reverse
    # of the Sharpe order, so a trial_id-only tie break would be wrong.
    def trial(trial_id: str, sharpe: float) -> TrialRecord:
        return TrialRecord(
            trial_id,
            "fam",
            "m",
            DIGEST,
            TrialStatus.COMPLETED,
            performance=SharpeMoments(sharpe, 1_000_000, 0.0, 3.0),
        )

    ledger = TrialLedger([trial("a-weaker", 0.5), trial("z-stronger", 0.6)])
    board = build_family_leaderboard(ledger, "fam")
    assert [item.dsr for item in board.entries] == [1.0, 1.0]
    assert [item.trial_id for item in board.entries] == ["z-stronger", "a-weaker"]
    assert board.entries[0].dsr_z > board.entries[1].dsr_z > 8.0
    assert [item.rank for item in board.entries] == [1, 2]


def test_exact_ties_still_break_by_trial_id() -> None:
    ledger = TrialLedger([_completed("b", 0.1), _completed("a", 0.1)])
    assert [e.trial_id for e in build_family_leaderboard(ledger, "fam").entries] == [
        "a",
        "b",
    ]


@pytest.mark.parametrize("late_family", ["fam", "other"])
def test_leaderboard_digest_binds_one_snapshot_during_append(monkeypatch, late_family):
    ledger = TrialLedger([_completed("a", 0.05), _completed("b", 0.12)])
    expected = build_family_leaderboard(ledger, "fam")
    original = TrialRecord.moments
    inserted = False

    def moments(record):
        nonlocal inserted
        if not inserted:
            inserted = True
            assert ledger.append(_failed("late", family=late_family))
        return original(record)

    # Deterministic interleaving: the writer appends after scoring has begun.
    monkeypatch.setattr(TrialRecord, "moments", moments)
    actual = build_family_leaderboard(ledger, "fam")
    assert inserted and len(ledger) == 3
    assert ledger.digest() != expected.ledger_digest
    assert actual == expected
    updated = build_family_leaderboard(ledger, "fam")
    assert updated.n_trials == (3 if late_family == "fam" else 2)
    assert updated.ledger_digest == ledger.digest()


@pytest.mark.parametrize("entrypoint", ["family_records", "leaderboard"])
def test_family_selection_refuses_foreign_text_before_equality(entrypoint):
    class ForeignFamily(str):
        def __eq__(self, other):
            pytest.fail("foreign family equality must not run")

    ledger = TrialLedger(
        [_completed("a", 0.05), _completed("b", 0.12, family="other")]
    )
    family = ForeignFamily("absent")
    with pytest.raises(TrialLedgerError, match="family"):
        if entrypoint == "family_records":
            ledger.family_records(family)
        else:
            build_family_leaderboard(ledger, family)
    assert [record.trial_id for record in ledger.family_records("fam")] == ["a"]
