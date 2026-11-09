"""Attempt-3 lineage v4 and the ``carry_paper_screen_v3`` plan.

The lineage names attempt 2 as an INVALID predecessor (cause: Issue #810 comment
6711911740, a software defect) and records that the attempt-2 seal is not reused. The
screen v3 plan equals ``carry_paper_screen_v2`` in every economic field; only the
operational collection-quality clause is new. Synthetic identities and tmp dirs only:
no attempt-1 or attempt-2 root, price, equity or timing value is read.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper import assessment, operations
from trade_rl.evaluation.paper import collection_policy as policy
from trade_rl.evaluation.paper import predecessor as pred
from trade_rl.evaluation.paper.account import PaperSettings
from trade_rl.evaluation.paper.operations import (
    screen_plan,
    screen_plan_v3,
    seal_paper_study,
    validate_attempt_lineage,
    validate_attempt_lineage_v4,
)

NOW = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
START = NOW + timedelta(hours=2)
ATTEMPT2_DIGEST = hashlib.sha256(b"synthetic attempt-2 protocol").hexdigest()
ATTEMPT2_TIP = hashlib.sha256(b"synthetic attempt-2 tip").hexdigest()
ATTEMPT2_OBSERVED = "2026-11-07T00:59:00.123456+00:00"
ROOT_PATH = "C:\\synthetic\\attempt2-root"


def lineage(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = dict(
        protocol_sha256=ATTEMPT2_DIGEST,
        final_tip_sha256=ATTEMPT2_TIP,
        last_observed_at=ATTEMPT2_OBSERVED,
        event_count=121,
        root_path=ROOT_PATH,
    )
    values.update(overrides)
    return pred.canonical_lineage_v4(**values)


def test_lineage_v4_names_attempt_two_as_an_invalidated_software_defect_predecessor():
    record = lineage()

    assert (
        record["schema"] == pred.LINEAGE_V4_SCHEMA == "carry_paper_attempt_lineage_v4"
    )
    assert record["attempt_number"] == 3
    assert record["predecessor"] == {
        "attempt_number": 2,
        "protocol_sha256": ATTEMPT2_DIGEST,
        "final_tip_sha256": ATTEMPT2_TIP,
        "disposition": "invalidated",
        "reason_codes": ["collector_failure"],
        "last_observed_at": ATTEMPT2_OBSERVED,
    }
    assert record["predecessor_root"]["status"] == "available"
    proof = record["predecessor_root"]["verification"]
    assert proof["reader"] == pred.ROOT_READER
    assert proof["root_path"] == ROOT_PATH and proof["event_count"] == 121
    cause = record["predecessor_invalidation"]
    assert cause["cause_record"].endswith("issues/810#issuecomment-6711911740")
    assert cause["kind"] == "software_defect"
    assert cause["economic_outcome_used"] is False
    assert record["predecessor_seal"] == {
        "reused": False,
        "successor_digest_must_differ": True,
    }
    # A lineage cannot encode an economic outcome: only the registered non-economic
    # vocabulary is used.
    assert set(record["predecessor"]["reason_codes"]) <= (
        operations._NON_ECONOMIC_ATTEMPT_REASONS
    )
    assert record["predecessor"]["disposition"] in operations._ATTEMPT_DISPOSITIONS


def test_the_classifier_and_validator_accept_the_canonical_record():
    record = lineage()
    assert pred.lineage_structure_problems(record) == []
    assert validate_attempt_lineage_v4(record, sealed_at=NOW) == record
    assert pred.lineage_structure_problems(None) == []


def _mutations() -> dict[str, Any]:
    def edit(path: tuple[str, ...], value: Any):
        def change(record: dict[str, Any]) -> None:
            node = record
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value

        return change

    def drop(path: tuple[str, ...]):
        def change(record: dict[str, Any]) -> None:
            node = record
            for key in path[:-1]:
                node = node[key]
            del node[path[-1]]

        return change

    return {
        "attempt_number": edit(("attempt_number",), 2),
        "attempt_number_float": edit(("attempt_number",), 3.0),
        "attempt_number_bool": edit(("attempt_number",), True),
        "predecessor_number": edit(("predecessor", "attempt_number"), 1),
        "disposition": edit(("predecessor", "disposition"), "incomplete"),
        "reasons": edit(("predecessor", "reason_codes"), ["observation_gap"]),
        "extra_reason": edit(
            ("predecessor", "reason_codes"), ["clock_reversal", "collector_failure"]
        ),
        "status": edit(("predecessor_root", "status"), "unavailable"),
        "reader": edit(("predecessor_root", "verification", "reader"), "other"),
        "event_count_bool": edit(
            ("predecessor_root", "verification", "event_count"), True
        ),
        "event_count_zero": edit(
            ("predecessor_root", "verification", "event_count"), 0
        ),
        "digest_mismatch": edit(
            ("predecessor_root", "verification", "protocol_sha256"), "b" * 64
        ),
        "tip_mismatch": edit(
            ("predecessor_root", "verification", "final_tip_sha256"), "c" * 64
        ),
        "observed_mismatch": edit(
            ("predecessor_root", "verification", "last_observed_at"),
            "2026-11-07T01:00:00+00:00",
        ),
        "empty_root_path": edit(("predecessor_root", "verification", "root_path"), ""),
        "short_digest": edit(("predecessor", "protocol_sha256"), "a" * 63),
        "upper_digest": edit(("predecessor", "protocol_sha256"), "A" * 64),
        "attempt_one_digest": edit(
            ("predecessor", "protocol_sha256"), pred.ATTEMPT_ONE_PROTOCOL_SHA256
        ),
        "cause_record": edit(("predecessor_invalidation", "cause_record"), "x"),
        "cause_kind": edit(("predecessor_invalidation", "kind"), "economic"),
        "outcome_used": edit(
            ("predecessor_invalidation", "economic_outcome_used"), True
        ),
        "seal_reused": edit(("predecessor_seal", "reused"), True),
        "digest_may_match": edit(
            ("predecessor_seal", "successor_digest_must_differ"), False
        ),
        "missing_invalidation": drop(("predecessor_invalidation",)),
        "missing_seal": drop(("predecessor_seal",)),
        "missing_root": drop(("predecessor_root",)),
        "extra_field": edit(("net_profit",), 1.0),
        "extra_nested_field": edit(("predecessor", "profit"), 1.0),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_any_departure_from_the_canonical_record_is_refused(name):
    record = lineage()
    _mutations()[name](record)

    assert pred.lineage_structure_problems(record) != []
    with pytest.raises(ValueError):
        validate_attempt_lineage_v4(record, sealed_at=NOW)


def test_a_str_subclass_key_cannot_hide_the_v4_rule():
    class Key(str):
        def __hash__(self) -> int:
            return hash("schema")

    record = lineage()
    record[Key("schema")] = record.pop("schema")
    assert pred.lineage_structure_problems(record) != []


def test_v4_validation_checks_time_ordering_against_the_seal():
    with pytest.raises(ValueError, match="newer than the seal"):
        validate_attempt_lineage_v4(
            lineage(last_observed_at=(NOW + timedelta(minutes=1)).isoformat()),
            sealed_at=NOW,
        )
    with pytest.raises(ValueError, match="last_observation_invalid"):
        validate_attempt_lineage_v4(
            lineage(last_observed_at="2026-11-07T00:59:00Z"), sealed_at=NOW
        )


def test_v2_and_v3_machinery_does_not_accept_a_v4_lineage():
    record = lineage()
    with pytest.raises(ValueError):
        validate_attempt_lineage(record)
    with pytest.raises(ValueError):
        screen_plan(attempt_lineage=record)
    v3 = pred.canonical_lineage("unavailable")
    with pytest.raises(ValueError):
        screen_plan_v3(attempt_lineage=v3)
    with pytest.raises(ValueError):
        validate_attempt_lineage_v4(v3)
    attempt_one = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": 1,
        "predecessor": None,
    }
    with pytest.raises(ValueError):
        screen_plan_v3(attempt_lineage=attempt_one)


def test_each_lineage_version_needs_its_own_private_seal_token(tmp_path):
    plan = screen_plan_v3(attempt_lineage=lineage())
    with pytest.raises(ValueError, match="attempt3"):
        pred.require_seal_token(plan, None)
    with pytest.raises(ValueError, match="attempt3"):
        pred.require_seal_token(plan, pred._ATTEMPT2_SEAL_TOKEN)
    pred.require_seal_token(plan, pred._ATTEMPT3_SEAL_TOKEN)
    v3_plan = screen_plan(attempt_lineage=pred.canonical_lineage("unavailable"))
    with pytest.raises(ValueError, match="attempt2"):
        pred.require_seal_token(v3_plan, pred._ATTEMPT3_SEAL_TOKEN)
    pred.require_seal_token(v3_plan, pred._ATTEMPT2_SEAL_TOKEN)

    with pytest.raises(ValueError, match="attempt3"):
        seal_paper_study(
            tmp_path / "nope",
            start_at=START,
            attempt_lineage=lineage(),
            clock=lambda: NOW,
        )
    assert not (tmp_path / "nope").exists()
    with pytest.raises(ValueError, match="attempt3"):
        seal_paper_study(
            tmp_path / "nope",
            start_at=START,
            attempt_lineage=lineage(),
            lineage_seal_token=pred._ATTEMPT2_SEAL_TOKEN,
            clock=lambda: NOW,
        )


def test_sealing_v4_binds_screen_v3_and_a_digest_that_differs_from_every_earlier_one(
    tmp_path,
):
    digest = seal_paper_study(
        tmp_path / "root",
        start_at=START,
        attempt_lineage=lineage(),
        lineage_seal_token=pred._ATTEMPT3_SEAL_TOKEN,
        clock=lambda: NOW,
    )
    protocol = json.loads((tmp_path / "root" / "protocol.json").read_bytes())
    plan = protocol["protocol"]["study"]["research_plan"]

    assert plan["schema"] == "carry_paper_screen_v3"
    assert plan == screen_plan_v3(attempt_lineage=lineage())
    assert digest not in {ATTEMPT2_DIGEST, pred.ATTEMPT_ONE_PROTOCOL_SHA256}
    assert (
        hashlib.sha256((tmp_path / "root" / "protocol.json").read_bytes()).hexdigest()
        == digest
    )


# ---------------------------------------------------------------- screen v3


def _economic(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"schema", "attempt_lineage", "operational_quality"}
    }


def test_screen_v3_economic_fields_equal_screen_v2_byte_for_byte():
    first = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": 1,
        "predecessor": None,
    }
    v2 = screen_plan(attempt_lineage=first)
    v3 = screen_plan_v3(attempt_lineage=lineage())

    assert v3["schema"] == "carry_paper_screen_v3"
    assert set(v3) == set(v2) | {"operational_quality"}
    assert canonical_json_bytes(_economic(v3)) == canonical_json_bytes(_economic(v2))
    assert _economic(v3)["production_eligible"] is False
    assert _economic(v3)["maximum_drawdown_exclusive"] == 0.1
    assert _economic(v3)["complete_coverage_and_actual_flat"] is True


def test_the_operational_clause_binds_the_registered_policy_constants():
    clause = screen_plan_v3(attempt_lineage=lineage())["operational_quality"]

    assert clause["failed_capture_rejects_screen"] is False
    assert clause["unregistered_failed_capture_rejects_screen"] is True
    assert clause["terminal_failure_classes"] == ["B", "D", "I"]
    assert clause["retries_per_slot"] == policy.RETRIES_PER_SLOT
    assert clause["retry_delay_seconds"] == policy.RETRY_DELAY_SECONDS
    assert clause["retry_start_window_seconds"] == policy.RETRY_START_WINDOW_SECONDS
    assert (
        clause["max_consecutive_skipped_slots"] == policy.MAX_CONSECUTIVE_SKIPPED_SLOTS
    )
    assert clause["max_skipped_slots_per_hour"] == policy.MAX_SKIPPED_SLOTS_PER_HOUR
    assert clause["max_skipped_slots_total"] == policy.MAX_SKIPPED_SLOTS_TOTAL
    assert clause["skip_window_slots"] == policy.SKIP_WINDOW_SLOTS
    assert clause["cadence_seconds"] == policy.CADENCE_SECONDS
    assert clause["frozen_gap_limit_seconds"] == 180
    assert clause["frozen_gap_limit_seconds"] == int(
        PaperSettings.__dataclass_fields__["maximum_gap_seconds"].default
    )
    assert clause["rules_hold_seconds"] == policy.RULES_HOLD_SECONDS
    assert clause["skip_reasons"] == list(policy.SKIP_REASONS)
    assert clause["attempt_purposes"] == list(policy.ATTEMPT_PURPOSES)
    assert clause["max_failed_captures_per_slot"] == policy.MAX_FAILED_CAPTURES_PER_SLOT
    assert clause["report_skip_count_and_funding_proximity"] is True
    assert clause["claim_boundary"] == "retained_slots_only"


def test_the_economic_screen_functions_are_the_ones_pinned_at_main():
    """Screen v3 reuses the unchanged economic code; editing it breaks this pin.

    The digests are of the source text at the commit that added screen v3 (current
    ``main``), with line endings normalised.
    """

    def digest(obj: object) -> str:
        text = inspect.getsource(obj).replace("\r\n", "\n")  # type: ignore[arg-type]
        return hashlib.sha256(text.encode()).hexdigest()

    assert digest(assessment.screen_reasons) == PINNED["screen_reasons"]
    assert digest(assessment._ObservationAccumulator) == PINNED["accumulator"]
    assert digest(assessment.Measurements) == PINNED["measurements"]


PINNED = {
    "screen_reasons": "9a612fe1742c6d01b924795c833ed04041da5b8a5c5b1912def5098f234d3999",
    "accumulator": "5fab383b8636f1cb9337cd9c2a0794f127281d8343a788a7aed1c6597fd6ca90",
    "measurements": "0bee964eebdb233d9d4e8f516f642648847d7c6a65f585220e6f1013bebd6f3f",
}


def test_screen_v3_changes_only_the_operational_clause_not_the_reason_vocabulary():
    reasons = assessment.screen_reasons(
        assessment.Measurements(
            net_profit=0,
            recorded_fees=0,
            block_returns=(),
            block_funding_counts=(),
            maximum_drawdown=1,
            first_delay_seconds=999,
            final_grace_seconds=0,
            maximum_gap_seconds=999,
            terminal_flat=False,
            pending=True,
            stop_reason="x",
            quality_failures=("x",),
            unpaid_funding=({"a": "b"},),
            filled_symbols=(),
            post_close_funding=True,
        )
    )
    assert set(reasons) == {
        "nonpositive_net_profit",
        "insufficient_fee_headroom",
        "nonpositive_block_return",
        "missing_block_funding",
        "post_close_funding",
        "drawdown_limit",
        "initial_coverage_gap",
        "observation_coverage_gap",
        "terminal_coverage_gap",
        "not_flat",
        "pending_intent",
        "nonterminal_stop",
        "quality_failure",
        "unpaid_funding",
        "missing_instrument_fills",
    }


# ------------------------------------------------- predecessor re-verification


def report(
    *,
    events: int = 121,
    tip: str = ATTEMPT2_TIP,
    observed: str | None = ATTEMPT2_OBSERVED,
):
    def read(_path: Any, *, expected_protocol_sha256: str, read_only: bool) -> Any:
        assert expected_protocol_sha256 == ATTEMPT2_DIGEST and read_only is True
        return {"events": events, "tip": tip, "status": {"observed_at": observed}}

    return read


def make_root(tmp_path: Path, *, marker: bool = True) -> Path:
    root = tmp_path / "attempt2"
    root.mkdir()
    if marker:
        (root / "collection-failure.json").write_text("{}")
    return root


IDENTITY = pred.PredecessorIdentity(
    protocol_sha256=ATTEMPT2_DIGEST,
    final_tip_sha256=ATTEMPT2_TIP,
    last_observed_at=ATTEMPT2_OBSERVED,
    event_count=121,
)


def test_verify_predecessor_root_accepts_matching_identity_and_a_failure_marker(
    tmp_path,
):
    root = make_root(tmp_path)
    assert (
        pred.verify_predecessor_root(
            root, read=report(), identity=IDENTITY, require_failure_marker=True
        )
        == []
    )


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"events": 120}, "predecessor_root_event_count_differs"),
        ({"tip": "d" * 64}, "predecessor_root_tip_differs"),
        (
            {"observed": "2026-11-07T01:00:00+00:00"},
            "predecessor_root_last_observation_differs",
        ),
    ],
)
def test_verify_predecessor_root_reports_each_difference(tmp_path, kwargs, reason):
    assert reason in pred.verify_predecessor_root(
        make_root(tmp_path), read=report(**kwargs), identity=IDENTITY
    )


def test_a_predecessor_without_a_failure_marker_is_not_an_invalid_predecessor(tmp_path):
    root = make_root(tmp_path, marker=False)
    assert "predecessor_root_failure_marker_missing" in pred.verify_predecessor_root(
        root, read=report(), identity=IDENTITY, require_failure_marker=True
    )
    # The marker is only checked for existence; it is never read.
    assert pred.verify_predecessor_root(root, read=report(), identity=IDENTITY) == []


def test_the_attempt_one_default_identity_is_unchanged():
    assert pred.attempt_one_identity() == pred.PredecessorIdentity(
        protocol_sha256=pred.ATTEMPT_ONE_PROTOCOL_SHA256,
        final_tip_sha256=pred.ATTEMPT_ONE_FINAL_TIP_SHA256,
        last_observed_at=pred.ATTEMPT_ONE_LAST_OBSERVED_AT,
        event_count=pred.ATTEMPT_ONE_EVENT_COUNT,
    )


def test_a_v4_available_root_is_re_read_on_every_entry(tmp_path):
    root = make_root(tmp_path)
    record = lineage(root_path=str(root))
    assert pred.available_root_problems(record, read=report()) == []

    broken = pred.available_root_problems(record, read=report(events=1))
    assert broken and broken[0].startswith("predecessor_root_not_reverifiable:")

    def unreadable(*_a: Any, **_k: Any) -> Any:
        raise OSError("gone")

    assert pred.available_root_problems(record, read=unreadable)
    (root / "collection-failure.json").unlink()
    assert pred.available_root_problems(record, read=report())


def test_collection_lineage_problems_gate_v4_like_every_other_version(tmp_path):
    root = make_root(tmp_path)
    plan = screen_plan_v3(attempt_lineage=lineage(root_path=str(root)))

    assert (
        pred.collection_lineage_problems(
            plan, root=tmp_path, start_at=START, now=lambda: NOW, read=report()
        )
        == []
    )
    assert pred.collection_lineage_problems(
        plan, root=tmp_path, start_at=START, now=lambda: NOW, read=report(events=2)
    )
    drifted = copy.deepcopy(plan)
    drifted["predecessor_extra"] = 1
    # The plan itself is compared by the evaluator and the launch gate; the collector
    # gate refuses a screen v3 plan that carries any other lineage, and vice versa.
    mixed = copy.deepcopy(plan)
    mixed["attempt_lineage"] = pred.canonical_lineage("unavailable")
    assert "screen_lineage_mismatch" in pred.collection_lineage_problems(
        mixed, root=tmp_path, start_at=START, now=lambda: NOW, read=report()
    )
    reverse = screen_plan(attempt_lineage=pred.canonical_lineage("unavailable"))
    reverse["attempt_lineage"] = lineage(root_path=str(root))
    assert "screen_lineage_mismatch" in pred.collection_lineage_problems(
        reverse, root=tmp_path, start_at=START, now=lambda: NOW, read=report()
    )
