"""The attempt-3 pre-registration packet stays UNREVIEWED and bound to the source.

Attempt 3 is a pre-registration: it states the timing tolerances and a proposed
collection-failure policy before any attempt-3 data exists. These tests keep the packet,
the specification and the code in step, and keep the packet from claiming a review, a
seal, an implementation or any economic result.
"""

from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_paper_attempt2_assurance_packet import (
    paper_code_dependency_paths,
)
from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.paper import account, supervisor
from trade_rl.evaluation.paper.account import PaperSettings
from trade_rl.evaluation.paper.attempt2 import SCREEN_SLOTS
from trade_rl.integrations.binance.forward import (
    MAX_CAPTURE_SPAN_SECONDS,
    MAX_CONSUME_LATENCY_MS,
    MAX_QUOTE_AGE_MS,
    MAX_QUOTE_FUTURE_SKEW_MS,
    MAX_REQUEST_SECONDS,
)
from trade_rl.simulation import depth as depth_module

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_PACKET = "report/issue810-carry-attempt3-research-assurance-20261108.json"
PACKET = ROOT / RELATIVE_PACKET
PINNED_DIGEST = "59990524a1acc0ed4ba1cdcbceaf87e52c27a67a7f5d1d1b5c163562835ff1dc"
ATTEMPT2_59990524a1acc0ed4ba1cdcbceaf87e52c27a67a7f5d1d1b5c163562835ff1dc = (
    "617988282ad6dfbf0ee7188f002676a1675dfac38c9ac969071bee3091b2e770"
)
ATTEMPT1_PROTOCOL_DIGEST = (
    "843467870d85d0b085e65cf904e9d458287c14fc1e31615b9aaf77029f0669b7"
)
G0_EXCEPTION_URL = (
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6680576996"
)
SPEC = "docs/specs/funding-carry-paper.md"
ARCHITECTURE = "docs/architecture/research-assurance.md"
STATUS = "docs/research/current-status.md"
BOUNDARIES = "docs/architecture/package-boundaries.md"
DOCUMENTS = (SPEC, ARCHITECTURE, STATUS)
TIMING_PREFIX = "Timing protocol parameters:"
POLICY_PREFIX = "Collection policy parameters:"
PENDING = "PENDING_OWNER_AUTHORIZATION"


def _packet() -> Any:
    return json.loads(PACKET.read_text(encoding="utf-8"))


def _text(packet: Any) -> str:
    return json.dumps(packet, ensure_ascii=False)


def _raw(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _flat(relative: str) -> str:
    return " ".join(_raw(relative).split())


def _section(relative: str, marker: str, end: str | None = None) -> str:
    document = _raw(relative)
    start = document.index(marker)
    stop = document.index(end, start) if end else len(document)
    return " ".join(document[start:stop].split())


def _spec_section() -> str:
    return _section(SPEC, "### Attempt 3 pre-registration")


def _arch_section() -> str:
    return _section(
        ARCHITECTURE,
        "#### Attempt 3 author packet (pre-implementation pre-registration)",
        "**Known limitations:**",
    )


def _status_paragraph() -> str:
    return _section(STATUS, "2026-11-08: funding-carry attempt 2", "\n\n")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _parameters(text: str, prefix: str) -> dict[str, int]:
    """The single distinct ``name=value`` list stated after ``prefix`` (repeats equal)."""
    found = set(re.findall(re.escape(prefix) + r" ([^.]*)\.", text))
    assert len(found) == 1, (prefix, found)
    values: dict[str, int] = {}
    for item in found.pop().split("; "):
        name, _, number = item.partition("=")
        assert re.fullmatch(r"[a-z_]+", name) and re.fullmatch(r"\d+", number), item
        assert name not in values, name
        values[name] = int(number)
    return values


def _source_literals() -> dict[str, int]:
    """Timing values read from the code, constants where they exist."""
    return {
        "quote_age_ms": MAX_QUOTE_AGE_MS,
        "future_skew_ms": MAX_QUOTE_FUTURE_SKEW_MS,
        "request_seconds": MAX_REQUEST_SECONDS,
        "capture_span_seconds": MAX_CAPTURE_SPAN_SECONDS,
        "consume_latency_ms": MAX_CONSUME_LATENCY_MS,
        "decision_window_seconds": 10,
        "depth_receipt_age_seconds": 5,
        "worst_case_mark_age_ms": MAX_QUOTE_AGE_MS + MAX_CONSUME_LATENCY_MS,
    }


def _mismatches(stated: dict[str, int]) -> list[str]:
    source = _source_literals()
    problems = [name for name in source if stated.get(name) != source[name]]
    problems += sorted(set(stated) - set(source))
    return problems


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", "assurance", *args, RELATIVE_PACKET],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_packet_is_unreviewed_valid_and_pins_its_digest() -> None:
    packet = _packet()

    result = evaluate_assurance(packet)

    assert result.status == "UNREVIEWED"
    assert result.errors == ()
    assert result.economic_execution_authorized is False
    assert result.record_digest == assurance_digest(packet) == PINNED_DIGEST
    assert (
        PINNED_DIGEST
        != ATTEMPT2_59990524a1acc0ed4ba1cdcbceaf87e52c27a67a7f5d1d1b5c163562835ff1dc
    )
    assert b"\r" not in PACKET.read_bytes()


def test_cli_check_inspect_and_digest_agree_and_report_unreviewed() -> None:
    check, inspect, digest = _cli("check"), _cli("inspect"), _cli("digest")

    assert check.returncode == 1, check.stderr
    assert json.loads(check.stdout)["status"] == "UNREVIEWED"
    assert inspect.returncode == 0, inspect.stderr
    assert json.loads(inspect.stdout)["record_digest"] == PINNED_DIGEST
    assert json.loads(inspect.stdout)["economic_execution_authorized"] is False
    assert digest.returncode == 0, digest.stderr
    assert json.loads(digest.stdout) == {"record_digest": PINNED_DIGEST}


def test_any_edit_changes_the_digest_and_a_self_authored_review_stays_blocked() -> None:
    packet = _packet()
    edited = copy.deepcopy(packet)
    edited["thesis"]["stop_rule"] += " "
    assert assurance_digest(edited) != PINNED_DIGEST

    identity = packet["identity"]
    review = {
        "schema_version": "research_assurance_review_v1",
        "reviewed_record_digest": PINNED_DIGEST,
        "protocol_head": identity["protocol_head"],
        "implementation_head": identity["implementation_head"],
        "reviewer_identity": "self",
        "reviewer_execution_identity": "self-run",
        "reviewer_independence": "ESTABLISHED",
        "result_blind": True,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "PASS",
        "status": "pass",
        "disposition": "ASSURANCE_PASS",
        "blocking_findings": [],
        "missing_evidence": [],
        "rationale": "author supplied",
        "limitations": ["none"],
    }
    result = evaluate_assurance(packet, review=review)
    assert result.status == "BLOCKED"
    assert result.economic_execution_authorized is False


def test_packet_grants_no_authorization_and_claims_no_review_or_implementation() -> (
    None
):
    packet = _packet()
    claims, thesis = packet["claims"], packet["thesis"]
    limitations = " ".join(thesis["limitations"].split())

    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["claim_level"] == "software_validity"
    assert claims["next_action_class"] == "documentation_only"
    assert packet["evidence"]["evidence_level"] == "software_validity"
    assert thesis["stage"] == "prospective_paper"
    assert "No PASS is claimed" in limitations
    assert "owner decision point" in limitations
    assert "is NOT implemented" in limitations
    assert "can never support a seal" in thesis["stop_rule"]
    assert G0_EXCEPTION_URL in thesis["stop_rule"]
    assert "does not extend to attempt 3" in thesis["stop_rule"]
    assert "ordinary route" in limitations
    forbidden = " ".join(claims["forbidden"])
    for required in (
        "sealing, launching or collecting attempt 3",
        "reusing the attempt-2 seal",
        "attempt-2 timing metadata",
        "predecessor-root-unavailable G0 exception",
        "treating the proposed collection policy as implemented",
    ):
        assert required in forbidden, required
    mechanism = packet["mechanism"]
    assert set(mechanism["changed_boundaries"]) == {"time", "state", "evidence"}
    assert set(mechanism["independent_oracles"]) == {"time", "state", "evidence"}
    for boundary in ("unit", "sign", "risk", "execution", "accounting"):
        assert "Unchanged" in mechanism["authorities"][boundary], boundary
    cited = set(re.findall(r"tests/[\w/]+\.py", _text(packet)))
    assert cited
    assert sorted(path for path in cited if not (ROOT / path).is_file()) == []
    state = mechanism["independent_oracles"]["state"]
    assert "no implementation and no test yet" in state["oracle"]
    assert "NOT_ESTABLISHED" in state["known_limitations"]


def test_packet_timing_values_equal_the_source_constants() -> None:
    packet = _packet()
    stated = _parameters(_text(packet), TIMING_PREFIX)

    assert _mismatches(stated) == []
    assert (
        stated["worst_case_mark_age_ms"]
        == stated["quote_age_ms"] + (stated["consume_latency_ms"])
    )
    # The ten-second consumption bound is the paper decision-to-execution window.
    assert stated["consume_latency_ms"] == stated["decision_window_seconds"] * 1000
    # A capture may span as long as the decision window but a request is shorter.
    assert stated["capture_span_seconds"] <= stated["decision_window_seconds"]
    assert stated["request_seconds"] * 1000 == stated["quote_age_ms"]


@pytest.mark.parametrize("name", sorted(_source_literals()))
def test_a_diverging_packet_value_is_detected(name: str) -> None:
    packet = _packet()
    stated = _parameters(_text(packet), TIMING_PREFIX)
    assert _mismatches(stated) == []

    assert _mismatches({**stated, name: stated[name] + 1}) == [name]
    assert _mismatches({key: v for key, v in stated.items() if key != name}) == [name]
    assert _mismatches({**stated, "unknown_tolerance": 1}) == ["unknown_tolerance"]


def test_literal_timing_values_are_pinned_in_the_sealed_paper_source() -> None:
    """The decision window and depth receipt age are literals in sealed paper code."""
    paper = Path(account.__file__).parent
    stated = _parameters(_text(_packet()), TIMING_PREFIX)
    window = stated["decision_window_seconds"]
    receipt_age = stated["depth_receipt_age_seconds"]

    assert f"(at - decision_at).total_seconds() > {window}:" in (
        paper / "account.py"
    ).read_text(encoding="utf-8")
    assert f"(at - decision_at).total_seconds() <= {window}:" in (
        paper / "supervisor.py"
    ).read_text(encoding="utf-8")
    # The formatter splits the second expression across lines, so ignore whitespace.
    depth = re.sub(r"\s+", "", Path(depth_module.__file__).read_text(encoding="utf-8"))
    for expression in (
        f"(execution_at - quote_at).total_seconds() > {receipt_age}",
        f"(execution_at - decision_at).total_seconds() > {window}",
    ):
        assert re.sub(r"\s+", "", expression) in depth, expression


def test_the_specification_states_the_same_parameters_as_the_packet() -> None:
    packet = _packet()
    spec = _spec_section()

    assert _parameters(spec, TIMING_PREFIX) == _parameters(_text(packet), TIMING_PREFIX)
    assert _parameters(spec, POLICY_PREFIX) == _parameters(_text(packet), POLICY_PREFIX)
    for name in (
        "MAX_QUOTE_AGE_MS",
        "MAX_QUOTE_FUTURE_SKEW_MS",
        "MAX_REQUEST_SECONDS",
        "MAX_CAPTURE_SPAN_SECONDS",
        "MAX_CONSUME_LATENCY_MS",
    ):
        assert name in spec, name
        assert name in _flat(BOUNDARIES), name
        assert name in _arch_section(), name
    assert "owner decision point" in spec
    assert "not implemented" in spec


def test_collection_policy_is_derived_from_the_frozen_cadence_and_slot_count() -> None:
    policy = _parameters(_text(_packet()), POLICY_PREFIX)
    stated = _parameters(_text(_packet()), TIMING_PREFIX)
    cadence = policy["cadence_seconds"]
    gap_limit = policy["frozen_gap_limit_seconds"]

    # Frozen values the policy must not move.
    default_gap = PaperSettings.__dataclass_fields__["maximum_gap_seconds"].default
    assert gap_limit == default_gap == 180
    assert "interval_seconds=60" in Path(supervisor.__file__).read_text(
        encoding="utf-8"
    )
    assert cadence == 60
    assert SCREEN_SLOTS == 129_604

    # One skipped slot is tolerated, two consecutive ones reach the frozen limit.
    consecutive = policy["max_consecutive_skipped_slots"]
    assert (consecutive + 1) * cadence < gap_limit <= (consecutive + 2) * cadence
    assert consecutive == 1

    # A retried cycle (decision capture and execution capture) fits inside one slot.
    spans = 2 * stated["capture_span_seconds"]
    assert policy["retry_start_window_seconds"] + spans <= cadence
    assert policy["retry_start_window_seconds"] >= policy["retry_delay_seconds"]

    # A retry after a worst-case failed execution capture cannot fit the decision window.
    assert (
        policy["retry_delay_seconds"] + stated["capture_span_seconds"]
        > stated["decision_window_seconds"]
    )

    hour = 3600 // cadence
    assert policy["retries_per_slot"] == 2
    assert consecutive <= policy["max_skipped_slots_per_hour"] < hour
    assert policy["max_skipped_slots_total"] == math.floor(SCREEN_SLOTS / 1000)
    assert policy["max_skipped_slots_total"] < SCREEN_SLOTS


def test_policy_text_forbids_the_unsafe_retries_and_claims_no_implementation() -> None:
    packet = _packet()
    evidence = " ".join(packet["mechanism"]["authorities"]["evidence"].split())
    spec = _spec_section()

    for text in (evidence, spec):
        assert "Class D" in text and "Class I" in text and "Class T" in text
        assert "429" in text and "451" in text
        assert "never after a decision" in text
        assert "carry_paper_screen_v3" in text
        assert "carry_paper_screen_v2" in text
    assert "a consumed capture is never discarded" in spec
    assert "a skip cannot extend it" in spec
    assert "A death inside a cycle is still never resumed" in spec
    assert "keep halt-on-first-failure" in spec
    assert "relaxes one attempt-2 operational-quality clause" in spec
    assert "relaxing no economic criterion" in spec
    assert "does not extend to attempt 3" in spec

    supervisor_source = Path(supervisor.__file__).read_text(encoding="utf-8")
    # The policy is pre-registered, not built: the collector still halts on any error.
    assert (
        "except BaseException as error:\n            self._halt(error)\n            raise"
        in (supervisor_source)
    )


def test_run_identity_is_new_and_the_attempt2_seal_is_not_reused() -> None:
    packet = _packet()
    limitations = packet["thesis"]["limitations"]
    spec = _spec_section()
    backslash = chr(92)
    attempt3_root = (
        "C:"
        + backslash
        + "trade_rl_runs"
        + backslash
        + "attempt3"
        + backslash
        + "root-"
    )
    attempt2_root = "C:" + backslash + "trade_rl_runs" + backslash + "attempt2"

    for text in (limitations, spec):
        assert attempt3_root + "<UTC seal date YYYYMMDD>" in text
        assert "storage-probe-<UTC date YYYYMMDD>" in text
        assert "attempt3-<first 8 hex of the" in text
        assert "carry-a3-supervisor" in text and "carry-a3-check" in text
    assert "never reused" in limitations and "never reuse" in spec
    assert "must differ from every earlier digest" in limitations
    assert "must differ from every earlier digest" in spec
    # No attempt-3 location is placed beneath an attempt-2 location.
    for text in (limitations, spec):
        paths = re.findall(r"C:\\trade_rl_runs\\attempt3\\[^\s`;,]*", text)
        assert paths
        for path in paths:
            assert "attempt2" not in path, path
    assert not attempt3_root.startswith(attempt2_root)
    # Attempt-2 artefacts are named only to say that they stay untouched.
    assert "not touched" in limitations and "stay untouched" in spec


def test_seal_start_authorization_and_launch_fields_are_pending_and_nothing_is_sealed() -> (
    None
):
    packet_text = _text(_packet())
    spec = _spec_section()
    arch = _arch_section()
    status = _status_paragraph()

    assert packet_text.count(PENDING) >= 1
    assert spec.count(PENDING) >= 6
    for label in (
        "Collection policy decision",
        "Review route",
        "Authorization comment",
        "Screen start (UTC)",
        "Protocol digest (seal)",
        "Launch commit",
    ):
        assert re.search(re.escape(label) + r" \| `" + PENDING + r"`", spec), label
    assert PENDING in arch and PENDING in status
    allowed = {
        PINNED_DIGEST,
        ATTEMPT2_59990524a1acc0ed4ba1cdcbceaf87e52c27a67a7f5d1d1b5c163562835ff1dc,
    }
    for name, text in (("packet", packet_text), ("spec", spec), ("status", status)):
        digests = set(re.findall(r"\b[0-9a-f]{64}\b", text))
        extra = (
            digests
            - allowed
            - ({ATTEMPT1_PROTOCOL_DIGEST} if name == "packet" else set())
        )
        assert extra == set(), (name, extra)
    assert "no seal was computed" in packet_text
    assert "nothing is sealed, launched, probed or collected" in spec


def test_packet_and_new_text_are_result_blind() -> None:
    packet = _packet()
    text = _text(packet)
    hypothesis = packet["thesis"]["hypothesis"]
    forbidden = (
        re.compile(
            r"\bpass(?:ed|es)\b[^.;]{0,60}\b(?:screens?|criteria|preregistered)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bpast development profit", re.IGNORECASE),
        re.compile(
            r"\bdevelopment (?:replay|run|result)s?[^.;]{0,40}\b(?:profit|passed|positive)",
            re.IGNORECASE,
        ),
        # Attempt-2 timing or outcome figures must not appear as evidence.
        re.compile(r"0\.25\s*[-\u2013]\s*0\.65"),
        re.compile(r"observed consume latency", re.IGNORECASE),
        re.compile(r"\b60 completed cycles\b", re.IGNORECASE),
    )
    sections = {
        "packet": text,
        "spec": _spec_section(),
        "architecture": _arch_section(),
        "status": _status_paragraph(),
    }
    for pattern in forbidden:
        for name, section in sections.items():
            assert pattern.search(section) is None, (name, pattern.pattern)
    for relative in (SPEC, STATUS, BOUNDARIES):
        assert "0.25-0.65" not in _raw(relative), relative
    assert "reused-development evidence only" in hypothesis
    assert "no development economic outcome is recorded" in hypothesis
    limitations = " ".join(packet["thesis"]["limitations"].split())
    assert "no attempt-1 or attempt-2 price, equity" in limitations.lower()
    assert "attempt-2 timing metadata" in limitations
    assert "deliberately not used" in limitations
    assert (
        "no attempt-1 or attempt-2 price" in " ".join(_spec_section().split()).lower()
    )


def test_documents_point_to_the_packet_digest_and_unreviewed_status() -> None:
    for relative in DOCUMENTS:
        document = _raw(relative)
        assert RELATIVE_PACKET in document, relative
        assert PINNED_DIGEST in document, relative
        assert "UNREVIEWED" in document, relative
    assert RELATIVE_PACKET in _raw(BOUNDARIES)
    plan = _raw("docs/plans/funding-carry-paper.md")
    assert RELATIVE_PACKET in plan
    assert "Status: Active" in plan[:200]
    assert "no PASS claimed" in _arch_section()


def test_attempt2_review_exception_is_not_extended_to_attempt3() -> None:
    packet = _packet()
    for text in (
        packet["thesis"]["stop_rule"],
        " ".join(_spec_section().split()),
        _arch_section(),
    ):
        assert G0_EXCEPTION_URL.rsplit("-", 1)[-1] in text
        assert "does not extend to attempt 3" in text
    assert "different authenticated GitHub principal" in _spec_section()
    assert "different-principal" in _arch_section()
    limitations = " ".join(packet["thesis"]["limitations"].split())
    assert "same-account exception does not apply" in limitations


def test_heads_bind_the_packet_to_the_constant_refactor_and_unimplemented_policy() -> (
    None
):
    identity = _packet()["identity"]
    head = identity["implementation_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    assert identity["protocol_head"] == head
    assert head in _raw(SPEC)
    assert head[:12] in _raw(ARCHITECTURE)

    if _git("cat-file", "-e", f"{head}^{{commit}}").returncode != 0:
        pytest.skip("implementation_head is not available in this checkout")
    forward = _git("show", f"{head}:trade_rl/integrations/binance/forward.py")
    assert forward.returncode == 0, forward.stderr
    assert "MAX_REQUEST_SECONDS = 5" in forward.stdout
    collector = _git("show", f"{head}:trade_rl/evaluation/paper/supervisor.py")
    assert collector.returncode == 0, collector.stderr
    assert (
        "except BaseException as error:\n            self._halt(error)\n            raise"
        in collector.stdout
    )
    assert "skipped_slot" not in collector.stdout

    published = _git("log", "-1", "--format=%H", "--", RELATIVE_PACKET)
    packet_commit = published.stdout.strip()
    if (
        published.returncode != 0
        or not re.fullmatch(r"[0-9a-f]{40}", packet_commit)
        or _git("merge-base", "--is-ancestor", head, packet_commit).returncode != 0
    ):
        pytest.skip(
            "packet source commit is unavailable in a shallow/exported checkout"
        )
    paths = paper_code_dependency_paths(head)
    assert "trade_rl/integrations/binance/forward.py" in paths
    changed = _git("diff", "--name-only", head, packet_commit, "--", *paths)
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.strip() == "", (
        "packet-bound code changed after implementation_head; a new packet and digest "
        "are required: " + changed.stdout
    )
