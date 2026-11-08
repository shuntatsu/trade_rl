"""The attempt-3 implementation packet stays UNREVIEWED and bound to the source.

The pre-registration packet is frozen (``test_paper_attempt3_assurance_packet.py``).
This packet claims that the registered collection-failure policy is implemented in the
collector. These tests import the policy constants from the source, so the packet, the
specification and the code cannot diverge, and they keep the packet from claiming a
review, a seal or any economic result.
"""

from __future__ import annotations

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
from tests.evaluation.test_paper_attempt3_assurance_packet import (
    DELEGATION_ID,
    G0_EXCEPTION_URL,
    PENDING,
    POLICY_PREFIX,
    TIMING_PREFIX,
    _mismatches,
    _parameters,
    _source_literals,
)
from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.paper import collection_policy as policy
from trade_rl.evaluation.paper.account import PaperSettings
from trade_rl.evaluation.paper.attempt2 import SCREEN_SLOTS

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_PACKET = (
    "report/issue810-carry-attempt3-implementation-research-assurance-20261108.json"
)
PACKET = ROOT / RELATIVE_PACKET
PINNED_DIGEST = "eeb2fcaa84c161bee4490a5ab0617a63ddcc38c4eb383f420379c7d175350d45"
PREREG_PACKET = "report/issue810-carry-attempt3-research-assurance-20261108.json"
PREREG_DIGEST = "29467597161a8856c97e2b9c46094e185bef84eed304ac7f33f75edf1ddbea10"
PREREG_HEAD = "ec7acfd1f7ca51afa805edf2051e32eff764591b"
SPEC = "docs/specs/funding-carry-paper.md"
ARCHITECTURE = "docs/architecture/research-assurance.md"
STATUS = "docs/research/current-status.md"
BOUNDARIES = "docs/architecture/package-boundaries.md"
PLAN = "docs/plans/funding-carry-paper.md"
DOCUMENTS = (SPEC, ARCHITECTURE, STATUS)


def _packet() -> Any:
    return json.loads(PACKET.read_text(encoding="utf-8"))


def _text(packet: Any) -> str:
    return json.dumps(packet, ensure_ascii=False)


def _raw(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _section(relative: str, marker: str, end: str | None = None) -> str:
    document = _raw(relative)
    start = document.index(marker)
    stop = document.index(end, start) if end else len(document)
    return " ".join(document[start:stop].split())


def _spec_section() -> str:
    return _section(
        SPEC,
        "#### Attempt 3 collection-policy implementation",
        "**Run identity and seal procedure",
    )


def _arch_section() -> str:
    return _section(
        ARCHITECTURE,
        "#### Attempt 3 implementation packet",
        "**Known limitations:**",
    )


def _status_paragraph() -> str:
    return _section(
        STATUS, "2026-11-08: the pre-registered attempt-3 collection policy", "\n\n"
    )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", "assurance", *args, RELATIVE_PACKET],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _policy_source() -> dict[str, int]:
    """Policy values read from the imported source, never typed here."""
    return {
        "retries_per_slot": policy.RETRIES_PER_SLOT,
        "retry_delay_seconds": policy.RETRY_DELAY_SECONDS,
        "retry_start_window_seconds": policy.RETRY_START_WINDOW_SECONDS,
        "max_consecutive_skipped_slots": policy.MAX_CONSECUTIVE_SKIPPED_SLOTS,
        "max_skipped_slots_per_hour": policy.MAX_SKIPPED_SLOTS_PER_HOUR,
        "max_skipped_slots_total": policy.MAX_SKIPPED_SLOTS_TOTAL,
        "cadence_seconds": policy.CADENCE_SECONDS,
        "frozen_gap_limit_seconds": int(
            PaperSettings.__dataclass_fields__["maximum_gap_seconds"].default
        ),
    }


def _policy_mismatches(stated: dict[str, int]) -> list[str]:
    source = _policy_source()
    problems = [name for name in source if stated.get(name) != source[name]]
    problems += sorted(set(stated) - set(source))
    return problems


def test_packet_is_unreviewed_valid_and_pins_its_digest() -> None:
    packet = _packet()

    result = evaluate_assurance(packet)

    assert result.status == "UNREVIEWED"
    assert result.errors == ()
    assert result.economic_execution_authorized is False
    assert result.record_digest == assurance_digest(packet) == PINNED_DIGEST
    assert PINNED_DIGEST != PREREG_DIGEST
    assert b"\r" not in PACKET.read_bytes()
    identity = packet["identity"]
    assert identity["protocol_head"] == identity["implementation_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", identity["implementation_head"])
    assert identity["implementation_head"] != PREREG_HEAD


def test_cli_check_inspect_and_digest_agree_and_report_unreviewed() -> None:
    check, inspect, digest = _cli("check"), _cli("inspect"), _cli("digest")

    assert check.returncode == 1, check.stderr
    assert json.loads(check.stdout)["status"] == "UNREVIEWED"
    assert inspect.returncode == 0, inspect.stderr
    assert json.loads(inspect.stdout)["record_digest"] == PINNED_DIGEST
    assert json.loads(inspect.stdout)["economic_execution_authorized"] is False
    assert digest.returncode == 0, digest.stderr
    assert json.loads(digest.stdout) == {"record_digest": PINNED_DIGEST}


def test_the_pre_registration_packet_is_untouched_and_not_re_pinned() -> None:
    frozen = json.loads((ROOT / PREREG_PACKET).read_text(encoding="utf-8"))
    assert assurance_digest(frozen) == PREREG_DIGEST
    assert frozen["identity"]["implementation_head"] == PREREG_HEAD


def test_packet_grants_no_authorization_and_claims_no_review_or_seal() -> None:
    packet = _packet()
    claims, thesis = packet["claims"], packet["thesis"]
    limitations = " ".join(thesis["limitations"].split())
    stop_rule = " ".join(thesis["stop_rule"].split())

    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["claim_level"] == "software_validity"
    assert claims["next_action_class"] == "documentation_only"
    assert packet["evidence"]["evidence_level"] == "software_validity"
    assert thesis["stage"] == "prospective_paper"
    assert "No PASS is claimed" in limitations
    assert DELEGATION_ID in limitations and "not the owner's drafting" in limitations
    assert "IMPLEMENTED at implementation_head" in limitations
    assert "cannot support a seal" in stop_rule
    assert G0_EXCEPTION_URL in stop_rule
    assert "does not extend to attempt 3" in stop_rule
    assert "not different-principal independence" in limitations
    assert "authorizes no seal, launch, probe or economic run" in limitations
    forbidden = " ".join(claims["forbidden"])
    for required in (
        "sealing, launching or collecting attempt 3",
        "treating the implementation as reviewed, as sufficient for a seal",
        "reusing the attempt-2 seal",
        "attempt-2 timing metadata",
        "predecessor-root-unavailable G0 exception",
        "funding-proximity distribution",
    ):
        assert required in forbidden, required
    assert "is not reviewed" in " ".join(claims["permitted"])
    # What is not built is stated, so the packet cannot be read as a complete attempt.
    for missing in ("carry_paper_screen_v3", "evaluator audit", "attempt-lineage"):
        assert missing in limitations and missing in stop_rule, missing
    assert set(packet["mechanism"]["changed_boundaries"]) == {
        "time",
        "state",
        "evidence",
    }
    for boundary in ("unit", "sign", "risk", "execution", "accounting"):
        assert "Unchanged" in packet["mechanism"]["authorities"][boundary], boundary
    cited = set(re.findall(r"tests/[\w/]+\.py", _text(packet)))
    assert cited
    assert sorted(path for path in cited if not (ROOT / path).is_file()) == []
    modules = set(re.findall(r"trade_rl/[\w/]+\.py", _text(packet)))
    assert sorted(path for path in modules if not (ROOT / path).is_file()) == []


def test_packet_and_spec_policy_values_equal_the_source_constants() -> None:
    packet = _packet()
    stated = _parameters(_text(packet), POLICY_PREFIX)
    assert _policy_mismatches(stated) == []
    # The pre-registered spec text states the same values as the implementation packet.
    spec = " ".join(_raw(SPEC).split())
    from_spec = _parameters(
        spec[spec.index("### Attempt 3 pre-registration") :], POLICY_PREFIX
    )
    assert from_spec == stated


@pytest.mark.parametrize("name", sorted(_policy_source()))
def test_a_diverging_policy_value_is_detected(name: str) -> None:
    stated = _parameters(_text(_packet()), POLICY_PREFIX)
    assert _policy_mismatches(stated) == []

    assert _policy_mismatches({**stated, name: stated[name] + 1}) == [name]
    assert _policy_mismatches({k: v for k, v in stated.items() if k != name}) == [name]
    assert _policy_mismatches({**stated, "unknown_budget": 1}) == ["unknown_budget"]


def test_packet_timing_values_equal_the_source_constants() -> None:
    stated = _parameters(_text(_packet()), TIMING_PREFIX)

    assert _mismatches(stated) == []
    assert set(stated) == set(_source_literals())


def test_policy_arithmetic_follows_the_frozen_cadence_gap_limit_and_slot_count() -> (
    None
):
    values = _policy_source()
    stated = _parameters(_text(_packet()), TIMING_PREFIX)
    cadence, gap = values["cadence_seconds"], values["frozen_gap_limit_seconds"]
    consecutive = values["max_consecutive_skipped_slots"]

    assert gap == 180 and cadence == 60 and SCREEN_SLOTS == 129_604
    assert (consecutive + 1) * cadence < gap <= (consecutive + 2) * cadence
    spans = 2 * stated["capture_span_seconds"]
    assert values["retry_start_window_seconds"] + spans <= cadence
    assert values["retry_start_window_seconds"] >= values["retry_delay_seconds"]
    assert (
        values["retry_delay_seconds"] + stated["capture_span_seconds"]
        > stated["decision_window_seconds"]
    )
    assert consecutive <= values["max_skipped_slots_per_hour"] < 3600 // cadence
    assert values["max_skipped_slots_total"] == math.floor(SCREEN_SLOTS / 1000)
    assert policy.SKIP_WINDOW_SLOTS == 3600 // cadence
    # A single skipped slot with the longest permitted retry never reaches the gap limit.
    worst_single = (
        2 * cadence
        + values["retry_start_window_seconds"]
        + stated["capture_span_seconds"]
    )
    assert worst_single < gap


def test_the_packet_and_the_spec_state_the_same_budget_triple() -> None:
    packet = _packet()
    values = _policy_source()
    expected = (
        values["max_consecutive_skipped_slots"],
        values["max_skipped_slots_per_hour"],
        values["max_skipped_slots_total"],
    )
    texts = {
        "decision_rule": packet["claims"]["decision_rule"],
        "limitations": packet["thesis"]["limitations"],
        "evidence": packet["mechanism"]["authorities"]["evidence"],
        "spec": _section(SPEC, "### Attempt 3 pre-registration"),
    }
    for name, text in texts.items():
        flat = " ".join(text.split())
        assert "older than one hour at decision time is a skipped slot" in flat, name
        assert "no decision is ever made on rules older than one hour" in flat, name
        match = re.search(
            r"\(?(\d+) consecutive, (\d+) per 60(?: slots)?, (\d+) total\)?", flat
        )
        assert match is not None, name
        assert tuple(map(int, match.groups())) == expected, name


def test_rules_refresh_classification_is_stated_identically_everywhere() -> None:
    packet = _packet()
    texts = {
        "limitations": packet["thesis"]["limitations"],
        "evidence": packet["mechanism"]["authorities"]["evidence"],
        "spec": _section(SPEC, "### Attempt 3 pre-registration"),
        "architecture": _arch_section(),
        "status": _status_paragraph(),
    }
    for name, text in texts.items():
        flat = " ".join(text.split())
        assert "Class D or Class I" in flat, name
        assert "refresh failure ends the attempt" in flat, name
        assert "consum" in flat and "budget" in flat, name
    for name in ("limitations", "evidence", "spec"):
        flat = " ".join(texts[name].split())
        assert "not retried" in flat, name
        assert "consumes no retry and no skip budget" in flat, name
    assert "conservative" in " ".join(texts["limitations"].split())
    assert "flagged for review" in texts["spec"] or "flagged for review" in " ".join(
        texts["spec"].split()
    )


def test_policy_text_forbids_the_unsafe_retries() -> None:
    packet = _packet()
    evidence = " ".join(packet["mechanism"]["authorities"]["evidence"].split())
    spec = " ".join(_section(SPEC, "### Attempt 3 pre-registration").split())
    for text in (evidence, spec):
        assert "Class D" in text and "Class I" in text and "Class T" in text
        assert "429" in text and "451" in text
        assert "never after a decision" in text
        assert "carry_paper_screen_v3" in text
    assert "A death inside a cycle is still never resumed" in evidence
    assert "a skipped slot cannot extend it" in evidence
    implemented = _spec_section()
    assert "never retried" in implemented
    assert "never replaced by another one" in implemented
    assert "so the skip budget does not rescue that case" in implemented


def test_run_identity_is_new_and_the_attempt2_seal_is_not_reused() -> None:
    limitations = _packet()["thesis"]["limitations"]
    assert "root-<UTC seal date YYYYMMDD>" in limitations
    assert "carry-a3-supervisor" in limitations and "carry-a3-check" in limitations
    assert "never reused" in limitations
    assert "must differ from every earlier digest" in limitations
    for path in re.findall(r"C:\\trade_rl_runs\\attempt3\\[^\s;,]*", limitations):
        assert "attempt2" not in path, path
    assert "not touched" in limitations


def test_seal_start_authorization_and_launch_fields_stay_pending() -> None:
    packet_text = _text(_packet())
    spec = _section(SPEC, "### Attempt 3 pre-registration")
    assert PENDING in packet_text
    for label in (
        "Authorization comment",
        "Screen start (UTC)",
        "Protocol digest (seal)",
        "Launch commit",
    ):
        assert re.search(re.escape(label) + r" \| `" + PENDING + r"`", spec), label
    assert "no seal was computed" in packet_text
    assert "nothing is sealed, launched, probed or collected" in spec.replace("\n", " ")
    assert "Nothing is sealed, launched, probed or collected" in _spec_section()
    seal_candidate = "3380e24801e5f5d3124e1e0bdee09f00a4bfcc03dee5fedd86af3b0c181b9781"
    allowed = {PINNED_DIGEST, PREREG_DIGEST, seal_candidate}
    attempt2 = "617988282ad6dfbf0ee7188f002676a1675dfac38c9ac969071bee3091b2e770"
    attempt1 = "843467870d85d0b085e65cf904e9d458287c14fc1e31615b9aaf77029f0669b7"
    for name, text in (("packet", packet_text), ("spec", spec)):
        found = set(re.findall(r"\b[0-9a-f]{64}\b", text))
        assert found - allowed - {attempt2, attempt1} == set(), (name, found)


def test_packet_and_new_text_are_result_blind() -> None:
    packet = _packet()
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
        re.compile(r"0\.25\s*[-\u2013]\s*0\.65"),
        re.compile(r"observed consume latency", re.IGNORECASE),
        re.compile(r"\b60 completed cycles\b", re.IGNORECASE),
    )
    sections = {
        "packet": _text(packet),
        "spec": _spec_section(),
        "architecture": _arch_section(),
        "status": _status_paragraph(),
    }
    for pattern in forbidden:
        for name, section in sections.items():
            assert pattern.search(section) is None, (name, pattern.pattern)
    for figure in ("5.62", "4.48", "0.6 s", "5.07"):
        assert figure not in _text(packet), figure
    limitations = " ".join(packet["thesis"]["limitations"].split())
    assert "no attempt-1 or attempt-2 price, equity" in limitations.lower()
    assert "attempt-2 timing metadata" in limitations
    assert "deliberately not used" in limitations
    assert "reused-development evidence only" in packet["thesis"]["hypothesis"]


def test_documents_point_to_the_packet_digest_and_unreviewed_status() -> None:
    for relative in DOCUMENTS:
        document = _raw(relative)
        assert RELATIVE_PACKET in document, relative
        assert PINNED_DIGEST in document, relative
        assert "UNREVIEWED" in document, relative
    assert RELATIVE_PACKET in _raw(BOUNDARIES)
    plan = _raw(PLAN)
    assert RELATIVE_PACKET in plan and "Status: Active" in plan[:200]
    head = _packet()["identity"]["implementation_head"]
    assert head in _raw(SPEC)
    assert head[:12] in _raw(ARCHITECTURE)
    assert "no PASS claimed" in _arch_section()
    assert "different-principal" in _arch_section()
    assert "does not extend to attempt 3" in _arch_section()


def test_the_ratified_readings_cite_the_planner_comment_everywhere() -> None:
    comment = "6717103371"
    packet = _packet()
    limitations = " ".join(packet["thesis"]["limitations"].split())
    assert comment in limitations and "5816161116" in limitations
    for relative in (SPEC, ARCHITECTURE):
        assert comment in _raw(relative), relative
    assert "5816161116" in _spec_section() and comment in _spec_section()
    assert (
        "before the first cycle" in _arch_section() or "first cycle" in _arch_section()
    )
    # The ratification relaxes nothing and authorizes nothing.
    assert "neither reading relaxes any criterion" in limitations
    assert "authorizes no seal or launch" in limitations
    assert "Class I, terminal and fail-closed" in " ".join(_arch_section().split())


def test_the_documents_name_what_is_still_to_be_built() -> None:
    implemented = _spec_section()
    for missing in ("carry_paper_screen_v3", "evaluator audit", "attempt-lineage"):
        assert missing in implemented, missing
    assert "cannot support a seal" in implemented
    assert "assessor" in implemented and "still rejects" in implemented
    assert "evaluator audit" in _status_paragraph()


def test_heads_bind_the_packet_to_the_implemented_collector() -> None:
    head = _packet()["identity"]["implementation_head"]
    if _git("cat-file", "-e", f"{head}^{{commit}}").returncode != 0:
        pytest.skip("implementation_head is not available in this checkout")
    policy_module = _git(
        "show", f"{head}:trade_rl/evaluation/paper/collection_policy.py"
    )
    assert policy_module.returncode == 0, policy_module.stderr
    assert "RETRIES_PER_SLOT = 2" in policy_module.stdout
    collector = _git("show", f"{head}:trade_rl/evaluation/paper/supervisor.py")
    assert collector.returncode == 0, collector.stderr
    for required in (
        "classify_failure",
        "first_budget_breach",
        "_register_missed_slots",
    ):
        assert required in collector.stdout, required
    control = _git("show", f"{head}:trade_rl/evaluation/paper/control.py")
    assert "CREATE TABLE IF NOT EXISTS skips" in control.stdout

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
    assert "trade_rl/evaluation/paper/collection_policy.py" in paths
    assert "trade_rl/integrations/binance/forward.py" in paths
    changed = _git("diff", "--name-only", head, packet_commit, "--", *paths)
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.strip() == "", (
        "packet-bound code changed after implementation_head; a new packet and digest "
        "are required: " + changed.stdout
    )
