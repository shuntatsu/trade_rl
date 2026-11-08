"""The attempt-3 seal-candidate packet stays UNREVIEWED and bound to the source.

The pre-registration and implementation packets are frozen. This packet claims that the
three remaining parts (``carry_paper_screen_v3``, the evaluator audit and the
attempt-lineage v4 naming attempt 2) are built. These tests import the policy constants,
the screen version, the lineage version and the schema names from the source, so the
packet, the specification and the code cannot diverge, and they keep the packet from
claiming a review, a seal or any economic result.
"""

from __future__ import annotations

import inspect
import json
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
from tests.evaluation.test_paper_attempt3_implementation_packet import (
    _policy_mismatches,
    _policy_source,
)
from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.paper import assessment, collection_audit, operations
from trade_rl.evaluation.paper import collection_policy as policy
from trade_rl.evaluation.paper import predecessor as pred

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_PACKET = (
    "report/issue810-carry-attempt3-seal-candidate-research-assurance-20261108.json"
)
PACKET = ROOT / RELATIVE_PACKET
PINNED_DIGEST = "4481ec5e42324f605978c8456f77dcd7a7f1e66c0b61b6378ccd9cc819fd8829"
PREREG_PACKET = "report/issue810-carry-attempt3-research-assurance-20261108.json"
PREREG_DIGEST = "29467597161a8856c97e2b9c46094e185bef84eed304ac7f33f75edf1ddbea10"
IMPL_PACKET = (
    "report/issue810-carry-attempt3-implementation-research-assurance-20261108.json"
)
IMPL_DIGEST = "eeb2fcaa84c161bee4490a5ab0617a63ddcc38c4eb383f420379c7d175350d45"
IMPL_HEAD = "0dacfb147bc975b3d0049d671615c282e8c1181c"
ATTEMPT2_PACKET_DIGEST = (
    "617988282ad6dfbf0ee7188f002676a1675dfac38c9ac969071bee3091b2e770"
)
SPEC = "docs/specs/funding-carry-paper.md"
ARCHITECTURE = "docs/architecture/research-assurance.md"
STATUS = "docs/research/current-status.md"
BOUNDARIES = "docs/architecture/package-boundaries.md"
PLAN = "docs/plans/funding-carry-paper.md"
DOCUMENTS = (SPEC, ARCHITECTURE, STATUS)
COMMENT_ATTEMPT2_CAUSE = "6711911740"


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
    return _section(SPEC, "#### Attempt 3 seal candidate")


def _arch_section() -> str:
    return _section(
        ARCHITECTURE,
        "#### Attempt 3 seal-candidate packet",
        "**Known limitations:**",
    )


def _status_paragraph() -> str:
    return _section(STATUS, "2026-11-08: the attempt-3 seal candidate is built", "\n\n")


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


def test_packet_is_unreviewed_valid_and_pins_its_digest() -> None:
    packet = _packet()

    result = evaluate_assurance(packet)

    assert result.status == "UNREVIEWED"
    assert result.errors == ()
    assert result.economic_execution_authorized is False
    assert result.record_digest == assurance_digest(packet) == PINNED_DIGEST
    assert PINNED_DIGEST not in {PREREG_DIGEST, IMPL_DIGEST, ATTEMPT2_PACKET_DIGEST}
    assert b"\r" not in PACKET.read_bytes()
    identity = packet["identity"]
    assert identity["protocol_head"] == identity["implementation_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", identity["implementation_head"])
    assert identity["implementation_head"] != IMPL_HEAD


def test_cli_check_inspect_and_digest_agree_and_report_unreviewed() -> None:
    check, inspect_, digest = _cli("check"), _cli("inspect"), _cli("digest")

    assert check.returncode == 1, check.stderr
    assert json.loads(check.stdout)["status"] == "UNREVIEWED"
    assert inspect_.returncode == 0, inspect_.stderr
    assert json.loads(inspect_.stdout)["record_digest"] == PINNED_DIGEST
    assert json.loads(inspect_.stdout)["economic_execution_authorized"] is False
    assert digest.returncode == 0, digest.stderr
    assert json.loads(digest.stdout) == {"record_digest": PINNED_DIGEST}


def test_the_earlier_packets_are_untouched_and_not_re_pinned() -> None:
    for relative, digest in ((PREREG_PACKET, PREREG_DIGEST), (IMPL_PACKET, IMPL_DIGEST)):
        frozen = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        assert assurance_digest(frozen) == digest, relative
    impl = json.loads((ROOT / IMPL_PACKET).read_text(encoding="utf-8"))
    assert impl["identity"]["implementation_head"] == IMPL_HEAD


def test_packet_grants_no_authorization_and_claims_no_review_or_seal() -> None:
    packet = _packet()
    claims, thesis = packet["claims"], packet["thesis"]
    limitations = " ".join(thesis["limitations"].split())
    stop_rule = " ".join(thesis["stop_rule"].split())
    decision = " ".join(claims["decision_rule"].split())

    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["claim_level"] == "software_validity"
    assert claims["next_action_class"] == "documentation_only"
    assert packet["evidence"]["evidence_level"] == "software_validity"
    assert thesis["stage"] == "prospective_paper"
    assert "No PASS is claimed" in limitations
    assert "not different-principal independence" in limitations
    assert "authorizes no seal, launch, probe or economic run" in limitations
    assert G0_EXCEPTION_URL in stop_rule
    assert "does not extend to attempt 3" in stop_rule
    assert "Do not seal, launch" in stop_rule
    assert "Do not seal attempt 3 under this packet alone" in decision
    forbidden = " ".join(claims["forbidden"])
    for required in (
        "sealing, launching or collecting attempt 3",
        "treating the seal candidate as reviewed, as a seal authorization",
        "reusing the attempt-2 seal",
        "writing to the attempt-2 root, its failure marker or the carry-a2",
        "attempt-2 timing metadata",
        "predecessor-root-unavailable G0 exception",
        "funding-proximity distribution",
        "re-classifying a sealed result",
    ):
        assert required in forbidden, required
    assert "none of it is reviewed" in " ".join(claims["permitted"])
    assert packet["mechanism"]["changed_boundaries"] == ["evidence"]
    for boundary in ("unit", "time", "state", "sign", "risk", "execution", "accounting"):
        assert "Unchanged" in packet["mechanism"]["authorities"][boundary], boundary
    cited = set(re.findall(r"tests/[\w/]+\.py", _text(packet)))
    assert cited
    assert sorted(path for path in cited if not (ROOT / path).is_file()) == []
    modules = set(re.findall(r"trade_rl/[\w/]+\.py", _text(packet)))
    assert modules
    assert sorted(path for path in modules if not (ROOT / path).is_file()) == []


def test_packet_and_spec_policy_and_timing_values_equal_the_source_constants() -> None:
    packet = _packet()
    text = _text(packet)
    stated = _parameters(text, POLICY_PREFIX)
    assert _policy_mismatches(stated) == []
    assert _mismatches(_parameters(text, TIMING_PREFIX)) == []
    assert set(_parameters(text, TIMING_PREFIX)) == set(_source_literals())
    spec = " ".join(_raw(SPEC).split())
    from_spec = _parameters(
        spec[spec.index("### Attempt 3 pre-registration") :], POLICY_PREFIX
    )
    assert from_spec == stated == _policy_source()


@pytest.mark.parametrize("name", sorted(_policy_source()))
def test_a_diverging_policy_value_is_detected(name: str) -> None:
    stated = _parameters(_text(_packet()), POLICY_PREFIX)
    assert _policy_mismatches({**stated, name: stated[name] + 1}) == [name]
    assert _policy_mismatches({k: v for k, v in stated.items() if k != name}) == [name]


def test_the_budget_triple_and_the_failed_capture_bound_are_stated_consistently() -> (
    None
):
    values = _policy_source()
    expected = (
        values["max_consecutive_skipped_slots"],
        values["max_skipped_slots_per_hour"],
        values["max_skipped_slots_total"],
    )
    packet = _packet()
    texts = {
        "decision_rule": packet["claims"]["decision_rule"],
        "limitations": packet["thesis"]["limitations"],
        "spec": _spec_section(),
    }
    for name, text in texts.items():
        flat = " ".join(text.split())
        match = re.search(
            r"\(?(\d+) consecutive, (\d+) per 60(?: slots)?, (\d+) total\)?", flat
        )
        assert match is not None, name
        assert tuple(map(int, match.groups())) == expected, name
    spec = " ".join(_spec_section().split())
    bound = re.search(r"`retries_per_slot \+ 2` \((\d+)\)", spec)
    assert bound is not None
    assert int(bound.group(1)) == policy.MAX_FAILED_CAPTURES_PER_SLOT
    assert policy.MAX_FAILED_CAPTURES_PER_SLOT == policy.RETRIES_PER_SLOT + 2


def test_the_screen_and_lineage_versions_and_schemas_equal_the_source() -> None:
    plan = operations.screen_plan_v3(
        attempt_lineage=pred.canonical_lineage_v4(
            protocol_sha256="a" * 64,
            final_tip_sha256="b" * 64,
            last_observed_at="2026-11-07T00:59:00+00:00",
            event_count=1,
            root_path="C:\\x",
        )
    )
    versions = {
        "screen": plan["schema"],
        "lineage": plan["attempt_lineage"]["schema"],
        "report": collection_audit.REPORT_SCHEMA,
        "assessment": re.search(
            r'"(carry_paper_assessment_v3)"', inspect.getsource(assessment)
        ),
    }
    assert versions["screen"] == pred.SCREEN_V3_SCHEMA == "carry_paper_screen_v3"
    assert versions["lineage"] == pred.LINEAGE_V4_SCHEMA
    assert versions["lineage"] == "carry_paper_attempt_lineage_v4"
    assert versions["report"] == "carry_paper_collection_report_v1"
    assert versions["assessment"] is not None
    assessment_schema = versions["assessment"].group(1)
    packet_text = " ".join(_text(_packet()).split())
    spec = " ".join(_raw(SPEC).split())
    for name in (
        versions["screen"],
        versions["lineage"],
        versions["report"],
        assessment_schema,
    ):
        assert name in spec, name
    for name in (versions["screen"], versions["lineage"]):
        assert name in packet_text, name
    # The screen v2 label is not what a v3 plan carries, and v2 is still produced.
    assert operations.screen_plan(
        attempt_lineage=pred.canonical_lineage("unavailable")
    )["schema"] == "carry_paper_screen_v2"
    assert plan["operational_quality"]["max_skipped_slots_total"] == (
        policy.MAX_SKIPPED_SLOTS_TOTAL
    )


def test_the_documented_economic_fields_are_the_screen_v2_fields() -> None:
    """The economic keys of both plans are exactly the keys the spec's v2 section names."""
    v2 = operations.screen_plan(attempt_lineage=pred.canonical_lineage("unavailable"))
    lineage = pred.canonical_lineage_v4(
        protocol_sha256="a" * 64,
        final_tip_sha256="b" * 64,
        last_observed_at="2026-11-07T00:59:00+00:00",
        event_count=1,
        root_path="C:\\x",
    )
    v3 = operations.screen_plan_v3(attempt_lineage=lineage)
    assert set(v3) - set(v2) == {"operational_quality"}
    assert set(v2) - set(v3) == set()
    differing = {key for key in v2 if v2[key] != v3[key]}
    assert differing == {"schema", "attempt_lineage"}


def test_run_identity_is_new_and_the_attempt2_seal_is_not_reused() -> None:
    limitations = " ".join(_packet()["thesis"]["limitations"].split())
    assert "root-<UTC seal date YYYYMMDD>" in limitations
    assert "carry-a3-supervisor" in limitations and "carry-a3-check" in limitations
    assert "never reused" in limitations
    assert "must differ from every earlier digest" in limitations
    for path in re.findall(r"C:\\trade_rl_runs\\attempt3\\[^\s;,]*", limitations):
        assert "attempt2" not in path, path
    spec = _spec_section()
    for path in re.findall(r"C:\\trade_rl_(?:runs|frozen)\\attempt3[^\s`;,]*", spec):
        assert "attempt2" not in path, path
    assert "not touched" in limitations or "are not touched" in limitations


def test_the_spec_gives_the_exact_commands_and_names_what_is_missing() -> None:
    spec = _spec_section()
    flat = " ".join(spec.split())
    assert "nothing below has been run" in flat
    for required in (
        "python -m trade_rl.evaluation.paper.storage_probe",
        "python -m trade_rl.evaluation.paper.attempt3 plan",
        "python -m trade_rl.evaluation.paper.attempt3 launch",
        "--predecessor-root C:\\trade_rl_runs\\attempt2\\root-20261107",
        "--predecessor-protocol-sha256",
        "carry-a3-supervisor",
        "carry-a3-check",
        "git worktree add --detach",
        "cli evaluate",
        "Still missing before a seal",
    ):
        assert required in spec or required in flat, required
    assert "`seal`" in flat
    # The operator CLI really has the commands and options the procedure names.
    parser = json.dumps(
        [action.dest for action in _subparser("plan")]
        + [action.dest for action in _subparser("seal")]
        + [action.dest for action in _subparser("launch")]
    )
    for option in (
        "predecessor_root",
        "predecessor_protocol_sha256",
        "bytes_per_observation",
        "authorization",
        "allow_network",
        "protocol_sha256",
        "log_file",
        "start_at",
    ):
        assert option in parser, option
    for task in ("carry-a3-supervisor", "carry-a3-check"):
        assert task in spec


def _subparser(name: str) -> list[Any]:
    from trade_rl.evaluation.paper import attempt3

    parser = attempt3.build_parser()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and name in choices:
            return list(choices[name]._actions)
    raise AssertionError(name)


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
    flat = _spec_section()
    assert "nothing is sealed, launched, probed or collected" in flat
    known = {
        PINNED_DIGEST,
        PREREG_DIGEST,
        IMPL_DIGEST,
        ATTEMPT2_PACKET_DIGEST,
        pred.ATTEMPT_ONE_PROTOCOL_SHA256,
    }
    for name, text in (("packet", packet_text), ("spec", spec)):
        found = set(re.findall(r"\b[0-9a-f]{64}\b", text))
        assert found - known - {pred.ATTEMPT_ONE_FINAL_TIP_SHA256} == set(), (
            name,
            found,
        )


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
    limitations = " ".join(packet["thesis"]["limitations"].split())
    assert "no attempt-2 timing metadata" in limitations.lower()
    assert "reused-development evidence only" in packet["thesis"]["hypothesis"]
    # No attempt-2 observed figure appears: only the incident time recorded in the
    # public failure record (cycle number and halt time) is mentioned.
    assert "2026-11-07T00:59:07Z" in limitations


def test_lineage_v4_text_is_a_software_defect_record_without_an_outcome() -> None:
    record = pred.canonical_lineage_v4(
        protocol_sha256="a" * 64,
        final_tip_sha256="b" * 64,
        last_observed_at="2026-11-07T00:59:00+00:00",
        event_count=1,
        root_path="C:\\x",
    )
    cause = record["predecessor_invalidation"]
    assert COMMENT_ATTEMPT2_CAUSE in cause["cause_record"]
    assert "software defect" in cause["cause"].lower()
    assert record["predecessor"]["reason_codes"] == ["collector_failure"]
    flat = json.dumps(record).lower()
    for word in ("profit", "equity", "drawdown", "return", "pnl"):
        assert word not in flat, word
    packet = _packet()
    limitations = " ".join(packet["thesis"]["limitations"].split())
    assert COMMENT_ATTEMPT2_CAUSE in limitations
    assert "economic_outcome_used false" in limitations


def test_the_documents_point_to_the_packet_digest_and_unreviewed_status() -> None:
    for relative in DOCUMENTS:
        document = _raw(relative)
        assert RELATIVE_PACKET in document, relative
        assert PINNED_DIGEST in document, relative
        assert "UNREVIEWED" in document, relative
        assert "{{" not in document, relative
    assert RELATIVE_PACKET in _raw(BOUNDARIES)
    plan = _raw(PLAN)
    assert RELATIVE_PACKET in plan and "Status: Active" in plan[:200]
    head = _packet()["identity"]["implementation_head"]
    assert head in _raw(SPEC)
    assert head[:12] in _raw(ARCHITECTURE)
    arch = _arch_section()
    assert "no PASS claimed" in arch
    assert "different-principal" in arch
    assert "does not extend to attempt 3" in arch
    assert "UNREVIEWED" in _spec_section()


def test_review_route_and_g0_exception_are_stated_narrowly() -> None:
    packet = _packet()
    limitations = " ".join(packet["thesis"]["limitations"].split())
    assert DELEGATION_ID in limitations
    assert "Claude Haiku 5.5" in limitations
    assert "rejects same-principal reviews" in limitations
    assert "predecessor-root-unavailable G0 exception is not extended" in limitations
    spec = _spec_section()
    assert "does not extend to attempt 3" in spec
    assert "lineage v4 admits only" in " ".join(
        packet["thesis"]["stop_rule"].split()
    ) or "Only an available root is admitted" in spec


def test_the_packet_names_what_the_audit_does_not_verify() -> None:
    flat = " ".join(_packet()["thesis"]["limitations"].split())
    assert "does not verify the retry timing" in flat
    assert "caller assertions" in flat
    independent = _packet()["mechanism"]["independent_oracles"]["evidence"]
    assert "not audited" in independent["known_limitations"]


def test_heads_bind_the_packet_to_the_built_screen_audit_and_lineage() -> None:
    head = _packet()["identity"]["implementation_head"]
    if _git("cat-file", "-e", f"{head}^{{commit}}").returncode != 0:
        pytest.skip("implementation_head is not available in this checkout")
    for module, required in (
        ("collection_audit.py", ("audit_registered_collection", "skip_report")),
        ("attempt3.py", ("plan_attempt3", "seal_attempt3", "launch_attempt3")),
        ("operations.py", ("screen_plan_v3", "operational_quality_clause")),
        ("predecessor.py", ("canonical_lineage_v4", "_ATTEMPT3_SEAL_TOKEN")),
        ("assessment.py", ("registered_collection_evidence", "reused_seal_reasons")),
    ):
        shown = _git("show", f"{head}:trade_rl/evaluation/paper/{module}")
        assert shown.returncode == 0, shown.stderr
        for name in required:
            assert name in shown.stdout, (module, name)
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
    assert "trade_rl/evaluation/paper/collection_audit.py" in paths
    assert "trade_rl/evaluation/paper/attempt3.py" in paths
    assert "trade_rl/evaluation/paper/collection_policy.py" in paths
    changed = _git("diff", "--name-only", head, packet_commit, "--", *paths)
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.strip() == "", (
        "packet-bound code changed after implementation_head; a new packet and digest "
        "are required: " + changed.stdout
    )
