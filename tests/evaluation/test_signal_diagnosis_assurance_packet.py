"""SD0 (#1146): the rendered G0-G2 author packet is valid, bound and UNREVIEWED."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

from tests.evaluation.signal_diagnosis_assurance_packet import (
    CITED_TESTS,
    REPORT_PATH,
    SD1_ORACLES,
    build_author_packet,
    canonical_packet_json,
)
from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.signal_diagnosis.protocol import (
    DATASET_ID,
    default_signal_diagnosis_protocol,
    protocol_digest,
)

ROOT = Path(__file__).resolve().parents[2]
HEAD = "a" * 40
OTHER_HEAD = "b" * 40
# The fourteen G0 Research Question Contract fields of docs/architecture/
# research-assurance.md, each mapped to the packet field and the marker that shows it.
G0_MARKERS = {
    "objective": ("final_decision", "Objective:"),
    "operational_target": ("final_decision", "Operational target:"),
    "economic_hypothesis": ("hypothesis", "H1:"),
    "causal_story": ("economic_mechanism", "Causal story:"),
    "assumptions": ("limitations", "Assumptions:"),
    "known_limitations": ("limitations", "Known-used and post hoc"),
    "falsifiers": ("cheapest_falsifier", "SD1 synthetic oracles"),
    "counterfactuals": ("counter_hypothesis", "Counterfactuals:"),
    "primary_metric": ("metric_proxy_rationale", "Primary metric"),
    "guardrail_metrics": ("metric_proxy_rationale", "Guardrails:"),
    "development_use": ("limitations", "Development use:"),
    "unused_data_policy": ("limitations", "Unused-data policy:"),
}


def _text(packet: object) -> str:
    return json.dumps(packet, ensure_ascii=False)


def _cli(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", "assurance", *args, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_packet_is_valid_unreviewed_and_grants_no_authorization() -> None:
    packet = build_author_packet(HEAD)

    result = evaluate_assurance(packet)

    assert result.status == "UNREVIEWED"
    assert result.errors == ()
    assert result.economic_execution_authorized is False
    assert result.record_digest == assurance_digest(packet)
    claims = packet["claims"]
    assert isinstance(claims, dict)
    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["claim_level"] == "software_validity"
    assert claims["next_action_class"] == "synthetic_verification_only"
    thesis = packet["thesis"]
    assert isinstance(thesis, dict)
    assert thesis["stage"] == "discovery"
    evidence = packet["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["evidence_level"] == "software_validity"
    # No economic evidence is asserted before any result exists.
    assert [k for k, v in evidence.items() if v is True] == ["no_development_rescue"]


def test_packet_is_deterministic_and_binds_the_exact_head() -> None:
    first = build_author_packet(HEAD)
    second = build_author_packet(HEAD)

    assert first == second
    assert assurance_digest(first) == assurance_digest(second)
    assert first["identity"] == {"protocol_head": HEAD, "implementation_head": HEAD}
    other = build_author_packet(OTHER_HEAD)
    assert assurance_digest(other) != assurance_digest(first)
    assert canonical_packet_json(HEAD) == json.dumps(
        first, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    malformed = copy.deepcopy(first)
    malformed["identity"] = {"protocol_head": "xyz", "implementation_head": "xyz"}
    assert evaluate_assurance(malformed).status == "BLOCKED"


def test_packet_is_bound_to_the_protocol_digest_and_dataset_identity() -> None:
    packet = build_author_packet(HEAD)
    digest = protocol_digest(default_signal_diagnosis_protocol())
    thesis = packet["thesis"]
    claims = packet["claims"]
    assert isinstance(thesis, dict) and isinstance(claims, dict)

    assert digest in thesis["final_decision"]
    assert digest in thesis["hypothesis"]
    assert any(digest in item for item in claims["permitted"])
    assert DATASET_ID in thesis["hypothesis"]
    assert REPORT_PATH in thesis["hypothesis"]
    assert (ROOT / REPORT_PATH).is_file()
    assert "signal_diagnosis_protocol_v3" in thesis["hypothesis"]


def test_packet_states_the_required_honest_limits() -> None:
    packet = build_author_packet(HEAD)
    thesis = packet["thesis"]
    claims = packet["claims"]
    assert isinstance(thesis, dict) and isinstance(claims, dict)
    text = _text(packet)

    assert "known-used" in thesis["hypothesis"]
    assert "post hoc" in thesis["hypothesis"]
    assert "no result value exists for this packet" in thesis["hypothesis"]
    limitations = thesis["limitations"]
    assert "Known-used and post hoc" in limitations
    assert "separate owner G4 lease" in limitations
    assert "No result values exist for this packet" in limitations
    assert "reads no market value" in limitations
    assert "UNREVIEWED" in limitations
    assert "separate owner G4 lease" in thesis["stop_rule"]
    assert "separate owner G4 lease" in claims["next_authorized_action"]
    assert "different authenticated GitHub principal" in thesis["stop_rule"]
    for oracle in SD1_ORACLES:
        assert oracle in thesis["cheapest_falsifier"], oracle
    assert "2026-08-01T00:00Z" in thesis["stop_rule"]
    assert any("edge" in item and "G4" in item for item in claims["forbidden"])
    assert "PASS" not in claims["next_authorized_action"]
    for word in ("owner-approved", "independently reviewed", "ASSURANCE_PASS"):
        assert word not in text


def test_packet_records_the_four_hypotheses_and_the_decision_table() -> None:
    packet = build_author_packet(HEAD)
    thesis = packet["thesis"]
    claims = packet["claims"]
    assert isinstance(thesis, dict) and isinstance(claims, dict)

    assert "H1:" in thesis["hypothesis"]
    for label in ("H0a", "H0b", "H0c"):
        assert label in thesis["counter_hypothesis"]
    rule = claims["decision_rule"]
    for label in ("INVALID", "NO_SIGNAL", "SIGNAL_NOT_TRADABLE", "SIGNAL_CANDIDATE"):
        assert label in rule and label in thesis["information_gain"]
    assert "drift baseline" in thesis["hypothesis"]
    assert "never an edge" in rule


def test_every_changed_boundary_has_an_oracle_and_cited_paths_exist() -> None:
    packet = build_author_packet(HEAD)
    mechanism = packet["mechanism"]
    assert isinstance(mechanism, dict)

    assert set(mechanism["changed_boundaries"]) == {
        "time",
        "sign",
        "execution",
        "evidence",
    }
    assert set(mechanism["independent_oracles"]) == set(mechanism["changed_boundaries"])
    for oracle in mechanism["independent_oracles"].values():
        assert set(oracle) == {
            "statement",
            "counterexample",
            "oracle",
            "known_limitations",
        }
    cited = set(re.findall(r"tests/[\w/]+\.py", _text(packet)))
    assert set(CITED_TESTS) <= cited
    assert [path for path in sorted(cited) if not (ROOT / path).is_file()] == []


# Aggregate magnitudes recorded in Issue #1146 section 1 before this protocol existed.
# They are disclosed, as cited and not recomputed, so the exposure can be audited.
CITED_PRIOR_VALUES = ("0.0675", "0.0164", "2.31", "-0.017588")
OTHER_RECORDED_RESULTS = ("-1.95", "45.6", "16.03", "15.80", "5.55")


def test_packet_text_contains_no_new_result_values() -> None:
    packet = build_author_packet(HEAD)
    thesis = packet["thesis"]
    assert isinstance(thesis, dict)
    text_without_limitations = _text(
        {k: v for k, v in packet.items() if k != "thesis"}
        | {"thesis": {k: v for k, v in thesis.items() if k != "limitations"}}
    )

    for recorded_value in (*CITED_PRIOR_VALUES, *OTHER_RECORDED_RESULTS):
        assert recorded_value not in text_without_limitations, recorded_value
    for recorded_value in OTHER_RECORDED_RESULTS:
        assert recorded_value not in thesis["limitations"], recorded_value
    assert re.search(r"pooled_ic|spearman_ic", _text(packet)) is None


def test_limitations_disclose_the_prior_ic_exposure_and_the_power_trade_off() -> None:
    packet = build_author_packet(HEAD)
    thesis = packet["thesis"]
    assert isinstance(thesis, dict)
    limitations = thesis["limitations"]

    for value in CITED_PRIOR_VALUES:
        assert value in limitations, value
    for needle in (
        "cited from Issue #1146 section 1",
        "not recomputed",
        "pooled IC",
        "rank IC",
        "mse_skill",
        "daily t",
        "4 of 4",
        "0.41",
        "0.74",
        "deliberate",
    ):
        assert needle in limitations, needle


def test_packet_freezes_the_significance_procedure_and_the_bh_family() -> None:
    packet = build_author_packet(HEAD)
    thesis = packet["thesis"]
    mechanism = packet["mechanism"]
    claims = packet["claims"]
    assert isinstance(thesis, dict) and isinstance(mechanism, dict)
    assert isinstance(claims, dict)

    for needle in (
        "one-sided",
        "moving-block bootstrap",
        "20-day blocks",
        "2000 resamples",
        "16 decision cells",
    ):
        assert needle in thesis["metric_proxy_rationale"], needle
    evidence = mechanism["independent_oracles"]["evidence"]
    assert (
        "Benjamini-Hochberg adjustment is computed inside the verdict"
        in (evidence["statement"])
    )
    assert "provenance" in evidence["counterexample"]
    assert "one-sided" in claims["decision_rule"]


def test_any_edit_changes_the_digest_and_a_self_authored_review_stays_blocked() -> None:
    packet = build_author_packet(HEAD)
    digest = assurance_digest(packet)
    edited = copy.deepcopy(packet)
    edited["thesis"]["stop_rule"] += " "  # type: ignore[index]
    assert assurance_digest(edited) != digest

    review = {
        "schema_version": "research_assurance_review_v1",
        "reviewed_record_digest": digest,
        "protocol_head": HEAD,
        "implementation_head": HEAD,
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


def test_cli_check_inspect_and_digest_agree_on_the_rendered_packet(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sd0-author-packet.json"
    target.write_text(canonical_packet_json(HEAD), encoding="utf-8")
    expected = assurance_digest(build_author_packet(HEAD))

    check = _cli(target, "check")
    inspect = _cli(target, "inspect")
    digest = _cli(target, "digest")

    assert check.returncode == 1, check.stderr
    assert json.loads(check.stdout)["status"] == "UNREVIEWED"
    assert inspect.returncode == 0, inspect.stderr
    assert json.loads(inspect.stdout)["record_digest"] == expected
    assert digest.returncode == 0, digest.stderr
    assert json.loads(digest.stdout) == {"record_digest": expected}


def test_packet_is_rendered_not_committed() -> None:
    """A packet inside the PR tree cannot name its own commit; it is posted instead."""

    assert not list((ROOT / "report").glob("*issue1146*assurance*"))


def test_the_fourteen_g0_contract_fields_are_covered() -> None:
    packet = build_author_packet(HEAD)
    thesis = packet["thesis"]
    claims = packet["claims"]
    assert isinstance(thesis, dict) and isinstance(claims, dict)

    for name, (field, marker) in G0_MARKERS.items():
        assert marker in thesis[field], name
    # what_this_can_prove / what_this_cannot_prove are the permitted/forbidden claims.
    assert claims["permitted"] and claims["forbidden"]
    assert len(G0_MARKERS) + 2 == 14


def test_packet_uses_the_v2_semantics_and_no_stale_v1_wording() -> None:
    packet = build_author_packet(HEAD)
    text = _text(packet)
    thesis = packet["thesis"]
    assert isinstance(thesis, dict)

    for stale in (
        "signal_diagnosis_protocol_v1",
        "signal_diagnosis_protocol_v2",
        "beta-removed",
        "beta removal",
        "survives beta",
        "five full calendar years",
        "four of the five",
        "2021-2025",
        "after-cost alpha",
    ):
        assert stale not in text, stale
    assert "2022-2025" in thesis["hypothesis"]
    assert "partial and reported only" in thesis["hypothesis"]
    assert "not heterogeneous market beta" in thesis["final_decision"]
    assert "not a beta-adjusted alpha" in thesis["hypothesis"]
    assert "four of four" in thesis["limitations"]
    assert "12 covered months" in thesis["hypothesis"]
    claims = packet["claims"]
    assert isinstance(claims, dict)
    assert any("beta-adjusted" in item for item in claims["forbidden"])
