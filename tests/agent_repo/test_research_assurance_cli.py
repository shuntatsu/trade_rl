from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.agent_repo.research_assurance import assurance_digest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tools/agent_repo/examples/research_assurance.json"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", "assurance", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_inspect_reports_unreviewed_without_false_authorization() -> None:
    proc = run_cli("inspect", str(EXAMPLE))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "UNREVIEWED"
    assert payload["economic_execution_authorized"] is False


def test_check_exits_nonzero_on_unreviewed_assurance() -> None:
    proc = run_cli("check", str(EXAMPLE))
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["status"] == "UNREVIEWED"


def test_digest_matches_canonical_author_packet() -> None:
    proc = run_cli("digest", str(EXAMPLE))
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["record_digest"] == assurance_digest(
        json.loads(EXAMPLE.read_text(encoding="utf-8"))
    )


def test_author_supplied_review_alone_never_passes(tmp_path: Path) -> None:
    record = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    review = {
        "schema_version": "research_assurance_review_v1",
        "reviewed_record_digest": assurance_digest(record),
        "protocol_head": record["identity"]["protocol_head"],
        "implementation_head": record["identity"]["implementation_head"],
        "reviewer_identity": "independent",
        "reviewer_execution_identity": "some-run",
        "reviewer_independence": "ESTABLISHED",
        "result_blind": True,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "PASS",
        "status": "pass",
        "disposition": "ASSURANCE_PASS",
        "blocking_findings": [],
        "missing_evidence": [],
        "rationale": "Challenge the causal question and possible alternative.",
        "limitations": ["No economic result."],
    }
    target = tmp_path / "review.json"
    target.write_text(json.dumps(review), encoding="utf-8")
    proc = run_cli("inspect", str(EXAMPLE), "--review", str(target))
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["status"] == "BLOCKED"
    assert "authenticated external review authority" in proc.stdout


def test_duplicate_fields_are_refused_not_last_write_wins(tmp_path: Path) -> None:
    record = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload = json.dumps(record).replace(
        '"schema_version": "research_assurance_v1"',
        '"schema_version": "wrong", "schema_version": "research_assurance_v1"',
    )
    target = tmp_path / "duplicate.json"
    target.write_text(payload, encoding="utf-8")
    proc = run_cli("inspect", str(target))
    assert proc.returncode == 2
    assert "duplicate" in proc.stderr


def test_nonfinite_json_constants_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "invalid.json"
    target.write_text('{"schema_version": NaN}', encoding="utf-8")
    assert run_cli("inspect", str(target)).returncode == 2
