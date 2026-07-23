from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.workflows.selection_authorization import (
    SelectionProposal,
    load_selection_authorization,
    write_selection_proposal,
)

NOW = datetime(2026, 7, 18, 3, 0, tzinfo=UTC)


def _write_private_key(path: Path, *, purpose: str, raw: bytes) -> None:
    path.write_text(
        json.dumps(
            {
                "algorithm": "ed25519",
                "key_id": f"{purpose}-key",
                "private_key": base64.b64encode(raw).decode("ascii"),
                "purpose": purpose,
                "schema_version": "ed25519_private_key_v1",
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _reconciliation_request() -> dict[str, object]:
    return {
        "cash_tolerance_fraction": 1e-8,
        "created_at": (NOW + timedelta(days=30)).isoformat(),
        "dataset_id": "1" * 64,
        "duplicate_fill_count": 0,
        "end_time": (NOW + timedelta(days=30)).isoformat(),
        "environment_digest": "2" * 64,
        "equity_tolerance_fraction": 1e-8,
        "fill_log_digest": "3" * 64,
        "matched_fill_count": 80,
        "maximum_cash_difference_fraction": 0.0,
        "maximum_equity_difference_fraction": 0.0,
        "maximum_position_notional_difference_fraction": 0.0,
        "observed_fill_count": 80,
        "open_order_count": 0,
        "order_log_digest": "4" * 64,
        "policy_digest": "5" * 64,
        "position_notional_tolerance_fraction": 1e-8,
        "schema_version": "paper_reconciliation_request_v1",
        "start_time": NOW.isoformat(),
        "submitted_order_count": 100,
        "terminal_order_count": 100,
        "training_run_digest": "6" * 64,
        "unknown_order_fill_count": 0,
    }


def test_selection_authorize_cli_writes_immutable_signed_authorization(
    tmp_path: Path,
) -> None:
    from trade_rl.cli import extended

    proposal = SelectionProposal.create(
        walk_forward_run_digest="a" * 64,
        gate_evidence_digest="b" * 64,
        execution_sensitivity_digest="c" * 64,
        dataset_id="d" * 64,
        selected_configuration="selected",
        candidate_config_digest="e" * 64,
        seeds=(0, 1, 2),
        git_commit="f" * 40,
        dependency_digest="1" * 64,
        resume_checkpoint_digests=(),
    )
    proposal_path = write_selection_proposal(tmp_path / "proposal.json", proposal)
    key_path = tmp_path / "selection-key.json"
    _write_private_key(
        key_path,
        purpose="selection-authorization",
        raw=b"\x31" * 32,
    )
    output = tmp_path / "authorization.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "selection",
            "authorize",
            "--proposal",
            str(proposal_path),
            "--private-key",
            str(key_path),
            "--approver",
            "research-committee",
            "--approved-at",
            NOW.isoformat(),
            "--expires-at",
            (NOW + timedelta(days=7)).isoformat(),
            "--output",
            str(output),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    authorization = load_selection_authorization(output)
    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert authorization.proposal_digest == proposal.digest
    assert payload == {
        "artifact_path": str(output),
        "authorization_digest": authorization.authorization_digest,
        "key_id": "selection-authorization-key",
        "production_status": "NO-GO",
        "proposal_digest": proposal.digest,
        "schema": "selection_authorization_result_v1",
        "status": "authorized_for_selected_final_training",
    }


def test_release_approve_cli_writes_external_attestation(tmp_path: Path) -> None:
    from tests.serving.helpers import create_bundle
    from trade_rl.cli import extended
    from trade_rl.domain.selection import PolicyMode
    from trade_rl.release.attestation import (
        default_attestation_path,
        load_release_attestation,
    )

    bundle_root = create_bundle(
        tmp_path / "bundle",
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        release_digest=None,
    )
    key_path = tmp_path / "release-key.json"
    _write_private_key(
        key_path,
        purpose="release-verification",
        raw=b"\x32" * 32,
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "release",
            "approve",
            "--bundle",
            str(bundle_root),
            "--private-key",
            str(key_path),
            "--git-commit",
            "a" * 40,
            "--dependency-digest",
            "b" * 64,
            "--approver",
            "release-committee",
            "--approved-at",
            NOW.isoformat(),
            "--expires-at",
            (NOW + timedelta(days=30)).isoformat(),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    output = default_attestation_path(bundle_root)
    attestation = load_release_attestation(output)
    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert payload == {
        "artifact_path": str(output),
        "attestation_digest": attestation.attestation_digest,
        "bundle_digest": attestation.bundle_digest,
        "key_id": "release-verification-key",
        "production_status": "NO-GO",
        "schema": "release_approval_result_v1",
        "status": "approved_for_release_activation",
    }


def test_confirmation_create_cli_signs_exact_external_measurements(
    tmp_path: Path,
) -> None:
    from trade_rl.cli import extended
    from trade_rl.evaluation.confirmation import load_confirmation_evidence

    key_path = tmp_path / "confirmation-key.json"
    _write_private_key(
        key_path,
        purpose="fresh-confirmation",
        raw=b"\x33" * 32,
    )
    request_path = tmp_path / "confirmation-request.json"
    request_path.write_text(
        json.dumps(
            {
                "created_at": (NOW + timedelta(days=30)).isoformat(),
                "dataset_id": "1" * 64,
                "dependency_digest": "2" * 64,
                "end_time": (NOW + timedelta(days=30)).isoformat(),
                "environment_digest": "3" * 64,
                "fill_log_digest": "4" * 64,
                "git_commit": "5" * 40,
                "order_log_digest": "6" * 64,
                "policy_digest": "7" * 64,
                "reconciliation_digest": "8" * 64,
                "required_after": NOW.isoformat(),
                "return_period_hours": 24.0,
                "returns": [0.001] * 30,
                "schema_version": "fresh_confirmation_request_v1",
                "start_time": NOW.isoformat(),
                "training_run_digest": "9" * 64,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "confirmation.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "confirmation",
            "create",
            "--request",
            str(request_path),
            "--private-key",
            str(key_path),
            "--output",
            str(output),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    evidence = load_confirmation_evidence(output)
    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert evidence.days == 30.0
    assert payload == {
        "artifact_path": str(output),
        "evidence_digest": evidence.evidence_digest,
        "key_id": "fresh-confirmation-key",
        "production_status": "NO-GO",
        "schema": "confirmation_creation_result_v1",
        "status": "sealed_for_fresh_confirmation_review",
    }


def test_reconciliation_create_cli_writes_derived_evidence(tmp_path: Path) -> None:
    from trade_rl.cli import extended
    from trade_rl.evaluation.paper_reconciliation import (
        load_paper_reconciliation_evidence,
    )

    request_path = tmp_path / "reconciliation-request.json"
    request_path.write_text(json.dumps(_reconciliation_request()), encoding="utf-8")
    output = tmp_path / "paper-reconciliation.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "reconciliation",
            "create",
            "--request",
            str(request_path),
            "--output",
            str(output),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    evidence = load_paper_reconciliation_evidence(output)
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert evidence.passed is True
    assert json.loads(stdout.getvalue()) == {
        "artifact_path": str(output),
        "evidence_digest": evidence.evidence_digest,
        "passed": True,
        "production_status": "NO-GO",
        "schema": "paper_reconciliation_creation_result_v1",
        "status": "sealed_for_fresh_confirmation_review",
    }


def test_reconciliation_create_cli_preserves_failed_evidence(tmp_path: Path) -> None:
    from trade_rl.cli import extended
    from trade_rl.evaluation.paper_reconciliation import (
        load_paper_reconciliation_evidence,
    )

    request = _reconciliation_request()
    request["matched_fill_count"] = 79
    request["unknown_order_fill_count"] = 1
    request_path = tmp_path / "reconciliation-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "paper-reconciliation.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "reconciliation",
            "create",
            "--request",
            str(request_path),
            "--output",
            str(output),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    evidence = load_paper_reconciliation_evidence(output)
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert evidence.passed is False
    assert json.loads(stdout.getvalue())["passed"] is False


def test_reconciliation_create_cli_rejects_caller_passed_field(
    tmp_path: Path,
) -> None:
    from trade_rl.cli import extended

    request = _reconciliation_request()
    request["passed"] = True
    request_path = tmp_path / "reconciliation-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "paper-reconciliation.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "reconciliation",
            "create",
            "--request",
            str(request_path),
            "--output",
            str(output),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert not output.exists()
    assert json.loads(stderr.getvalue()) == {
        "error": "paper reconciliation request fields are invalid",
        "error_type": "ValueError",
        "production_status": "NO-GO",
        "schema": "paper_reconciliation_creation_error_v1",
        "status": "failed",
    }


def test_serving_package_cli_forwards_explicit_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.cli import extended

    captured: dict[str, object] = {}

    def fake_package(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            bundle_digest="1" * 64,
            confirmation_evidence_digest="2" * 64,
            run_kind="research_selected_final",
            training_run_digest="3" * 64,
        )

    monkeypatch.setattr(extended, "package_selected_training_run", fake_package)
    monkeypatch.setattr(extended, "load_public_verification_keys", lambda _: {})
    reconciliation = tmp_path / "paper-reconciliation.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "serving",
            "package",
            "--training-run",
            str(tmp_path / "training"),
            "--confirmation",
            str(tmp_path / "confirmation.json"),
            "--paper-reconciliation",
            str(reconciliation),
            "--confirmation-public-keys",
            str(tmp_path / "keys.json"),
            "--output",
            str(tmp_path / "bundle"),
            "--signal-digest",
            "4" * 64,
            "--selection-digest",
            "5" * 64,
            "--trusted-now",
            NOW.isoformat(),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert captured["paper_reconciliation_path"] == reconciliation


def test_serving_package_parser_requires_explicit_reconciliation() -> None:
    from trade_rl.cli import extended

    with pytest.raises(SystemExit):
        extended._serving_package_parser().parse_args(
            [
                "--training-run",
                "training",
                "--confirmation",
                "confirmation.json",
                "--confirmation-public-keys",
                "keys.json",
                "--output",
                "bundle",
                "--signal-digest",
                "1" * 64,
                "--selection-digest",
                "2" * 64,
                "--trusted-now",
                NOW.isoformat(),
            ]
        )
