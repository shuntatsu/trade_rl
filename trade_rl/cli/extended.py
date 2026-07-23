"""Artifact-producing and offline-approval command handlers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from trade_rl.release.asymmetric import load_public_verification_keys

_CONFIRMATION_REQUEST_SCHEMA = "fresh_confirmation_request_v1"
_CONFIRMATION_REQUEST_FIELDS = {
    "created_at",
    "dataset_id",
    "dependency_digest",
    "end_time",
    "environment_digest",
    "fill_log_digest",
    "git_commit",
    "order_log_digest",
    "policy_digest",
    "reconciliation_digest",
    "required_after",
    "return_period_hours",
    "returns",
    "schema_version",
    "start_time",
    "training_run_digest",
}
_PAPER_RECONCILIATION_REQUEST_SCHEMA = "paper_reconciliation_request_v1"
_PAPER_RECONCILIATION_REQUEST_FIELDS = {
    "cash_tolerance_fraction",
    "created_at",
    "dataset_id",
    "duplicate_fill_count",
    "end_time",
    "environment_digest",
    "equity_tolerance_fraction",
    "fill_log_digest",
    "matched_fill_count",
    "maximum_cash_difference_fraction",
    "maximum_equity_difference_fraction",
    "maximum_position_notional_difference_fraction",
    "observed_fill_count",
    "open_order_count",
    "order_log_digest",
    "policy_digest",
    "position_notional_tolerance_fraction",
    "schema_version",
    "start_time",
    "submitted_order_count",
    "terminal_order_count",
    "training_run_digest",
    "unknown_order_fill_count",
}


def execute_training_run(**kwargs: Any) -> Any:
    """Lazy adapter retained as a monkeypatchable CLI boundary."""

    from trade_rl.workflows.training_run import execute_training_run as implementation

    return implementation(**kwargs)


def execute_market_walk_forward(**kwargs: Any) -> Any:
    """Lazy adapter retained as a monkeypatchable CLI boundary."""

    from trade_rl.workflows.market_walk_forward import (
        execute_market_walk_forward as implementation,
    )

    return implementation(**kwargs)


def package_selected_training_run(**kwargs: Any) -> Any:
    """Lazy adapter retained as a monkeypatchable CLI boundary."""

    from trade_rl.serving.package import package_selected_training_run as implementation

    return implementation(**kwargs)


def _write_json(stream: TextIO, payload: object) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    stream.write("\n")


def _artifact_run_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    return parser


def _training_run_parser() -> argparse.ArgumentParser:
    parser = _artifact_run_parser("trade-rl train run")
    parser.add_argument("--selection-proposal", type=Path)
    parser.add_argument("--selection-authorization", type=Path)
    parser.add_argument("--selection-public-keys", type=Path)
    parser.add_argument("--require-selection-authorization", action="store_true")
    parser.add_argument("--execution-evidence", type=Path)
    return parser


def _serving_package_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl serving package")
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--paper-reconciliation", type=Path, required=True)
    parser.add_argument("--confirmation-public-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal-digest", required=True)
    parser.add_argument("--selection-digest", required=True)
    parser.add_argument("--trusted-now", required=True)
    return parser


def _selection_authorize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl selection authorize")
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--approver", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _release_approve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl release approve")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--dependency-digest", required=True)
    parser.add_argument("--approver", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _confirmation_create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl confirmation create")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _reconciliation_create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl reconciliation create")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _parse_datetime(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _error(
    stderr: TextIO,
    error: Exception,
    *,
    schema: str,
) -> int:
    _write_json(
        stderr,
        {
            "error": str(error),
            "error_type": type(error).__name__,
            "production_status": "NO-GO",
            "schema": schema,
            "status": "failed",
        },
    )
    return 1


def _run_selection_authorize(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    args = _selection_authorize_parser().parse_args(list(argv))
    try:
        from trade_rl.release.offline_keys import load_offline_signing_key
        from trade_rl.workflows.offline_selection_approval import (
            create_selection_authorization,
        )
        from trade_rl.workflows.selection_authorization import (
            load_selection_proposal,
            write_selection_authorization,
        )

        proposal = load_selection_proposal(args.proposal)
        key = load_offline_signing_key(
            args.private_key,
            required_purpose="selection-authorization",
        )
        authorization = create_selection_authorization(
            proposal,
            approver=args.approver,
            approved_at=_parse_datetime(args.approved_at, field="approved-at"),
            expires_at=_parse_datetime(args.expires_at, field="expires-at"),
            key_id=key.key_id,
            private_key=key.private_key,
        )
        path = write_selection_authorization(args.output, authorization)
    except Exception as error:
        return _error(stderr, error, schema="selection_authorization_error_v1")
    _write_json(
        stdout,
        {
            "artifact_path": str(path),
            "authorization_digest": authorization.authorization_digest,
            "key_id": authorization.key_id,
            "production_status": "NO-GO",
            "proposal_digest": proposal.digest,
            "schema": "selection_authorization_result_v1",
            "status": "authorized_for_selected_final_training",
        },
    )
    return 0


def _run_release_approve(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    args = _release_approve_parser().parse_args(list(argv))
    try:
        from trade_rl.release.attestation import (
            default_attestation_path,
            write_release_attestation,
        )
        from trade_rl.release.offline_approval import create_release_attestation
        from trade_rl.release.offline_keys import load_offline_signing_key
        from trade_rl.serving.bundle import load_serving_bundle

        bundle = load_serving_bundle(args.bundle)
        if bundle.release is not None:
            raise ValueError(
                "serving bundle already has an external release attestation"
            )
        manifest = bundle.manifest
        key = load_offline_signing_key(
            args.private_key,
            required_purpose="release-verification",
        )
        attestation = create_release_attestation(
            bundle_digest=manifest.bundle_digest,
            dataset_id=manifest.dataset_id,
            training_run_digest=manifest.training_run_digest,
            run_kind=manifest.run_kind,
            selection_proposal_digest=manifest.selection_proposal_digest,
            selection_authorization_digest=manifest.selection_authorization_digest,
            walk_forward_run_digest=manifest.walk_forward_run_digest,
            gate_evidence_digest=manifest.gate_evidence_digest,
            confirmation_evidence_digest=manifest.confirmation_evidence_digest,
            selected_policy_digest=manifest.policy_digest,
            git_commit=args.git_commit,
            dependency_digest=args.dependency_digest,
            approver=args.approver,
            approved_at=_parse_datetime(args.approved_at, field="approved-at"),
            expires_at=_parse_datetime(args.expires_at, field="expires-at"),
            key_id=key.key_id,
            private_key=key.private_key,
        )
        output = args.output or default_attestation_path(args.bundle)
        path = write_release_attestation(output, attestation)
    except Exception as error:
        return _error(stderr, error, schema="release_approval_error_v1")
    _write_json(
        stdout,
        {
            "artifact_path": str(path),
            "attestation_digest": attestation.attestation_digest,
            "bundle_digest": attestation.bundle_digest,
            "key_id": attestation.key_id,
            "production_status": "NO-GO",
            "schema": "release_approval_result_v1",
            "status": "approved_for_release_activation",
        },
    )
    return 0


def _strict_confirmation_request(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("confirmation request must be an object")
    if set(raw) != _CONFIRMATION_REQUEST_FIELDS:
        raise ValueError("confirmation request fields are invalid")
    if raw.get("schema_version") != _CONFIRMATION_REQUEST_SCHEMA:
        raise ValueError("confirmation request schema is unsupported")
    return raw


def _request_string(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"confirmation request {field} must be a non-empty string")
    return value


def _request_returns(raw: Mapping[str, object]) -> tuple[float, ...]:
    values = raw.get("returns")
    if not isinstance(values, list) or not values:
        raise ValueError("confirmation request returns must be a non-empty list")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in values
    ):
        raise ValueError("confirmation request returns must contain numbers")
    return tuple(float(item) for item in values)


def _request_number(raw: Mapping[str, object], field: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"confirmation request {field} must be numeric")
    return float(value)


def _strict_reconciliation_request(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("paper reconciliation request must be an object")
    if set(raw) != _PAPER_RECONCILIATION_REQUEST_FIELDS:
        raise ValueError("paper reconciliation request fields are invalid")
    if raw.get("schema_version") != _PAPER_RECONCILIATION_REQUEST_SCHEMA:
        raise ValueError("paper reconciliation request schema is unsupported")
    return raw


def _reconciliation_string(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"paper reconciliation request {field} must be a non-empty string"
        )
    return value


def _reconciliation_integer(raw: Mapping[str, object], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"paper reconciliation request {field} must be an integer")
    return value


def _reconciliation_number(raw: Mapping[str, object], field: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"paper reconciliation request {field} must be numeric")
    return float(value)


def _run_confirmation_create(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    args = _confirmation_create_parser().parse_args(list(argv))
    try:
        from trade_rl.evaluation.confirmation import write_confirmation_evidence
        from trade_rl.evaluation.offline_confirmation import (
            create_fresh_confirmation_evidence,
        )
        from trade_rl.release.offline_keys import load_offline_signing_key

        raw = _strict_confirmation_request(args.request)
        key = load_offline_signing_key(
            args.private_key,
            required_purpose="fresh-confirmation",
        )
        evidence = create_fresh_confirmation_evidence(
            dataset_id=_request_string(raw, "dataset_id"),
            environment_digest=_request_string(raw, "environment_digest"),
            policy_digest=_request_string(raw, "policy_digest"),
            training_run_digest=_request_string(raw, "training_run_digest"),
            git_commit=_request_string(raw, "git_commit"),
            dependency_digest=_request_string(raw, "dependency_digest"),
            required_after=_parse_datetime(
                _request_string(raw, "required_after"),
                field="required-after",
            ),
            start_time=_parse_datetime(
                _request_string(raw, "start_time"),
                field="start-time",
            ),
            end_time=_parse_datetime(
                _request_string(raw, "end_time"),
                field="end-time",
            ),
            returns=_request_returns(raw),
            return_period_hours=_request_number(raw, "return_period_hours"),
            order_log_digest=_request_string(raw, "order_log_digest"),
            fill_log_digest=_request_string(raw, "fill_log_digest"),
            reconciliation_digest=_request_string(raw, "reconciliation_digest"),
            created_at=_parse_datetime(
                _request_string(raw, "created_at"),
                field="created-at",
            ),
            key_id=key.key_id,
            private_key=key.private_key,
        )
        path = write_confirmation_evidence(args.output, evidence)
    except Exception as error:
        return _error(stderr, error, schema="confirmation_creation_error_v1")
    _write_json(
        stdout,
        {
            "artifact_path": str(path),
            "evidence_digest": evidence.evidence_digest,
            "key_id": evidence.envelope.key_id,
            "production_status": "NO-GO",
            "schema": "confirmation_creation_result_v1",
            "status": "sealed_for_fresh_confirmation_review",
        },
    )
    return 0


def _run_reconciliation_create(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    args = _reconciliation_create_parser().parse_args(list(argv))
    try:
        from trade_rl.evaluation.paper_reconciliation import (
            PaperReconciliationEvidence,
            write_paper_reconciliation_evidence,
        )

        raw = _strict_reconciliation_request(args.request)
        evidence = PaperReconciliationEvidence.create(
            dataset_id=_reconciliation_string(raw, "dataset_id"),
            environment_digest=_reconciliation_string(raw, "environment_digest"),
            policy_digest=_reconciliation_string(raw, "policy_digest"),
            training_run_digest=_reconciliation_string(raw, "training_run_digest"),
            start_time=_parse_datetime(
                _reconciliation_string(raw, "start_time"), field="start-time"
            ),
            end_time=_parse_datetime(
                _reconciliation_string(raw, "end_time"), field="end-time"
            ),
            created_at=_parse_datetime(
                _reconciliation_string(raw, "created_at"), field="created-at"
            ),
            order_log_digest=_reconciliation_string(raw, "order_log_digest"),
            fill_log_digest=_reconciliation_string(raw, "fill_log_digest"),
            submitted_order_count=_reconciliation_integer(raw, "submitted_order_count"),
            terminal_order_count=_reconciliation_integer(raw, "terminal_order_count"),
            observed_fill_count=_reconciliation_integer(raw, "observed_fill_count"),
            matched_fill_count=_reconciliation_integer(raw, "matched_fill_count"),
            unknown_order_fill_count=_reconciliation_integer(
                raw, "unknown_order_fill_count"
            ),
            duplicate_fill_count=_reconciliation_integer(raw, "duplicate_fill_count"),
            open_order_count=_reconciliation_integer(raw, "open_order_count"),
            maximum_position_notional_difference_fraction=_reconciliation_number(
                raw, "maximum_position_notional_difference_fraction"
            ),
            maximum_cash_difference_fraction=_reconciliation_number(
                raw, "maximum_cash_difference_fraction"
            ),
            maximum_equity_difference_fraction=_reconciliation_number(
                raw, "maximum_equity_difference_fraction"
            ),
            position_notional_tolerance_fraction=_reconciliation_number(
                raw, "position_notional_tolerance_fraction"
            ),
            cash_tolerance_fraction=_reconciliation_number(
                raw, "cash_tolerance_fraction"
            ),
            equity_tolerance_fraction=_reconciliation_number(
                raw, "equity_tolerance_fraction"
            ),
        )
        path = write_paper_reconciliation_evidence(args.output, evidence)
    except Exception as error:
        return _error(
            stderr,
            error,
            schema="paper_reconciliation_creation_error_v1",
        )
    _write_json(
        stdout,
        {
            "artifact_path": str(path),
            "evidence_digest": evidence.evidence_digest,
            "passed": evidence.passed,
            "production_status": "NO-GO",
            "schema": "paper_reconciliation_creation_result_v1",
            "status": "sealed_for_fresh_confirmation_review",
        },
    )
    return 0


def _run_serving_package(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    args = _serving_package_parser().parse_args(list(argv))
    try:
        manifest = package_selected_training_run(
            training_root=args.training_run,
            confirmation_path=args.confirmation,
            paper_reconciliation_path=args.paper_reconciliation,
            output_root=args.output,
            signal_digest=args.signal_digest,
            selection_digest=args.selection_digest,
            trusted_confirmation_keys=load_public_verification_keys(
                args.confirmation_public_keys
            ),
            trusted_now=_parse_datetime(args.trusted_now, field="trusted-now"),
        )
    except Exception as error:
        return _error(stderr, error, schema="serving_package_error_v1")
    _write_json(
        stdout,
        {
            "artifact_path": str(args.output),
            "bundle_digest": manifest.bundle_digest,
            "confirmation_evidence_digest": manifest.confirmation_evidence_digest,
            "production_status": "NO-GO",
            "run_kind": manifest.run_kind,
            "schema": "serving_package_result_v1",
            "status": "packaged_for_external_release_review",
            "training_run_digest": manifest.training_run_digest,
        },
    )
    return 0


def _run_training(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    args = _training_run_parser().parse_args(list(argv))
    try:
        result = execute_training_run(
            config_path=args.config,
            dataset_path=args.dataset,
            store_root=args.output,
            run_id=args.run_id,
            selection_proposal_path=args.selection_proposal,
            selection_authorization_path=args.selection_authorization,
            selection_public_keys_path=args.selection_public_keys,
            require_selection_authorization=args.require_selection_authorization,
            execution_evidence_path=args.execution_evidence,
        )
    except Exception as error:
        return _error(stderr, error, schema="training_run_error_v1")
    _write_json(
        stdout,
        {
            "artifact_path": str(result.path),
            "dataset_id": result.dataset_id,
            "policy_digest": result.policy_digest,
            "production_status": result.production_status,
            "run_digest": result.run_digest,
            "run_id": result.run_id,
            "run_kind": result.run_kind,
            "schema": "training_run_result_v1",
            "selection_authorization_digest": result.selection_authorization_digest,
            "selection_proposal_digest": result.selection_proposal_digest,
            "status": result.status,
        },
    )
    return 0


def _run_walk_forward(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    args = _artifact_run_parser("trade-rl walk-forward run").parse_args(list(argv))
    try:
        result = execute_market_walk_forward(
            config_path=args.config,
            dataset_path=args.dataset,
            store_root=args.output,
            run_id=args.run_id,
        )
    except Exception as error:
        return _error(stderr, error, schema="walk_forward_run_error_v1")
    _write_json(
        stdout,
        {
            "artifact_path": str(result.path),
            "dataset_id": result.dataset_id,
            "evaluation_digest": result.evaluation_digest,
            "production_status": result.production_status,
            "run_digest": result.run_digest,
            "run_id": result.run_id,
            "schema": "walk_forward_run_result_v1",
            "status": result.status,
        },
    )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Dispatch artifact-producing and explicitly offline approval commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    if arguments[:2] == ["train", "run"]:
        return _run_training(arguments[2:], stdout=output, stderr=errors)
    if arguments[:2] == ["walk-forward", "run"]:
        return _run_walk_forward(arguments[2:], stdout=output, stderr=errors)
    if arguments[:2] == ["serving", "package"]:
        return _run_serving_package(arguments[2:], stdout=output, stderr=errors)
    if arguments[:2] == ["selection", "authorize"]:
        return _run_selection_authorize(arguments[2:], stdout=output, stderr=errors)
    if arguments[:2] == ["release", "approve"]:
        return _run_release_approve(arguments[2:], stdout=output, stderr=errors)
    if arguments[:2] == ["confirmation", "create"]:
        return _run_confirmation_create(arguments[2:], stdout=output, stderr=errors)
    if arguments[:2] == ["reconciliation", "create"]:
        return _run_reconciliation_create(arguments[2:], stdout=output, stderr=errors)
    raise ValueError("unsupported artifact-producing CLI command")


if __name__ == "__main__":
    raise SystemExit(main())
