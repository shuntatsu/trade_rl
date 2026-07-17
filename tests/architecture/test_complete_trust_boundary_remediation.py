from __future__ import annotations

import base64
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.release.asymmetric import (
    PublicVerificationKey,
    generate_private_key,
    public_key_bytes,
)
from trade_rl.workflows.selection_authorization import (
    SelectionAuthorization,
    SelectionProposal,
)

NOW = datetime(2026, 7, 18, 1, 0, tzinfo=UTC)
PRIVATE_KEY = generate_private_key()
PUBLIC_KEY = PublicVerificationKey(
    key_id="approval-2026",
    public_key=public_key_bytes(PRIVATE_KEY),
    purpose="selection-authorization",
    valid_from=NOW - timedelta(days=1),
    valid_until=NOW + timedelta(days=365),
)


def _proposal(**overrides: object) -> SelectionProposal:
    values: dict[str, object] = {
        "walk_forward_run_digest": "a" * 64,
        "gate_evidence_digest": "b" * 64,
        "execution_sensitivity_digest": "c" * 64,
        "dataset_id": "d" * 64,
        "selected_configuration": "oracle-bc-ppo-15m-target",
        "candidate_config_digest": "e" * 64,
        "seeds": (0, 1, 2),
        "git_commit": "f" * 40,
        "dependency_digest": "1" * 64,
        "resume_checkpoint_digests": (),
    }
    values.update(overrides)
    return SelectionProposal.create(**values)


def test_selection_authorization_uses_public_key_and_rejects_resume_injection() -> None:
    proposal = _proposal()
    authorization = SelectionAuthorization.authorize(
        proposal,
        approver="research-approver",
        approved_at=NOW,
        expires_at=NOW + timedelta(days=7),
        key_id=PUBLIC_KEY.key_id,
        private_key=PRIVATE_KEY,
    )
    authorization.verify(
        proposal,
        trusted_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
        trusted_at=NOW,
    )

    injected = _proposal(resume_checkpoint_digests=((0, "2" * 64),))
    with pytest.raises(ValueError, match="resume|proposal|digest"):
        authorization.verify(
            injected,
            trusted_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_at=NOW,
        )


def test_selection_authorization_rejects_forged_self_hash() -> None:
    proposal = _proposal()
    raw = {
        "proposal_digest": proposal.digest,
        "approver": "attacker",
        "approved_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(days=7)).isoformat(),
        "key_id": PUBLIC_KEY.key_id,
        "signature": base64.b64encode(b"x" * 64).decode("ascii"),
        "schema_version": "selection_authorization_ed25519_v2",
    }
    forged = SelectionAuthorization.from_mapping(raw)
    with pytest.raises(ValueError, match="signature"):
        forged.verify(
            proposal,
            trusted_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_at=NOW,
        )


def test_confirmation_must_begin_after_frozen_boundary() -> None:
    from trade_rl.evaluation.confirmation import FreshConfirmationEvidence

    required_after = NOW
    evidence = FreshConfirmationEvidence.create(
        dataset_id="1" * 64,
        environment_digest="2" * 64,
        policy_digest="3" * 64,
        training_run_digest="4" * 64,
        git_commit="5" * 40,
        dependency_digest="6" * 64,
        start_time=required_after - timedelta(days=1),
        end_time=required_after + timedelta(days=29),
        returns=(0.0001,) * (30 * 24),
        return_period_hours=1.0,
        order_log_digest="7" * 64,
        fill_log_digest="8" * 64,
        reconciliation_digest="9" * 64,
        created_at=required_after + timedelta(days=29),
        required_after=required_after,
        key_id="confirmation-2026",
        private_key=PRIVATE_KEY,
    )
    confirmation_key = PublicVerificationKey(
        key_id="confirmation-2026",
        public_key=public_key_bytes(PRIVATE_KEY),
        purpose="fresh-confirmation",
        valid_from=NOW - timedelta(days=1),
        valid_until=NOW + timedelta(days=365),
    )
    with pytest.raises(ValueError, match="required|fresh|start"):
        evidence.verify(
            {confirmation_key.key_id: confirmation_key},
            expected_required_after=required_after,
            trusted_now=required_after + timedelta(days=30),
        )


def test_confirmation_rejects_future_interval() -> None:
    from trade_rl.evaluation.confirmation import FreshConfirmationEvidence

    required_after = NOW
    evidence = FreshConfirmationEvidence.create(
        dataset_id="1" * 64,
        environment_digest="2" * 64,
        policy_digest="3" * 64,
        training_run_digest="4" * 64,
        git_commit="5" * 40,
        dependency_digest="6" * 64,
        start_time=required_after + timedelta(days=1),
        end_time=required_after + timedelta(days=31),
        returns=(0.0001,) * (30 * 24),
        return_period_hours=1.0,
        order_log_digest="7" * 64,
        fill_log_digest="8" * 64,
        reconciliation_digest="9" * 64,
        created_at=required_after,
        required_after=required_after,
        key_id="confirmation-2026",
        private_key=PRIVATE_KEY,
    )
    confirmation_key = PublicVerificationKey(
        key_id="confirmation-2026",
        public_key=public_key_bytes(PRIVATE_KEY),
        purpose="fresh-confirmation",
        valid_from=NOW - timedelta(days=1),
        valid_until=NOW + timedelta(days=365),
    )
    with pytest.raises(ValueError, match="future|trusted"):
        evidence.verify(
            {confirmation_key.key_id: confirmation_key},
            expected_required_after=required_after,
            trusted_now=required_after + timedelta(days=2),
        )


def test_verified_binance_history_requires_a_valid_signature() -> None:
    from trade_rl.workflows.binance_metadata_modes import (
        load_verified_binance_rule_history,
    )

    payload = {
        "schema_version": "binance_instrument_rule_history_v4",
        "policy_version": "binance_metadata_modes_v2",
        "market": "usds-m",
        "symbol_order": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "coverage": {
            "start_time": "2024-12-01T00:00:00+00:00",
            "end_time": "2026-07-01T00:00:00+00:00",
        },
        "issued_at": "2026-07-17T00:00:00+00:00",
        "source_uri": "operator://signed-binance-rules",
        "symbols": {},
    }
    document = {
        "payload": payload,
        "envelope": {
            "key_id": "metadata-2026",
            "purpose": "binance-rule-history",
            "payload_digest": content_digest(payload),
            "signed_at": NOW.isoformat(),
            "signature": base64.b64encode(b"x" * 64).decode("ascii"),
            "schema_version": "signed_evidence_ed25519_v1",
            "algorithm": "ed25519",
        },
    }
    metadata_key = PublicVerificationKey(
        key_id="metadata-2026",
        public_key=public_key_bytes(PRIVATE_KEY),
        purpose="binance-rule-history",
        valid_from=NOW - timedelta(days=1),
        valid_until=NOW + timedelta(days=365),
    )
    with pytest.raises(ValueError, match="signature"):
        load_verified_binance_rule_history(
            document,
            trusted_keys={metadata_key.key_id: metadata_key},
            trusted_now=NOW,
        )


def test_training_manifest_selected_final_requires_authorization_chain(
    tmp_path: Path,
) -> None:
    from trade_rl.artifacts.run_manifest import TrainingRunManifest

    artifact = tmp_path / "ensemble.json"
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="authorization|selected"):
        TrainingRunManifest.build(
            root=tmp_path,
            run_id="selected-final",
            dataset_id="1" * 64,
            environment_digest="2" * 64,
            ensemble_digest="3" * 64,
            training_config_digest="4" * 64,
            provenance_digest="5" * 64,
            artifact_paths=("ensemble.json",),
            created_at=NOW,
            run_kind="research_selected_final",
            selection_proposal_digest=None,
            selection_authorization_digest=None,
            walk_forward_run_digest=None,
            gate_evidence_digest=None,
            completed_at=NOW,
        )


def test_workflow_checker_cannot_be_satisfied_by_comments(tmp_path: Path) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    checker_path = (
        Path(__file__).resolve().parents[2] / ".github" / "check_workflow_security.py"
    )
    spec = spec_from_file_location("workflow_security", checker_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "unsafe.yml").write_text(
        """name: unsafe
on: workflow_dispatch
permissions:
  contents: read
jobs:
  attack:
    runs-on: [self-hosted, gpu]
    # environment: gpu-full-training
    # github.actor == github.repository_owner
    # github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
""",
        encoding="utf-8",
    )
    errors = module.validate_workflow_security(tmp_path)
    assert any("environment" in error for error in errors)
    assert any("actor" in error or "owner" in error for error in errors)
    assert any("mutable" in error for error in errors)


def _completed(
    command: tuple[str, ...], stdout: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_supervisor_absent_expected_generation_is_failure() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "binance-multitimeframe"
        / "full_run_supervisor.py"
    )
    spec = spec_from_file_location("full_run_supervisor", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return _completed(command, "")

    with pytest.raises(RuntimeError, match="absent|missing"):
        module.supervised_run_status(
            expected_generation="generation-1",
            expected_git_commit="a" * 40,
            runner=runner,
        )


def test_training_image_is_digest_pinned_and_generation_scoped() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile.training").read_text(encoding="utf-8")
    compose = (root / "compose.training.yaml").read_text(encoding="utf-8")
    assert "python:3.12-slim@sha256:" in dockerfile
    assert (
        "/workspace/var/runs/${TRADE_RL_RUN_GENERATION}/cuda-preflight.json"
        in dockerfile
    )
    assert "TRADE_RL_METADATA_KEYS" not in compose
    assert "TRADE_RL_CONFIRMATION_KEYS" not in compose


def test_privileged_workflows_checkout_the_event_sha() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        ".github/workflows/launch-binance-frozen-226.yml",
        ".github/workflows/gpu-nightly.yml",
        ".github/workflows/multitimeframe-live-full.yml",
    ):
        content = (root / relative).read_text(encoding="utf-8")
        assert "ref: ${{ github.sha }}" in content
        assert "ref: main" not in content
