from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.release.asymmetric import (
    PublicVerificationKey,
)
from trade_rl.release.offline_signing import generate_private_key, public_key_bytes
from trade_rl.workflows.offline_selection_approval import create_selection_authorization
from trade_rl.workflows.selection_authorization import (
    SelectionAuthorization,
    SelectionProposal,
    load_selection_authorization,
    load_selection_proposal,
    write_selection_authorization,
    write_selection_proposal,
)

NOW = datetime(2026, 7, 18, tzinfo=UTC)
PRIVATE_KEY = generate_private_key()
PUBLIC_KEY = PublicVerificationKey(
    key_id="selection-2026",
    public_key=public_key_bytes(PRIVATE_KEY),
    purpose="selection-authorization",
    valid_from=NOW - timedelta(days=1),
    valid_until=NOW + timedelta(days=365),
)


def _proposal() -> SelectionProposal:
    return SelectionProposal.create(
        walk_forward_run_digest="a" * 64,
        gate_evidence_digest="b" * 64,
        execution_sensitivity_digest="c" * 64,
        dataset_id="d" * 64,
        selected_configuration="ppo-15m-target",
        candidate_config_digest="e" * 64,
        seeds=(0, 1, 2),
        git_commit="f" * 40,
        dependency_digest="1" * 64,
        resume_checkpoint_digests=(),
    )


def _authorization(proposal: SelectionProposal) -> SelectionAuthorization:
    return create_selection_authorization(
        proposal,
        approver="research-approver",
        approved_at=NOW,
        expires_at=NOW + timedelta(days=7),
        key_id=PUBLIC_KEY.key_id,
        private_key=PRIVATE_KEY,
    )


def test_selection_proposal_and_authorization_round_trip(tmp_path: Path) -> None:
    proposal = _proposal()
    authorization = _authorization(proposal)
    proposal_path = write_selection_proposal(
        tmp_path / "selection-proposal.json", proposal
    )
    authorization_path = write_selection_authorization(
        tmp_path / "selection-authorization.json",
        authorization,
    )

    loaded_proposal = load_selection_proposal(proposal_path)
    loaded_authorization = load_selection_authorization(authorization_path)

    assert loaded_proposal == proposal
    assert loaded_authorization == authorization
    loaded_authorization.verify(
        loaded_proposal,
        trusted_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
        trusted_at=NOW,
    )


def test_selection_loader_rejects_seed_type_coercion(tmp_path: Path) -> None:
    proposal = _proposal().to_mapping()
    proposal["seeds"] = ["0", 1, 2]
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ValueError, match="proposal"):
        load_selection_proposal(path)


def test_selection_authorization_is_immutable(tmp_path: Path) -> None:
    proposal = _proposal()
    first = _authorization(proposal)
    path = write_selection_authorization(tmp_path / "authorization.json", first)
    other = create_selection_authorization(
        proposal,
        approver="other-approver",
        approved_at=NOW,
        expires_at=NOW + timedelta(days=7),
        key_id=PUBLIC_KEY.key_id,
        private_key=PRIVATE_KEY,
    )

    with pytest.raises(FileExistsError, match="overwrite"):
        write_selection_authorization(path, other)


def test_selection_authorization_rejects_expired_approval() -> None:
    proposal = _proposal()
    authorization = _authorization(proposal)

    with pytest.raises(ValueError, match="expired"):
        authorization.verify(
            proposal,
            trusted_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_at=NOW + timedelta(days=8),
        )


def test_selection_authorization_rejects_different_proposal() -> None:
    proposal = _proposal()
    authorization = _authorization(proposal)
    other = SelectionProposal.create(
        walk_forward_run_digest=proposal.walk_forward_run_digest,
        gate_evidence_digest=proposal.gate_evidence_digest,
        execution_sensitivity_digest=proposal.execution_sensitivity_digest,
        dataset_id=proposal.dataset_id,
        selected_configuration=proposal.selected_configuration,
        candidate_config_digest="2" * 64,
        seeds=proposal.seeds,
        git_commit=proposal.git_commit,
        dependency_digest=proposal.dependency_digest,
        resume_checkpoint_digests=proposal.resume_checkpoint_digests,
    )

    with pytest.raises(ValueError, match="proposal digest"):
        authorization.verify(
            other,
            trusted_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_at=NOW,
        )
