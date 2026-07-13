from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade_rl.domain.datasets import DatasetManifest
from trade_rl.domain.evaluation import GateCheck, GateDecision
from trade_rl.domain.policies import PolicyEnsembleManifest, PolicyMember
from trade_rl.domain.releases import ReleaseManifest
from trade_rl.domain.selection import PolicyMode, SelectionDecision
from trade_rl.domain.signals import SignalArtifactManifest, SignalStatus


DATASET_ID = "a" * 64
SIGNAL_DIGEST = "b" * 64
POLICY_DIGEST = "c" * 64
EVALUATION_DIGEST = "d" * 64
NOW = datetime(2026, 7, 13, tzinfo=UTC)


def dataset() -> DatasetManifest:
    return DatasetManifest(
        dataset_id=DATASET_ID,
        symbols=("BTCUSDT", "ETHUSDT"),
        feature_names=("ret_z1", "rsi"),
        base_timeframe="15m",
        created_at=NOW,
    )


def rejected_signal() -> SignalArtifactManifest:
    return SignalArtifactManifest(
        digest=SIGNAL_DIGEST,
        dataset_id=DATASET_ID,
        model_kind="gbm",
        target="cs_demean",
        horizon=12,
        status=SignalStatus.REJECTED,
        alpha_enabled=False,
        created_at=NOW,
    )


def test_baseline_only_selection_has_no_policy_digest() -> None:
    decision = SelectionDecision(
        dataset_id=DATASET_ID,
        mode=PolicyMode.BASELINE_ONLY,
        selected_configuration="A",
        selected_policy_digest=None,
        signal_digest=SIGNAL_DIGEST,
        evaluation_digest=EVALUATION_DIGEST,
        selected_at=NOW,
        reasons=("A is the identity baseline",),
    )

    assert decision.selected_policy_digest is None


def test_baseline_only_selection_rejects_policy_digest() -> None:
    with pytest.raises(ValueError, match="baseline_only"):
        SelectionDecision(
            dataset_id=DATASET_ID,
            mode=PolicyMode.BASELINE_ONLY,
            selected_configuration="A",
            selected_policy_digest=POLICY_DIGEST,
            signal_digest=SIGNAL_DIGEST,
            evaluation_digest=EVALUATION_DIGEST,
            selected_at=NOW,
            reasons=("identity baseline",),
        )


def test_residual_selection_requires_policy_digest() -> None:
    with pytest.raises(ValueError, match="policy digest"):
        SelectionDecision(
            dataset_id=DATASET_ID,
            mode=PolicyMode.RESIDUAL_POLICY,
            selected_configuration="B",
            selected_policy_digest=None,
            signal_digest=SIGNAL_DIGEST,
            evaluation_digest=EVALUATION_DIGEST,
            selected_at=NOW,
            reasons=("candidate won",),
        )


def test_policy_ensemble_requires_declared_members() -> None:
    members = (
        PolicyMember(seed=0, checkpoint_digest="1" * 64),
        PolicyMember(seed=1, checkpoint_digest="2" * 64),
    )

    with pytest.raises(ValueError, match="member count"):
        PolicyEnsembleManifest(
            digest=POLICY_DIGEST,
            dataset_id=DATASET_ID,
            action_schema="baseline_residual_v1",
            expected_members=3,
            members=members,
            created_at=NOW,
        )


def test_rejected_signal_cannot_enable_alpha() -> None:
    with pytest.raises(ValueError, match="rejected signal"):
        SignalArtifactManifest(
            digest=SIGNAL_DIGEST,
            dataset_id=DATASET_ID,
            model_kind="gbm",
            target="cs_demean",
            horizon=12,
            status=SignalStatus.REJECTED,
            alpha_enabled=True,
            created_at=NOW,
        )


def test_failed_mandatory_gate_blocks_release() -> None:
    gate = GateDecision(
        passed=False,
        checks=(
            GateCheck(name="positive_holdout_return", passed=True, mandatory=True),
            GateCheck(name="positive_return_significant", passed=False, mandatory=True),
        ),
        decided_at=NOW,
    )
    selection = SelectionDecision(
        dataset_id=DATASET_ID,
        mode=PolicyMode.BASELINE_ONLY,
        selected_configuration="A",
        selected_policy_digest=None,
        signal_digest=SIGNAL_DIGEST,
        evaluation_digest=EVALUATION_DIGEST,
        selected_at=NOW,
        reasons=("identity baseline",),
    )

    with pytest.raises(ValueError, match="mandatory gate"):
        ReleaseManifest.create(
            version="2026.07.13",
            git_commit="e" * 40,
            dataset=dataset(),
            signal=rejected_signal(),
            selection=selection,
            gate=gate,
            bundle_digest="f" * 64,
            created_at=NOW,
        )
