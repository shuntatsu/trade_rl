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
TRAINING_CONFIG_DIGEST = "e" * 64
GATE_EVALUATION_DIGEST = "f" * 64
ENVIRONMENT_DIGEST = "1" * 64
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


def policy_manifest(
    *,
    expected_members: int,
    members: tuple[PolicyMember, ...],
    requested_timesteps: int = 1_024,
    actual_timesteps: int = 2_048,
    initial_capital: float = 250_000.0,
) -> PolicyEnsembleManifest:
    return PolicyEnsembleManifest(
        digest=POLICY_DIGEST,
        dataset_id=DATASET_ID,
        action_schema="baseline_residual_v1",
        observation_schema="baseline_residual_observation_v2",
        training_config_digest=TRAINING_CONFIG_DIGEST,
        environment_digest=ENVIRONMENT_DIGEST,
        initial_capital=initial_capital,
        requested_timesteps=requested_timesteps,
        actual_timesteps=actual_timesteps,
        resolved_device="cpu",
        expected_members=expected_members,
        members=members,
        created_at=NOW,
    )


def baseline_selection() -> SelectionDecision:
    return SelectionDecision(
        dataset_id=DATASET_ID,
        mode=PolicyMode.BASELINE_ONLY,
        selected_configuration="A",
        selected_policy_digest=None,
        signal_digest=SIGNAL_DIGEST,
        evaluation_digest=EVALUATION_DIGEST,
        selected_at=NOW,
        reasons=("A is the identity baseline",),
    )


def residual_selection() -> SelectionDecision:
    return SelectionDecision(
        dataset_id=DATASET_ID,
        mode=PolicyMode.RESIDUAL_POLICY,
        selected_configuration="B",
        selected_policy_digest=POLICY_DIGEST,
        signal_digest=SIGNAL_DIGEST,
        evaluation_digest=EVALUATION_DIGEST,
        selected_at=NOW,
        reasons=("candidate won",),
    )


def passing_gate(
    *,
    dataset_id: str = DATASET_ID,
    selected_policy_digest: str | None = None,
) -> GateDecision:
    return GateDecision(
        dataset_id=dataset_id,
        selected_policy_digest=selected_policy_digest,
        evaluation_digest=GATE_EVALUATION_DIGEST,
        passed=True,
        checks=(GateCheck(name="positive_holdout_return", passed=True),),
        decided_at=NOW,
    )


def test_baseline_only_selection_has_no_policy_digest() -> None:
    assert baseline_selection().selected_policy_digest is None


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
        policy_manifest(expected_members=3, members=members)


def test_policy_ensemble_rejects_actual_work_below_request() -> None:
    members = (PolicyMember(seed=0, checkpoint_digest="1" * 64),)

    with pytest.raises(ValueError, match="actual_timesteps"):
        policy_manifest(
            expected_members=1,
            members=members,
            requested_timesteps=2_048,
            actual_timesteps=1_024,
        )


def test_policy_ensemble_rejects_non_positive_aum() -> None:
    members = (PolicyMember(seed=0, checkpoint_digest="1" * 64),)

    with pytest.raises(ValueError, match="initial_capital"):
        policy_manifest(
            expected_members=1,
            members=members,
            initial_capital=0.0,
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
        dataset_id=DATASET_ID,
        selected_policy_digest=None,
        evaluation_digest=GATE_EVALUATION_DIGEST,
        passed=False,
        checks=(
            GateCheck(name="positive_holdout_return", passed=True, mandatory=True),
            GateCheck(name="positive_return_significant", passed=False, mandatory=True),
        ),
        decided_at=NOW,
    )

    with pytest.raises(ValueError, match="mandatory gate"):
        ReleaseManifest.create(
            version="2026.07.13",
            git_commit="e" * 40,
            dataset=dataset(),
            signal=rejected_signal(),
            selection=baseline_selection(),
            gate=gate,
            bundle_digest="9" * 64,
            created_at=NOW,
        )


def test_release_rejects_gate_dataset_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="gate dataset identity"):
        ReleaseManifest.create(
            version="2026.07.13",
            git_commit="e" * 40,
            dataset=dataset(),
            signal=rejected_signal(),
            selection=baseline_selection(),
            gate=passing_gate(dataset_id="0" * 64),
            bundle_digest="9" * 64,
            created_at=NOW,
        )


def test_release_rejects_gate_selected_policy_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="gate selected policy identity"):
        ReleaseManifest.create(
            version="2026.07.13",
            git_commit="e" * 40,
            dataset=dataset(),
            signal=rejected_signal(),
            selection=residual_selection(),
            gate=passing_gate(selected_policy_digest="1" * 64),
            bundle_digest="9" * 64,
            created_at=NOW,
        )


def test_release_records_final_gate_evaluation_identity() -> None:
    release = ReleaseManifest.create(
        version="2026.07.13",
        git_commit="e" * 40,
        dataset=dataset(),
        signal=rejected_signal(),
        selection=residual_selection(),
        gate=passing_gate(selected_policy_digest=POLICY_DIGEST),
        bundle_digest="9" * 64,
        created_at=NOW,
    )

    assert release.selection_evaluation_digest == EVALUATION_DIGEST
    assert release.gate_evaluation_digest == GATE_EVALUATION_DIGEST
