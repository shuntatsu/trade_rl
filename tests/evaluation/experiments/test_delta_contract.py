from __future__ import annotations

import pytest

from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.experiments.delta import (
    ControlledVerification,
    ControlledVerificationStatus,
)
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError


def test_controlled_verification_requires_non_empty_resolved_delta() -> None:
    with pytest.raises(ArtifactIntegrityError, match="CONTROLLED.*delta"):
        ControlledVerification(
            study_digest="a" * 64,
            experiment_digest="b" * 64,
            baseline_evidence_digest="c" * 64,
            candidate_evidence_digest="d" * 64,
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            status=ControlledVerificationStatus.CONTROLLED,
            changed_paths=(),
            violations=(),
        )
