from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherHoldoutMetric,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateConfig,
    CausalAlphaV3CandidateEvidence,
    CausalAlphaV3EpisodeMetric,
    CausalAlphaV3SelectionEvidence,
    CausalAlphaV3TeacherAdmissionEvidence,
    UniversalCausalAlphaV3TeacherPackage,
)


def _candidate(name: str = "baseline") -> CausalAlphaV3CandidateConfig:
    return CausalAlphaV3CandidateConfig(
        name=name,
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.1, 0.2),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.0,
            edge_margin=0.0,
            alpha_rebalance_decisions=16,
            strong_reversal_threshold=2.0,
            max_target_delta=0.1,
        ),
    )


def _target_digest(symbol: str, values: np.ndarray) -> str:
    return content_and_arrays_digest(
        {"schema_version": "causal_alpha_v3_target_weights_v1", "symbol": symbol},
        (("target_weights", values),),
    )


def _metric(
    candidate: CausalAlphaV3CandidateConfig, episode: int = 0
) -> CausalAlphaV3EpisodeMetric:
    return CausalAlphaV3EpisodeMetric(
        candidate_digest=candidate.digest,
        symbol="BTCUSDT",
        episode_index=episode,
        contract_digest=content_digest(f"contract:{episode}"),
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.4,
        total_execution_cost=10.0,
        trade_count=8,
        hard_risk_violation=False,
        unexplained_execution_rejection_count=0,
    )


def _selection() -> CausalAlphaV3SelectionEvidence:
    candidate = _candidate()
    metric = _metric(candidate)
    evidence = CausalAlphaV3CandidateEvidence.from_episode_metrics(
        candidate=candidate,
        episode_metrics=(metric,),
        admissible=True,
        rejection_reasons=(),
    )
    return CausalAlphaV3SelectionEvidence(
        candidates=(evidence,),
        selected_candidate_digest=candidate.digest,
        grid_digest=content_digest("grid"),
        thresholds_digest=content_digest("thresholds"),
        generator_code_digest=content_digest("generator"),
        sample_scope_digest=content_digest("samples"),
        holdout_episode_digests={"BTCUSDT": content_digest("holdout")},
    )


def _admission(
    selection: CausalAlphaV3SelectionEvidence,
) -> CausalAlphaV3TeacherAdmissionEvidence:
    metric = CausalAlphaTeacherHoldoutMetric(
        symbol="BTCUSDT",
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.2,
        total_execution_cost=4.0,
        trade_count=4,
        maximum_drawdown=0.01,
    )
    return CausalAlphaV3TeacherAdmissionEvidence(
        selection_digest=selection.digest,
        selected_candidate_digest=selection.selected_candidate_digest,
        holdout_episode_digests=selection.holdout_episode_digests,
        admission=evaluate_causal_alpha_teacher_admission((metric,)),
    )


def test_v3_contracts_close_digests_and_never_enable_promotion() -> None:
    selection = _selection()

    assert len(selection.digest) == 64
    assert selection.promotion_eligible is False
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(selection, thresholds_digest=content_digest("changed"))


def test_v3_candidate_rejects_duplicate_episode_scope() -> None:
    candidate = _candidate()
    metric = _metric(candidate)

    with pytest.raises(ValueError, match="duplicated"):
        CausalAlphaV3CandidateEvidence.from_episode_metrics(
            candidate=candidate,
            episode_metrics=(metric, metric),
            admissible=True,
            rejection_reasons=(),
        )


def test_v3_package_owns_read_only_targets() -> None:
    selection = _selection()
    admission = _admission(selection)
    source = np.array([0.0, 0.1, -0.1], dtype=np.float32)
    package = UniversalCausalAlphaV3TeacherPackage(
        selection=selection,
        teacher_admission=admission,
        target_weights={"BTCUSDT": source},
        target_digests={"BTCUSDT": _target_digest("BTCUSDT", source)},
        teacher_config_digest=content_digest("teacher"),
    )
    source[0] = 1.0

    assert package.target_weights["BTCUSDT"][0] == 0.0
    assert not package.target_weights["BTCUSDT"].flags.writeable
    with pytest.raises(ValueError):
        package.target_weights["BTCUSDT"][0] = 1.0


def test_v3_package_rejects_admission_identity_drift() -> None:
    selection = _selection()
    admission = _admission(selection)
    drifted = replace(admission, selection_digest=content_digest("other"), digest="")

    with pytest.raises(ValueError, match="selection identity drifted"):
        UniversalCausalAlphaV3TeacherPackage(
            selection=selection,
            teacher_admission=drifted,
            target_weights={"BTCUSDT": np.zeros(3)},
            target_digests={"BTCUSDT": _target_digest("BTCUSDT", np.zeros(3))},
            teacher_config_digest=content_digest("teacher"),
        )
