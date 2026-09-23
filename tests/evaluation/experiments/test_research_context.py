from __future__ import annotations

import pytest

from tests.evaluation.experiments.test_contracts import resolved_config
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts.experiment import ControlledFactor
from trade_rl.evaluation.experiments.contracts.research import (
    ConsumedEvidence,
    EvidenceKind,
    EvidenceUse,
    StudyResearchContext,
)
from trade_rl.evaluation.experiments.contracts.study import StudyPlan
from trade_rl.evaluation.experiments.errors import ContractViolationError


def _evidence(
    digest: str,
    *,
    start: str = "2023-01-01T00:00:00.000000000",
    stop: str = "2025-01-01T00:00:00.000000000",
    uses: tuple[EvidenceUse, ...] = (
        EvidenceUse.HYPOTHESIS_FORMATION,
        EvidenceUse.OBSERVATION_SELECTION,
    ),
) -> ConsumedEvidence:
    return ConsumedEvidence(
        evidence_kind=EvidenceKind.EXPERIMENT_DECISION,
        evidence_digest=digest,
        development_start=start,
        development_stop_exclusive=stop,
        uses=uses,
    )


def _context(
    *,
    parents: tuple[str, ...] = ("d" * 64, "c" * 64),
    evidence: tuple[ConsumedEvidence, ...] | None = None,
) -> StudyResearchContext:
    return StudyResearchContext(
        parent_context_digests=parents,
        consumed_evidence=(
            evidence
            if evidence is not None
            else (_evidence("b" * 64), _evidence("a" * 64))
        ),
    )


def _study_v3(
    context: StudyResearchContext | None,
    *,
    final_start: str | None = "2026-03-01T00:00:00.000000000",
    final_stop: str | None = "2026-04-01T00:00:00.000000000",
) -> StudyPlan:
    return StudyPlan(
        research_question="Does one controlled observation hypothesis improve robustness?",
        dataset_id="1" * 64,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="2" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        baseline_config=resolved_config(),
        ppo_seeds=(2, 5),
        allowed_factors=(ControlledFactor.FEATURE_SET,),
        max_experiments=4,
        n_bootstrap=1_000,
        bootstrap_seed=7,
        implementation_digest="3" * 64,
        runtime_environment_digest="4" * 64,
        final_evaluation_start=final_start,
        final_evaluation_stop_exclusive=final_stop,
        research_context=context,
        schema_version="controlled_study_plan_v3",
    )


def test_research_context_is_order_independent_and_digest_bound() -> None:
    left = _context()
    right = _context(
        parents=("c" * 64, "d" * 64),
        evidence=(_evidence("a" * 64), _evidence("b" * 64)),
    )

    assert left == right
    assert left.parent_context_digests == ("c" * 64, "d" * 64)
    assert tuple(item.evidence_digest for item in left.consumed_evidence) == (
        "a" * 64,
        "b" * 64,
    )
    assert left.digest == right.digest == content_digest(left.to_payload())


def test_consumed_evidence_requires_canonical_scope_and_non_empty_uses() -> None:
    with pytest.raises(ContractViolationError, match="strictly later|stop"):
        _evidence(
            "a" * 64,
            start="2025-01-01T00:00:00.000000000",
            stop="2025-01-01T00:00:00.000000000",
        )
    with pytest.raises(ContractViolationError, match="canonical|nanosecond"):
        _evidence(
            "a" * 64,
            start="2023-01-01T00:00:00",
        )
    with pytest.raises(ContractViolationError, match="uses"):
        _evidence("a" * 64, uses=())


def test_research_context_rejects_duplicate_parent_or_evidence_identity() -> None:
    with pytest.raises(ContractViolationError, match="parent_context_digests"):
        _context(parents=("c" * 64, "c" * 64))

    with pytest.raises(ContractViolationError, match="consumed_evidence"):
        _context(
            evidence=(
                _evidence("a" * 64),
                _evidence(
                    "a" * 64,
                    uses=(EvidenceUse.RESULT_INTERPRETATION,),
                ),
            )
        )


def test_study_plan_v3_binds_research_context_and_allows_development_only_use() -> None:
    context = _context()
    final_eligible = _study_v3(context)
    development_only = _study_v3(context, final_start=None, final_stop=None)

    assert final_eligible.research_context == context
    assert final_eligible.to_payload()["research_context"] == context.to_payload()
    assert final_eligible.digest == content_digest(final_eligible.to_payload())
    assert development_only.final_evaluation_start is None
    assert development_only.final_evaluation_stop_exclusive is None


def test_study_plan_v3_requires_context_and_rejects_consumed_final_overlap() -> None:
    with pytest.raises(ContractViolationError, match="research_context"):
        _study_v3(None)

    overlapping = StudyResearchContext(
        parent_context_digests=(),
        consumed_evidence=(
            _evidence(
                "a" * 64,
                start="2026-02-15T00:00:00.000000000",
                stop="2026-03-15T00:00:00.000000000",
                uses=(EvidenceUse.RESULT_INTERPRETATION,),
            ),
        ),
    )
    with pytest.raises(ContractViolationError, match="final.*consumed|consumed.*final"):
        _study_v3(overlapping)

    later_evidence = StudyResearchContext(
        parent_context_digests=(),
        consumed_evidence=(
            _evidence(
                "b" * 64,
                start="2026-04-15T00:00:00.000000000",
                stop="2026-05-15T00:00:00.000000000",
                uses=(EvidenceUse.HYPOTHESIS_FORMATION,),
            ),
        ),
    )
    with pytest.raises(ContractViolationError, match="final.*consumed|consumed.*final"):
        _study_v3(later_evidence)


def test_legacy_study_schemas_forbid_research_context() -> None:
    context = _context()
    base = dict(
        research_question="legacy",
        dataset_id="1" * 64,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="2" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        baseline_config=resolved_config(),
        ppo_seeds=(2, 5),
        allowed_factors=(ControlledFactor.FEATURE_SET,),
        max_experiments=4,
        n_bootstrap=1_000,
        bootstrap_seed=7,
        implementation_digest="3" * 64,
        runtime_environment_digest="4" * 64,
        research_context=context,
    )

    with pytest.raises(ContractViolationError, match="v1.*research|research.*v1"):
        StudyPlan(**base)
    with pytest.raises(ContractViolationError, match="v2.*research|research.*v2"):
        StudyPlan(
            **base,
            final_evaluation_start="2026-03-01T00:00:00.000000000",
            final_evaluation_stop_exclusive="2026-04-01T00:00:00.000000000",
            schema_version="controlled_study_plan_v2",
        )
