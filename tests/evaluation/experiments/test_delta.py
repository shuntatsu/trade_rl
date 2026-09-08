from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts import (
    ControlledFactor,
    ExperimentDefinition,
    ResolvedRunConfig,
    StudyPlan,
)
from trade_rl.evaluation.experiments.delta import (
    FACTOR_RULES,
    ControlledVerificationStatus,
    verify_controlled_delta,
)
from trade_rl.evaluation.experiments.evidence import EvidenceSet, LoadedEvidenceSet
from trade_rl.evaluation.runs.artifact import LoadedCandidateRun

STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "ppo",
)

EXPECTED_RULES = {
    ControlledFactor.FEATURE_SET: (
        frozenset({("feature_names",), ("feature_indices",)}),
        frozenset(
            {"cash", "constant_long", "constant_short", "trend", "mean_reversion"}
        ),
    ),
    ControlledFactor.RULE_SIGNAL: (
        frozenset({("signal_name",), ("signal_index",)}),
        frozenset(
            {
                "cash",
                "constant_long",
                "constant_short",
                "ridge24",
                "lightgbm24",
                "ppo",
            }
        ),
    ),
    ControlledFactor.RULE_THRESHOLDS: (
        frozenset({("rule_entry_threshold",), ("rule_exit_threshold",)}),
        frozenset(
            {
                "cash",
                "constant_long",
                "constant_short",
                "ridge24",
                "lightgbm24",
                "ppo",
            }
        ),
    ),
    ControlledFactor.FORECAST_THRESHOLDS: (
        frozenset(
            {("forecast_entry_threshold",), ("forecast_exit_threshold",)}
        ),
        frozenset(
            {
                "cash",
                "constant_long",
                "constant_short",
                "trend",
                "mean_reversion",
                "ppo",
            }
        ),
    ),
    ControlledFactor.FIT_SYMBOL_SCOPE: (
        frozenset({("fit_symbol_names",), ("fit_symbol_indices",)}),
        frozenset(
            {"cash", "constant_long", "constant_short", "trend", "mean_reversion"}
        ),
    ),
    ControlledFactor.PPO_TRAINING_BUDGET: (
        frozenset({("ppo_total_timesteps",)}),
        frozenset(
            {
                "cash",
                "constant_long",
                "constant_short",
                "trend",
                "mean_reversion",
                "ridge24",
                "lightgbm24",
            }
        ),
    ),
    ControlledFactor.GROSS_BUDGET: (
        frozenset({("gross_budget",)}),
        frozenset({"cash"}),
    ),
}


def _resolved(**overrides: object) -> ResolvedRunConfig:
    values: dict[str, object] = {
        "signal_name": "signal",
        "signal_index": 0,
        "feature_names": ("signal", "volatility"),
        "feature_indices": (0, 1),
        "fit_symbol_names": ("BTCUSDT", "ETHUSDT"),
        "fit_symbol_indices": (0, 1),
        "fit_cutoff": "2026-01-01T00:00:00.000000000",
        "rule_entry_threshold": 0.10,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 2,
        "evaluation_start": "2026-02-01T00:00:00.000000000",
        "evaluation_stop_exclusive": "2026-03-01T00:00:00.000000000",
        "gross_budget": 0.5,
        "initial_capital": 100_000.0,
        "execution_overlay": "zero_overlay_dataset_fields_authoritative",
    }
    values.update(overrides)
    return ResolvedRunConfig(**values)  # type: ignore[arg-type]


def _plan(base: ResolvedRunConfig) -> StudyPlan:
    return StudyPlan(
        research_question="Does one preregistered factor improve development evidence?",
        dataset_id="a" * 64,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        baseline_config=base,
        ppo_seeds=(2, 5),
        allowed_factors=tuple(ControlledFactor),
        max_experiments=8,
        n_bootstrap=100,
        bootstrap_seed=7,
        implementation_digest="c" * 64,
        runtime_environment_digest="d" * 64,
    )


def _semantic(config: ResolvedRunConfig) -> dict[str, object]:
    payload = config.to_payload()
    payload.pop("ppo_seed")
    return payload


def _run(
    *,
    plan: StudyPlan,
    seed: int,
    strategy_drift: str | None = None,
    dataset_id: str | None = None,
    implementation_digest: str | None = None,
    runtime_digest: str | None = None,
    strategy_roster: tuple[str, ...] = STRATEGIES,
) -> LoadedCandidateRun:
    returns: dict[str, np.ndarray] = {}
    by_symbol: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(plan.symbols):
        strategies: list[dict[str, object]] = []
        for strategy_index, name in enumerate(strategy_roster):
            key = f"symbol_{symbol_index}_strategy_{strategy_index}"
            values = np.asarray(
                [0.001 * (strategy_index + 1), -0.0005, 0.0008],
                dtype=np.float64,
            )
            if name == "ppo":
                values = values + seed * 1e-5
            if name == strategy_drift:
                values = values.copy()
                values[0] += 0.01
            returns[key] = values
            strategies.append({"name": name, "return_key": key, "metrics": {}})
        by_symbol.append(
            {
                "symbol_index": symbol_index,
                "symbol": symbol,
                "strategies": strategies,
            }
        )
    return LoadedCandidateRun(
        root=Path(f"/synthetic/seed-{seed}"),
        summary={
            "schema_version": "lean_candidate_result_v1",
            "dataset_id": plan.dataset_id if dataset_id is None else dataset_id,
            "dataset_artifact": {
                "schema_version": plan.dataset_artifact_schema,
                "artifact_digest": plan.dataset_artifact_digest,
            },
            "symbols": list(plan.symbols),
            "by_symbol": by_symbol,
        },
        returns=returns,
        provenance={
            "implementation_digest": (
                plan.implementation_digest
                if implementation_digest is None
                else implementation_digest
            ),
            "runtime_environment_digest": (
                plan.runtime_environment_digest
                if runtime_digest is None
                else runtime_digest
            ),
        },
    )


def _loaded_evidence(
    config: ResolvedRunConfig,
    plan: StudyPlan,
    *,
    strategy_drift: str | None = None,
    seeds: tuple[int, ...] | None = None,
    dataset_id: str | None = None,
    implementation_digest: str | None = None,
    runtime_digest: str | None = None,
    strategy_roster: tuple[str, ...] = STRATEGIES,
) -> LoadedEvidenceSet:
    semantic = _semantic(config)
    resolved_seeds = plan.ppo_seeds if seeds is None else seeds
    run_digests = tuple(
        (seed, content_digest({"seed": seed, "semantic": semantic}))
        for seed in resolved_seeds
    )
    context = content_digest({"study": plan.digest, "semantic": semantic})
    semantic_digest = content_digest(semantic)
    fingerprint = content_digest(
        {
            "schema_version": "controlled_evidence_set_v1",
            "semantic_config_digest": semantic_digest,
            "ppo_seeds": list(resolved_seeds),
            "run_digests": [
                {"ppo_seed": seed, "artifact_digest": digest}
                for seed, digest in run_digests
            ],
            "research_context_digest": context,
        }
    )
    evidence = EvidenceSet(
        fingerprint=fingerprint,
        semantic_config_digest=semantic_digest,
        ppo_seeds=resolved_seeds,
        run_digests=run_digests,
        research_context_digest=context,
    )
    runs = {
        seed: _run(
            plan=plan,
            seed=seed,
            strategy_drift=strategy_drift,
            dataset_id=dataset_id,
            implementation_digest=implementation_digest,
            runtime_digest=runtime_digest,
            strategy_roster=strategy_roster,
        )
        for seed in resolved_seeds
    }
    return LoadedEvidenceSet(evidence=evidence, semantic_config=semantic, runs=runs)


def _candidate_for_factor(
    base: ResolvedRunConfig,
    factor: ControlledFactor,
) -> ResolvedRunConfig:
    if factor is ControlledFactor.FEATURE_SET:
        return replace(
            base,
            feature_names=("signal", "volatility", "momentum"),
            feature_indices=(0, 1, 2),
        )
    if factor is ControlledFactor.RULE_SIGNAL:
        return replace(base, signal_name="momentum", signal_index=2)
    if factor is ControlledFactor.RULE_THRESHOLDS:
        return replace(base, rule_entry_threshold=0.20)
    if factor is ControlledFactor.FORECAST_THRESHOLDS:
        return replace(base, forecast_entry_threshold=0.02)
    if factor is ControlledFactor.FIT_SYMBOL_SCOPE:
        return replace(
            base,
            fit_symbol_names=("BTCUSDT",),
            fit_symbol_indices=(0,),
        )
    if factor is ControlledFactor.PPO_TRAINING_BUDGET:
        return replace(base, ppo_total_timesteps=512)
    if factor is ControlledFactor.GROSS_BUDGET:
        return replace(base, gross_budget=0.8)
    raise AssertionError(f"unsupported factor in test: {factor}")


def _definition(
    *,
    plan: StudyPlan,
    baseline: LoadedEvidenceSet,
    factor: ControlledFactor,
    candidate: ResolvedRunConfig,
) -> ExperimentDefinition:
    return ExperimentDefinition(
        study_digest=plan.digest,
        sequence=1,
        hypothesis=f"Test {factor.value}",
        baseline_evidence_digest=baseline.evidence.fingerprint,
        factor=factor,
        candidate_requested_config_digest=content_digest(candidate.to_payload()),
        candidate_config=candidate,
    )


def test_factor_registry_exactly_matches_approved_spec() -> None:
    assert set(FACTOR_RULES) == set(ControlledFactor)
    for factor, (allowed_paths, unaffected) in EXPECTED_RULES.items():
        rule = FACTOR_RULES[factor]
        assert rule.allowed_paths == allowed_paths
        assert rule.unaffected_strategies == unaffected


@pytest.mark.parametrize("factor", tuple(ControlledFactor))
def test_each_declared_factor_accepts_only_its_registered_delta(
    factor: ControlledFactor,
) -> None:
    base = _resolved()
    plan = _plan(base)
    baseline = _loaded_evidence(base, plan)
    candidate_config = _candidate_for_factor(base, factor)
    candidate = _loaded_evidence(candidate_config, plan)
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=factor,
        candidate=candidate_config,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.CONTROLLED
    assert verification.changed_paths
    assert set(verification.changed_paths) <= FACTOR_RULES[factor].allowed_paths
    assert verification.violations == ()


@pytest.mark.parametrize("factor", tuple(ControlledFactor))
def test_unrelated_second_delta_is_invalid(factor: ControlledFactor) -> None:
    base = _resolved()
    plan = _plan(base)
    baseline = _loaded_evidence(base, plan)
    candidate_config = replace(
        _candidate_for_factor(base, factor),
        initial_capital=base.initial_capital + 1_000.0,
    )
    candidate = _loaded_evidence(candidate_config, plan)
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=factor,
        candidate=candidate_config,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.INVALID
    assert ("initial_capital",) in verification.changed_paths
    assert any("uncontrolled" in item for item in verification.violations)


def test_declared_factor_no_op_is_invalid() -> None:
    base = _resolved()
    plan = _plan(base)
    baseline = _loaded_evidence(base, plan)
    candidate = _loaded_evidence(base, plan)
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate=base,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.INVALID
    assert verification.changed_paths == ()
    assert any("no-op" in item for item in verification.violations)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fit_cutoff", "2026-01-02T00:00:00.000000000"),
        ("evaluation_start", "2026-02-02T00:00:00.000000000"),
        ("evaluation_stop_exclusive", "2026-03-02T00:00:00.000000000"),
        ("initial_capital", 200_000.0),
    ],
)
def test_study_fixed_resolved_fields_cannot_drift(field: str, value: object) -> None:
    base = _resolved()
    plan = _plan(base)
    baseline = _loaded_evidence(base, plan)
    candidate_config = replace(
        _candidate_for_factor(base, ControlledFactor.PPO_TRAINING_BUDGET),
        **{field: value},
    )
    candidate = _loaded_evidence(candidate_config, plan)
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate=candidate_config,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.INVALID
    assert (field,) in verification.changed_paths


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_id": "f" * 64},
        {"implementation_digest": "e" * 64},
        {"runtime_digest": "e" * 64},
        {"seeds": (2, 7)},
        {"strategy_roster": STRATEGIES[:-1]},
    ],
)
def test_study_fixed_evidence_contracts_prevent_controlled_status(
    kwargs: dict[str, object],
) -> None:
    base = _resolved()
    plan = _plan(base)
    baseline = _loaded_evidence(base, plan)
    candidate_config = _candidate_for_factor(
        base,
        ControlledFactor.PPO_TRAINING_BUDGET,
    )
    candidate = _loaded_evidence(candidate_config, plan, **kwargs)  # type: ignore[arg-type]
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate=candidate_config,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.INVALID
    assert verification.violations


@pytest.mark.parametrize("factor", tuple(ControlledFactor))
def test_unaffected_strategy_raw_return_drift_is_invalid(
    factor: ControlledFactor,
) -> None:
    base = _resolved()
    plan = _plan(base)
    baseline = _loaded_evidence(base, plan)
    candidate_config = _candidate_for_factor(base, factor)
    unaffected = sorted(EXPECTED_RULES[factor][1])[0]
    candidate = _loaded_evidence(
        candidate_config,
        plan,
        strategy_drift=unaffected,
    )
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=factor,
        candidate=candidate_config,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.INVALID
    assert any("unaffected strategy" in item for item in verification.violations)


@pytest.mark.parametrize("factor", tuple(ControlledFactor))
def test_affected_strategy_drift_is_not_rejected_by_unaffected_oracle(
    factor: ControlledFactor,
) -> None:
    base = _resolved()
    plan = _plan(base)
    baseline = _loaded_evidence(base, plan)
    candidate_config = _candidate_for_factor(base, factor)
    affected = sorted(set(STRATEGIES) - EXPECTED_RULES[factor][1])[0]
    candidate = _loaded_evidence(candidate_config, plan, strategy_drift=affected)
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=factor,
        candidate=candidate_config,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.CONTROLLED


def test_candidate_evidence_must_match_frozen_definition_exactly() -> None:
    base = _resolved()
    plan = _plan(base)
    baseline = _loaded_evidence(base, plan)
    defined_config = _candidate_for_factor(base, ControlledFactor.RULE_THRESHOLDS)
    substituted_config = replace(defined_config, rule_entry_threshold=0.30)
    candidate = _loaded_evidence(substituted_config, plan)
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=ControlledFactor.RULE_THRESHOLDS,
        candidate=defined_config,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.INVALID
    assert any("frozen definition" in item for item in verification.violations)
