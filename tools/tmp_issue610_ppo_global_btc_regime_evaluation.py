"""Result-blind bridge contracts for Issue #610 development evaluation.

This helper lives outside the ``trade_rl`` package on purpose.  It may
orchestrate the sealed implementation, but must not become part of the
candidate implementation digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import median
from typing import TypeAlias

import numpy as np

from trade_rl.evaluation.experiments.contracts import ResolvedRunConfig, StudyPlan
from trade_rl.strategies.rl.ppo import (
    PPO_GLOBAL_BTC_REGIME_CONTEXT,
    PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
    PPO_OBSERVATION_SCHEMA,
)

ACCEPT_CANDIDATE = "ACCEPT_CANDIDATE"
KEEP_BASELINE = "KEEP_BASELINE"
INVALID = "INVALID"

EXPECTED_SEMANTIC_CHANGED_FIELDS: tuple[tuple[str, ...], ...] = (
    ("ppo_global_context",),
    ("ppo_observation_schema",),
    ("schema_version",),
)

ReturnMatrix: TypeAlias = Mapping[
    int,
    Mapping[tuple[str, str], np.ndarray],
]


def build_candidate_carrier_plan(
    *,
    source_plan: StudyPlan,
    candidate_config: ResolvedRunConfig,
    implementation_digest: str,
    runtime_environment_digest: str,
) -> StudyPlan:
    """Build a separate StudyPlan carrier for the sealed candidate EvidenceSet."""

    if candidate_config.ppo_seed != source_plan.ppo_seeds[0]:
        raise ValueError(
            "candidate carrier baseline seed must match source seed roster"
        )
    return StudyPlan(
        research_question=(
            f"{source_plan.research_question} [Issue 610 candidate execution carrier]"
        ),
        dataset_id=source_plan.dataset_id,
        dataset_artifact_schema=source_plan.dataset_artifact_schema,
        dataset_artifact_digest=source_plan.dataset_artifact_digest,
        symbols=source_plan.symbols,
        baseline_config=candidate_config,
        ppo_seeds=source_plan.ppo_seeds,
        allowed_factors=source_plan.allowed_factors,
        max_experiments=source_plan.max_experiments,
        n_bootstrap=source_plan.n_bootstrap,
        bootstrap_seed=source_plan.bootstrap_seed,
        implementation_digest=implementation_digest,
        runtime_environment_digest=runtime_environment_digest,
        schema_version=source_plan.schema_version,
    )


def _changed_top_level_fields(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (key,)
        for key in sorted(set(baseline) | set(candidate))
        if baseline.get(key) != candidate.get(key)
        or (key in baseline) != (key in candidate)
    )


def validate_controlled_semantic_delta(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Require exactly the preregistered PPO observation-contract change."""

    changed = _changed_top_level_fields(baseline, candidate)
    violations: list[str] = []
    if changed != EXPECTED_SEMANTIC_CHANGED_FIELDS:
        violations.append("semantic changed-field set differs from preregistration")
    if baseline.get("schema_version") != "resolved_run_config_v2":
        violations.append("baseline resolved-run schema is not v2")
    if candidate.get("schema_version") != "resolved_run_config_v3":
        violations.append("candidate resolved-run schema is not v3")
    if baseline.get("ppo_observation_schema") != PPO_OBSERVATION_SCHEMA:
        violations.append("baseline PPO observation schema is not frozen v2")
    if (
        candidate.get("ppo_observation_schema")
        != PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA
    ):
        violations.append("candidate PPO observation schema is not frozen v3")
    if candidate.get("ppo_global_context") != PPO_GLOBAL_BTC_REGIME_CONTEXT:
        violations.append(
            "candidate PPO global context is not frozen global-BTC regime"
        )
    if "ppo_global_context" in baseline:
        violations.append("baseline unexpectedly defines PPO global context")
    return changed, tuple(violations)


def validate_unaffected_raw_returns(
    baseline: ReturnMatrix,
    candidate: ReturnMatrix,
    *,
    seeds: tuple[int, ...],
    symbols: tuple[str, ...],
    unaffected_strategies: tuple[str, ...],
) -> tuple[str, ...]:
    """Require exact raw-return invariance for every unaffected matrix cell."""

    violations: list[str] = []
    for seed in seeds:
        baseline_seed = baseline.get(seed)
        candidate_seed = candidate.get(seed)
        if baseline_seed is None or candidate_seed is None:
            violations.append(f"unaffected return seed missing: seed={seed}")
            continue
        for symbol in symbols:
            for strategy in unaffected_strategies:
                key = (symbol, strategy)
                baseline_values = baseline_seed.get(key)
                candidate_values = candidate_seed.get(key)
                label = f"seed={seed} {symbol}/{strategy}"
                if baseline_values is None or candidate_values is None:
                    violations.append(f"unaffected raw returns missing: {label}")
                    continue
                if (
                    baseline_values.dtype != candidate_values.dtype
                    or baseline_values.shape != candidate_values.shape
                    or not np.array_equal(baseline_values, candidate_values)
                ):
                    violations.append(f"unaffected raw returns changed: {label}")
    return tuple(violations)


def validate_candidate_ppo_returns(
    candidate: ReturnMatrix,
    *,
    seeds: tuple[int, ...],
    symbols: tuple[str, ...],
) -> tuple[str, ...]:
    """Require a complete, non-empty and finite PPO return matrix."""

    violations: list[str] = []
    for seed in seeds:
        candidate_seed = candidate.get(seed)
        if candidate_seed is None:
            violations.append(f"candidate PPO seed missing: seed={seed}")
            continue
        for symbol in symbols:
            values = candidate_seed.get((symbol, "ppo"))
            label = f"seed={seed} {symbol}"
            if values is None:
                violations.append(f"candidate PPO returns missing: {label}")
                continue
            if values.size == 0:
                violations.append(f"candidate PPO returns empty: {label}")
                continue
            if not bool(np.all(np.isfinite(values))):
                violations.append(f"candidate PPO returns non-finite: {label}")
    return tuple(violations)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("comparison mapping is malformed")
    return value


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("comparison number is malformed")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError("comparison number is non-finite")
    return resolved


def evaluate_frozen_gate(
    comparison: Mapping[str, object],
    *,
    seeds: tuple[int, ...],
    symbols: tuple[str, ...],
    no_new_termination: bool,
    unaffected_raw_returns_equal: bool,
    validity_violations: tuple[str, ...],
) -> dict[str, object]:
    """Apply the exact Issue #609 development decision rule, fail-closed."""

    violations = list(validity_violations)
    positive_symbols = 0
    cross_symbol_positive = False
    positive_seed_medians = 0

    try:
        if comparison.get("schema_version") != "controlled_evidence_comparison_v2":
            raise ValueError("comparison schema is not frozen v2")
        if comparison.get("seeds") != list(seeds):
            raise ValueError("comparison seed roster mismatch")

        by_symbol = _mapping(comparison.get("by_symbol"))
        if tuple(by_symbol) != symbols:
            raise ValueError("comparison symbol roster/order mismatch")

        seed_excesses: dict[int, list[float]] = {seed: [] for seed in seeds}
        for symbol in symbols:
            symbol_payload = _mapping(by_symbol.get(symbol))
            strategies = _mapping(symbol_payload.get("strategies"))
            ppo = _mapping(strategies.get("ppo"))
            aggregate = _mapping(ppo.get("seed_aggregate"))
            symbol_median = _finite_number(aggregate.get("median_excess_total_return"))
            positive_symbols += int(symbol_median > 0.0)

            by_seed = _mapping(ppo.get("by_seed"))
            if tuple(by_seed) != tuple(str(seed) for seed in seeds):
                raise ValueError("comparison PPO seed roster/order mismatch")
            for seed in seeds:
                seed_payload = _mapping(by_seed.get(str(seed)))
                seed_excesses[seed].append(
                    _finite_number(seed_payload.get("excess_total_return"))
                )

        cross_symbol = _mapping(comparison.get("cross_symbol"))
        ppo_cross = _mapping(cross_symbol.get("ppo"))
        cross_symbol_positive = (
            _finite_number(ppo_cross.get("median_excess_total_return")) > 0.0
        )
        positive_seed_medians = sum(
            float(median(seed_excesses[seed])) > 0.0 for seed in seeds
        )
    except (KeyError, TypeError, ValueError) as error:
        violations.append(str(error))

    if not unaffected_raw_returns_equal:
        violations.append("unaffected raw-return invariance failed")

    evidence_valid = not violations
    all_symbols_positive = positive_symbols == len(symbols)
    all_seed_medians_positive = positive_seed_medians == len(seeds)
    all_gates_pass = (
        evidence_valid
        and all_symbols_positive
        and cross_symbol_positive
        and all_seed_medians_positive
        and no_new_termination
        and unaffected_raw_returns_equal
    )
    if not evidence_valid:
        decision = INVALID
    elif all_gates_pass:
        decision = ACCEPT_CANDIDATE
    else:
        decision = KEEP_BASELINE

    return {
        "decision": decision,
        "evidence_valid": evidence_valid,
        "validity_violations": violations,
        "positive_factor_effect_symbols": positive_symbols,
        "all_symbols_positive_factor_effect": all_symbols_positive,
        "cross_symbol_median_positive": cross_symbol_positive,
        "positive_cross_symbol_median_seeds": positive_seed_medians,
        "all_seeds_positive_cross_symbol_median": all_seed_medians_positive,
        "no_new_termination": no_new_termination,
        "unaffected_raw_returns_equal": unaffected_raw_returns_equal,
        "all_gates_pass": all_gates_pass,
    }


__all__ = [
    "ACCEPT_CANDIDATE",
    "EXPECTED_SEMANTIC_CHANGED_FIELDS",
    "INVALID",
    "KEEP_BASELINE",
    "build_candidate_carrier_plan",
    "evaluate_frozen_gate",
    "validate_candidate_ppo_returns",
    "validate_controlled_semantic_delta",
    "validate_unaffected_raw_returns",
]
