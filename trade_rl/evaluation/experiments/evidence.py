"""Study-owned multi-seed candidate evidence sets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments.contracts import ResolvedRunConfig, StudyPlan
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs.artifact import (
    LoadedCandidateRun,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
)
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    ResolvedCandidateRunSpec,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import execute_candidate_run
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance

_EVIDENCE_SCHEMA = "controlled_evidence_set_v1"
_ANALYSIS_SCHEMA = "controlled_evidence_seed_invariance_v1"
_DETERMINISTIC_STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
)
_EXPECTED_STRATEGIES = frozenset((*_DETERMINISTIC_STRATEGIES, "ppo"))


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Immutable identity of one Study-owned multi-seed evidence point."""

    fingerprint: str
    study_digest: str
    research_context_digest: str
    semantic_config_digest: str
    ppo_seeds: tuple[int, ...]
    run_digests: tuple[tuple[int, str], ...]
    analysis_digest: str
    schema_version: str = _EVIDENCE_SCHEMA


@dataclass(frozen=True, slots=True)
class LoadedEvidenceSet:
    """Verified EvidenceSet plus its run payloads."""

    evidence: EvidenceSet
    resolved_config: dict[str, object]
    runs: Mapping[int, LoadedCandidateRun]
    analysis: dict[str, object]


def _resolved_run_config(spec: ResolvedCandidateRunSpec) -> ResolvedRunConfig:
    config = spec.config
    lean = spec.lean_config
    return ResolvedRunConfig(
        signal_name=config.signal_name,
        signal_index=lean.signal_index,
        feature_names=config.feature_names,
        feature_indices=lean.feature_indices,
        fit_symbol_names=config.fit_symbol_names,
        fit_symbol_indices=lean.fit_symbol_indices,
        fit_cutoff=str(lean.fit_cutoff),
        rule_entry_threshold=lean.rule_entry_threshold,
        rule_exit_threshold=lean.rule_exit_threshold,
        forecast_entry_threshold=lean.forecast_entry_threshold,
        forecast_exit_threshold=lean.forecast_exit_threshold,
        ppo_total_timesteps=lean.ppo_total_timesteps,
        ppo_seed=lean.ppo_seed,
        evaluation_start=str(config.evaluation_start),
        evaluation_stop_exclusive=str(config.evaluation_stop_exclusive),
        gross_budget=config.gross_budget,
        initial_capital=config.initial_capital,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )


def _check_study_fixed_config(plan: StudyPlan, resolved: ResolvedRunConfig) -> None:
    baseline = plan.baseline_config
    fixed_pairs = (
        ("fit_cutoff", resolved.fit_cutoff, baseline.fit_cutoff),
        ("evaluation_start", resolved.evaluation_start, baseline.evaluation_start),
        (
            "evaluation_stop_exclusive",
            resolved.evaluation_stop_exclusive,
            baseline.evaluation_stop_exclusive,
        ),
        ("initial_capital", resolved.initial_capital, baseline.initial_capital),
        ("execution_overlay", resolved.execution_overlay, baseline.execution_overlay),
    )
    for field, actual, expected in fixed_pairs:
        if actual != expected:
            raise ArtifactIntegrityError(f"Study-fixed {field} drifted")


def _check_dataset(plan: StudyPlan, dataset_root: str | Path):
    artifact = inspect_published_market_dataset_artifact(dataset_root)
    dataset = load_market_dataset_artifact(dataset_root)
    if dataset.dataset_id != plan.dataset_id:
        raise ArtifactIntegrityError("Study dataset identity mismatch")
    if artifact.schema_version != plan.dataset_artifact_schema:
        raise ArtifactIntegrityError("Study dataset artifact schema mismatch")
    if artifact.artifact_digest != plan.dataset_artifact_digest:
        raise ArtifactIntegrityError("Study dataset artifact digest mismatch")
    if tuple(dataset.symbols) != plan.symbols:
        raise ArtifactIntegrityError("Study dataset symbol roster mismatch")
    return artifact, dataset


def _check_current_provenance(
    plan: StudyPlan,
    *,
    research_context_digest: str,
) -> dict[str, object]:
    require_sha256(research_context_digest, field="research_context_digest")
    provenance = build_candidate_run_provenance(
        research_context_digest=research_context_digest
    )
    if provenance.get("implementation_digest") != plan.implementation_digest:
        raise ArtifactIntegrityError("Study implementation provenance mismatch")
    if provenance.get("runtime_environment_digest") != plan.runtime_environment_digest:
        raise ArtifactIntegrityError("Study runtime provenance mismatch")
    return provenance


def _run_return_map(
    run: LoadedCandidateRun,
    *,
    expected_symbols: tuple[str, ...],
) -> dict[tuple[str, str], np.ndarray]:
    symbols = run.summary.get("symbols")
    if symbols != list(expected_symbols):
        raise ArtifactIntegrityError("EvidenceSet symbol roster mismatch")
    by_symbol = run.summary.get("by_symbol")
    if not isinstance(by_symbol, list) or len(by_symbol) != len(expected_symbols):
        raise ArtifactIntegrityError("EvidenceSet symbol results are incomplete")

    result: dict[tuple[str, str], np.ndarray] = {}
    for expected_symbol, entry in zip(expected_symbols, by_symbol, strict=True):
        if not isinstance(entry, dict) or entry.get("symbol") != expected_symbol:
            raise ArtifactIntegrityError("EvidenceSet symbol result ordering mismatch")
        strategies = entry.get("strategies")
        if not isinstance(strategies, list):
            raise ArtifactIntegrityError("EvidenceSet strategy results are malformed")
        names: set[str] = set()
        for strategy in strategies:
            if not isinstance(strategy, dict):
                raise ArtifactIntegrityError("EvidenceSet strategy entry is malformed")
            name = strategy.get("name")
            return_key = strategy.get("return_key")
            if not isinstance(name, str) or not isinstance(return_key, str):
                raise ArtifactIntegrityError("EvidenceSet strategy evidence is malformed")
            if name in names:
                raise ArtifactIntegrityError("EvidenceSet strategy roster has duplicates")
            names.add(name)
            if return_key not in run.returns:
                raise ArtifactIntegrityError("EvidenceSet return key is missing")
            result[(expected_symbol, name)] = run.returns[return_key]
        if names != _EXPECTED_STRATEGIES:
            raise ArtifactIntegrityError("EvidenceSet strategy roster mismatch")
    return result


def _verify_seed_invariance(
    runs: Mapping[int, LoadedCandidateRun],
    *,
    seeds: tuple[int, ...],
    symbols: tuple[str, ...],
    research_context_digest: str,
) -> dict[str, object]:
    first_seed = seeds[0]
    first = runs[first_seed]
    first_map = _run_return_map(first, expected_symbols=symbols)
    for seed in seeds:
        run = runs[seed]
        if run.provenance.get("research_context_digest") != research_context_digest:
            raise ArtifactIntegrityError("EvidenceSet research context mismatch")
        candidate_config = run.summary.get("candidate_config")
        if not isinstance(candidate_config, dict) or candidate_config.get("ppo_seed") != seed:
            raise ArtifactIntegrityError("EvidenceSet PPO seed evidence mismatch")
        current_map = _run_return_map(run, expected_symbols=symbols)
        if seed == first_seed:
            continue
        for symbol in symbols:
            for strategy in _DETERMINISTIC_STRATEGIES:
                if not np.array_equal(
                    first_map[(symbol, strategy)],
                    current_map[(symbol, strategy)],
                ):
                    raise ArtifactIntegrityError(
                        f"deterministic strategy {strategy} drifted across PPO seeds"
                    )
    return {
        "schema_version": _ANALYSIS_SCHEMA,
        "status": "VERIFIED",
        "ppo_seeds": list(seeds),
        "symbols": list(symbols),
        "deterministic_strategy_names": list(_DETERMINISTIC_STRATEGIES),
        "variable_strategy_name": "ppo",
    }


def _identity_payload(
    *,
    study_digest: str,
    research_context_digest: str,
    semantic_config: ResolvedRunConfig,
    seeds: tuple[int, ...],
    run_digests: tuple[tuple[int, str], ...],
    analysis_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": _EVIDENCE_SCHEMA,
        "study_digest": study_digest,
        "research_context_digest": research_context_digest,
        "semantic_config": semantic_config.to_payload(),
        "semantic_config_digest": semantic_config.digest,
        "ppo_seeds": list(seeds),
        "runs": [
            {"ppo_seed": seed, "artifact_digest": digest}
            for seed, digest in run_digests
        ],
        "analysis_digest": analysis_digest,
    }


def _manifest_payload(evidence: EvidenceSet, resolved_config: ResolvedRunConfig) -> dict[str, object]:
    payload = _identity_payload(
        study_digest=evidence.study_digest,
        research_context_digest=evidence.research_context_digest,
        semantic_config=resolved_config,
        seeds=evidence.ppo_seeds,
        run_digests=evidence.run_digests,
        analysis_digest=evidence.analysis_digest,
    )
    return {**payload, "fingerprint": evidence.fingerprint}


def execute_evidence_set(
    *,
    store: StudyStore,
    target: Path,
    dataset_root: str | Path,
    plan: StudyPlan,
    config: CandidateRunConfig,
    research_context_digest: str,
) -> EvidenceSet:
    """Generate, verify, and atomically publish one Study-owned EvidenceSet."""

    artifact, dataset = _check_dataset(plan, dataset_root)
    _check_current_provenance(
        plan,
        research_context_digest=research_context_digest,
    )
    completed: EvidenceSet | None = None

    def builder(staging: Path) -> None:
        nonlocal completed
        loaded_runs: dict[int, LoadedCandidateRun] = {}
        run_digests: list[tuple[int, str]] = []
        semantic_config: ResolvedRunConfig | None = None

        for seed in plan.ppo_seeds:
            seed_config = replace(config, ppo_seed=seed)
            spec = resolve_candidate_run_spec(
                dataset,
                dataset_artifact_schema=artifact.schema_version,
                dataset_artifact_digest=artifact.artifact_digest,
                config=seed_config,
            )
            resolved = _resolved_run_config(spec)
            _check_study_fixed_config(plan, resolved)
            normalized = replace(resolved, ppo_seed=plan.ppo_seeds[0])
            if semantic_config is None:
                semantic_config = normalized
            elif normalized != semantic_config:
                raise ArtifactIntegrityError(
                    "EvidenceSet resolved config changed beyond ppo_seed"
                )

            before = _check_current_provenance(
                plan,
                research_context_digest=research_context_digest,
            )
            result = execute_candidate_run(dataset, spec)
            after = _check_current_provenance(
                plan,
                research_context_digest=research_context_digest,
            )
            if (
                before.get("implementation_digest"),
                before.get("runtime_environment_digest"),
            ) != (
                after.get("implementation_digest"),
                after.get("runtime_environment_digest"),
            ):
                raise ArtifactIntegrityError("Run provenance changed during execution")

            run_root = staging / "runs" / f"seed-{seed}"
            publish_candidate_run(run_root, result, after)
            loaded = load_candidate_run_artifact(run_root)
            identity = inspect_candidate_run_artifact(run_root)
            loaded_runs[seed] = loaded
            run_digests.append((seed, identity.artifact_digest))

        if semantic_config is None:
            raise ArtifactIntegrityError("EvidenceSet contains no seed Runs")
        analysis = _verify_seed_invariance(
            loaded_runs,
            seeds=plan.ppo_seeds,
            symbols=plan.symbols,
            research_context_digest=research_context_digest,
        )
        analysis_digest = content_digest(analysis)
        run_digest_tuple = tuple(run_digests)
        identity_payload = _identity_payload(
            study_digest=plan.digest,
            research_context_digest=research_context_digest,
            semantic_config=semantic_config,
            seeds=plan.ppo_seeds,
            run_digests=run_digest_tuple,
            analysis_digest=analysis_digest,
        )
        evidence = EvidenceSet(
            fingerprint=content_digest(identity_payload),
            study_digest=plan.digest,
            research_context_digest=research_context_digest,
            semantic_config_digest=semantic_config.digest,
            ppo_seeds=plan.ppo_seeds,
            run_digests=run_digest_tuple,
            analysis_digest=analysis_digest,
        )
        (staging / "analysis.json").write_bytes(canonical_json_bytes(analysis))
        (staging / "manifest.json").write_bytes(
            canonical_json_bytes(_manifest_payload(evidence, semantic_config))
        )
        completed = evidence

    with store.mutation_lock():
        store.publish_directory_once(target, builder)
    if completed is None:
        raise ArtifactIntegrityError("EvidenceSet publication did not complete")
    return completed


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError(f"EvidenceSet {label} must be a regular file")
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"EvidenceSet {label} is malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ArtifactIntegrityError(f"EvidenceSet {label} must be a JSON object")
    return cast(dict[str, object], raw)


def _resolved_from_payload(payload: object) -> ResolvedRunConfig:
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError("EvidenceSet semantic config is malformed")
    try:
        return ResolvedRunConfig(
            signal_name=cast(str, payload["signal_name"]),
            signal_index=cast(int, payload["signal_index"]),
            feature_names=tuple(cast(list[str], payload["feature_names"])),
            feature_indices=tuple(cast(list[int], payload["feature_indices"])),
            fit_symbol_names=tuple(cast(list[str], payload["fit_symbol_names"])),
            fit_symbol_indices=tuple(cast(list[int], payload["fit_symbol_indices"])),
            fit_cutoff=cast(str, payload["fit_cutoff"]),
            rule_entry_threshold=cast(float, payload["rule_entry_threshold"]),
            rule_exit_threshold=cast(float, payload["rule_exit_threshold"]),
            forecast_entry_threshold=cast(float, payload["forecast_entry_threshold"]),
            forecast_exit_threshold=cast(float, payload["forecast_exit_threshold"]),
            ppo_total_timesteps=cast(int, payload["ppo_total_timesteps"]),
            ppo_seed=cast(int, payload["ppo_seed"]),
            evaluation_start=cast(str, payload["evaluation_start"]),
            evaluation_stop_exclusive=cast(str, payload["evaluation_stop_exclusive"]),
            gross_budget=cast(float, payload["gross_budget"]),
            initial_capital=cast(float, payload["initial_capital"]),
            execution_overlay=cast(str, payload["execution_overlay"]),
            schema_version=cast(str, payload.get("schema_version", "resolved_run_config_v1")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("EvidenceSet semantic config is malformed") from error


def load_evidence_set(root: str | Path) -> LoadedEvidenceSet:
    """Load and re-verify one published EvidenceSet from disk."""

    evidence_root = Path(root)
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ArtifactIntegrityError("EvidenceSet root must be a regular directory")
    manifest = _read_json_object(evidence_root / "manifest.json", label="manifest")
    analysis = _read_json_object(evidence_root / "analysis.json", label="analysis")
    if manifest.get("schema_version") != _EVIDENCE_SCHEMA:
        raise ArtifactIntegrityError("unsupported EvidenceSet schema")
    if analysis.get("schema_version") != _ANALYSIS_SCHEMA:
        raise ArtifactIntegrityError("unsupported EvidenceSet analysis schema")
    if content_digest(analysis) != manifest.get("analysis_digest"):
        raise ArtifactIntegrityError("EvidenceSet analysis digest mismatch")

    resolved = _resolved_from_payload(manifest.get("semantic_config"))
    if resolved.digest != manifest.get("semantic_config_digest"):
        raise ArtifactIntegrityError("EvidenceSet semantic config digest mismatch")
    try:
        seeds = tuple(cast(list[int], manifest["ppo_seeds"]))
        run_items = cast(list[dict[str, object]], manifest["runs"])
        run_digests = tuple(
            (cast(int, item["ppo_seed"]), cast(str, item["artifact_digest"]))
            for item in run_items
        )
        study_digest = cast(str, manifest["study_digest"])
        context = cast(str, manifest["research_context_digest"])
        fingerprint = cast(str, manifest["fingerprint"])
        analysis_digest = cast(str, manifest["analysis_digest"])
    except (KeyError, TypeError) as error:
        raise ArtifactIntegrityError("EvidenceSet manifest is malformed") from error
    require_sha256(study_digest, field="study_digest")
    require_sha256(context, field="research_context_digest")
    require_sha256(fingerprint, field="fingerprint")
    require_sha256(analysis_digest, field="analysis_digest")
    if tuple(seed for seed, _ in run_digests) != seeds:
        raise ArtifactIntegrityError("EvidenceSet run seed roster mismatch")

    runs: dict[int, LoadedCandidateRun] = {}
    for seed, expected_digest in run_digests:
        require_sha256(expected_digest, field="run artifact digest")
        run_root = evidence_root / "runs" / f"seed-{seed}"
        identity = inspect_candidate_run_artifact(run_root)
        if identity.artifact_digest != expected_digest:
            raise ArtifactIntegrityError("EvidenceSet Run artifact digest mismatch")
        runs[seed] = load_candidate_run_artifact(run_root)

    _verify_seed_invariance(
        runs,
        seeds=seeds,
        symbols=tuple(cast(list[str], analysis.get("symbols", []))),
        research_context_digest=context,
    )
    identity_payload = _identity_payload(
        study_digest=study_digest,
        research_context_digest=context,
        semantic_config=resolved,
        seeds=seeds,
        run_digests=run_digests,
        analysis_digest=analysis_digest,
    )
    if content_digest(identity_payload) != fingerprint:
        raise ArtifactIntegrityError("EvidenceSet fingerprint mismatch")

    evidence = EvidenceSet(
        fingerprint=fingerprint,
        study_digest=study_digest,
        research_context_digest=context,
        semantic_config_digest=resolved.digest,
        ppo_seeds=seeds,
        run_digests=run_digests,
        analysis_digest=analysis_digest,
    )
    return LoadedEvidenceSet(
        evidence=evidence,
        resolved_config=resolved.to_payload(),
        runs=runs,
        analysis=analysis,
    )


__all__ = [
    "EvidenceSet",
    "LoadedEvidenceSet",
    "execute_evidence_set",
    "load_evidence_set",
]
