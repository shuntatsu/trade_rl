"""Study-owned multi-seed candidate evidence sets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.data import (
    MarketDataset,
    PublishedDatasetArtifact,
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
    """Raw Study-owned multi-seed evidence identity."""

    fingerprint: str
    semantic_config_digest: str
    ppo_seeds: tuple[int, ...]
    run_digests: tuple[tuple[int, str], ...]
    research_context_digest: str
    schema_version: str = _EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        try:
            require_sha256(self.fingerprint, field="fingerprint")
            require_sha256(
                self.semantic_config_digest,
                field="semantic_config_digest",
            )
            require_sha256(
                self.research_context_digest,
                field="research_context_digest",
            )
        except ValueError as error:
            raise ArtifactIntegrityError(str(error)) from error
        if self.schema_version != _EVIDENCE_SCHEMA:
            raise ArtifactIntegrityError("unsupported EvidenceSet schema")
        if len(self.ppo_seeds) < 2 or len(set(self.ppo_seeds)) != len(self.ppo_seeds):
            raise ArtifactIntegrityError("EvidenceSet seed roster must be unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.ppo_seeds
        ):
            raise ArtifactIntegrityError("EvidenceSet seeds must be non-negative integers")
        if tuple(seed for seed, _ in self.run_digests) != self.ppo_seeds:
            raise ArtifactIntegrityError(
                "EvidenceSet run digests must follow the seed roster"
            )
        for _, digest in self.run_digests:
            try:
                require_sha256(digest, field="run artifact digest")
            except ValueError as error:
                raise ArtifactIntegrityError(str(error)) from error
        expected = _evidence_fingerprint(
            semantic_config_digest=self.semantic_config_digest,
            ppo_seeds=self.ppo_seeds,
            run_digests=self.run_digests,
            research_context_digest=self.research_context_digest,
        )
        if self.fingerprint != expected:
            raise ArtifactIntegrityError("EvidenceSet fingerprint mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "semantic_config_digest": self.semantic_config_digest,
            "ppo_seeds": list(self.ppo_seeds),
            "run_digests": [
                {"ppo_seed": seed, "artifact_digest": digest}
                for seed, digest in self.run_digests
            ],
            "research_context_digest": self.research_context_digest,
        }


@dataclass(frozen=True, slots=True)
class LoadedEvidenceSet:
    """Verified EvidenceSet plus its Study-owned Candidate Runs."""

    evidence: EvidenceSet
    semantic_config: dict[str, object]
    runs: Mapping[int, LoadedCandidateRun]


def _evidence_fingerprint(
    *,
    semantic_config_digest: str,
    ppo_seeds: tuple[int, ...],
    run_digests: tuple[tuple[int, str], ...],
    research_context_digest: str,
) -> str:
    return content_digest(
        {
            "schema_version": _EVIDENCE_SCHEMA,
            "semantic_config_digest": semantic_config_digest,
            "ppo_seeds": list(ppo_seeds),
            "run_digests": [
                {"ppo_seed": seed, "artifact_digest": digest}
                for seed, digest in run_digests
            ],
            "research_context_digest": research_context_digest,
        }
    )


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


def _without_seed(config: ResolvedRunConfig) -> dict[str, object]:
    payload = config.to_payload()
    payload.pop("ppo_seed")
    return payload


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


def _verify_plan_inputs(
    *,
    dataset_root: str | Path,
    plan: StudyPlan,
    research_context_digest: str,
) -> tuple[PublishedDatasetArtifact, MarketDataset]:
    try:
        require_sha256(
            research_context_digest,
            field="research_context_digest",
        )
    except ValueError as error:
        raise ArtifactIntegrityError(str(error)) from error

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

    provenance = build_candidate_run_provenance(
        research_context_digest=research_context_digest,
    )
    if provenance.get("implementation_digest") != plan.implementation_digest:
        raise ArtifactIntegrityError("Study implementation provenance mismatch")
    if provenance.get("runtime_environment_digest") != plan.runtime_environment_digest:
        raise ArtifactIntegrityError("Study runtime provenance mismatch")
    return artifact, dataset


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
                raise ArtifactIntegrityError(
                    "EvidenceSet strategy evidence is malformed"
                )
            if name in names:
                raise ArtifactIntegrityError(
                    "EvidenceSet strategy roster has duplicates"
                )
            names.add(name)
            values = run.returns.get(return_key)
            if values is None:
                raise ArtifactIntegrityError("EvidenceSet return key is missing")
            result[(expected_symbol, name)] = values
        if names != _EXPECTED_STRATEGIES:
            raise ArtifactIntegrityError("EvidenceSet strategy roster mismatch")
    return result


def _run_summary_seed(run: LoadedCandidateRun) -> int:
    candidate_config = run.summary.get("candidate_config")
    if not isinstance(candidate_config, dict):
        raise ArtifactIntegrityError("EvidenceSet candidate_config is malformed")
    seed = candidate_config.get("ppo_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactIntegrityError("EvidenceSet PPO seed evidence is malformed")
    return seed


def _verify_seed_invariance(
    runs: Mapping[int, LoadedCandidateRun],
    *,
    seeds: tuple[int, ...],
    symbols: tuple[str, ...],
    research_context_digest: str,
) -> None:
    first_seed = seeds[0]
    first_map = _run_return_map(runs[first_seed], expected_symbols=symbols)
    for seed in seeds:
        run = runs[seed]
        if run.provenance.get("research_context_digest") != research_context_digest:
            raise ArtifactIntegrityError("EvidenceSet research context mismatch")
        if _run_summary_seed(run) != seed:
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


def execute_evidence_set(
    *,
    store: StudyStore,
    target: str | Path,
    dataset_root: str | Path,
    plan: StudyPlan,
    config: CandidateRunConfig,
    research_context_digest: str,
) -> EvidenceSet:
    """Generate, verify, and atomically publish one Study-owned EvidenceSet."""

    artifact, dataset = _verify_plan_inputs(
        dataset_root=dataset_root,
        plan=plan,
        research_context_digest=research_context_digest,
    )
    normalized_config = replace(config, ppo_seed=plan.ppo_seeds[0])
    normalized_spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=normalized_config,
    )
    normalized_contract = _resolved_run_config(normalized_spec)
    _check_study_fixed_config(plan, normalized_contract)
    completed: EvidenceSet | None = None

    def builder(staging: Path) -> None:
        nonlocal completed
        loaded_runs: dict[int, LoadedCandidateRun] = {}
        run_digests: list[tuple[int, str]] = []
        seedless_contract = _without_seed(normalized_contract)

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
            if _without_seed(resolved) != seedless_contract:
                raise ArtifactIntegrityError(
                    "EvidenceSet resolved config changed beyond ppo_seed"
                )

            before = build_candidate_run_provenance(
                research_context_digest=research_context_digest,
            )
            if before.get("implementation_digest") != plan.implementation_digest:
                raise ArtifactIntegrityError(
                    "Study implementation provenance changed during EvidenceSet"
                )
            if before.get("runtime_environment_digest") != plan.runtime_environment_digest:
                raise ArtifactIntegrityError(
                    "Study runtime provenance changed during EvidenceSet"
                )
            result = execute_candidate_run(dataset, spec)
            after = build_candidate_run_provenance(
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

        _verify_seed_invariance(
            loaded_runs,
            seeds=plan.ppo_seeds,
            symbols=plan.symbols,
            research_context_digest=research_context_digest,
        )
        run_digest_tuple = tuple(run_digests)
        fingerprint = _evidence_fingerprint(
            semantic_config_digest=normalized_contract.digest,
            ppo_seeds=plan.ppo_seeds,
            run_digests=run_digest_tuple,
            research_context_digest=research_context_digest,
        )
        evidence = EvidenceSet(
            fingerprint=fingerprint,
            semantic_config_digest=normalized_contract.digest,
            ppo_seeds=plan.ppo_seeds,
            run_digests=run_digest_tuple,
            research_context_digest=research_context_digest,
        )
        manifest = evidence.to_payload()
        manifest["semantic_config"] = normalized_contract.to_payload()
        atomic_write_bytes(staging / "manifest.json", canonical_json_bytes(manifest))
        completed = evidence

    with store.mutation_lock():
        store.publish_directory_once(target, builder)
    if completed is None:
        raise ArtifactIntegrityError("EvidenceSet publication did not complete")
    return completed


def _read_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ArtifactIntegrityError("EvidenceSet manifest must not be a symlink")
    try:
        with open_regular_binary(manifest_path, field="EvidenceSet manifest") as handle:
            payload = handle.read()
        raw = cast(object, json.loads(payload.decode("utf-8")))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("EvidenceSet manifest is malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ArtifactIntegrityError("EvidenceSet manifest must be a JSON object")
    return cast(dict[str, object], raw)


def _parse_evidence(
    manifest: dict[str, object],
) -> tuple[EvidenceSet, dict[str, object]]:
    semantic_config = manifest.get("semantic_config")
    seeds_raw = manifest.get("ppo_seeds")
    run_digests_raw = manifest.get("run_digests")
    if not isinstance(semantic_config, dict) or any(
        not isinstance(key, str) for key in semantic_config
    ):
        raise ArtifactIntegrityError("EvidenceSet semantic_config is malformed")
    if not isinstance(seeds_raw, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_raw
    ):
        raise ArtifactIntegrityError("EvidenceSet seed roster is malformed")
    if not isinstance(run_digests_raw, list):
        raise ArtifactIntegrityError("EvidenceSet run digests are malformed")

    run_digests: list[tuple[int, str]] = []
    for entry in run_digests_raw:
        if not isinstance(entry, dict):
            raise ArtifactIntegrityError("EvidenceSet run digest entry is malformed")
        seed = entry.get("ppo_seed")
        digest = entry.get("artifact_digest")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(digest, str)
        ):
            raise ArtifactIntegrityError("EvidenceSet run digest entry is malformed")
        run_digests.append((seed, digest))

    fingerprint = manifest.get("fingerprint")
    semantic_digest = manifest.get("semantic_config_digest")
    context = manifest.get("research_context_digest")
    schema = manifest.get("schema_version")
    if not all(
        isinstance(value, str)
        for value in (fingerprint, semantic_digest, context, schema)
    ):
        raise ArtifactIntegrityError("EvidenceSet identity fields are malformed")
    if content_digest(semantic_config) != semantic_digest:
        raise ArtifactIntegrityError("EvidenceSet semantic config digest mismatch")

    evidence = EvidenceSet(
        fingerprint=cast(str, fingerprint),
        semantic_config_digest=cast(str, semantic_digest),
        ppo_seeds=tuple(cast(list[int], seeds_raw)),
        run_digests=tuple(run_digests),
        research_context_digest=cast(str, context),
        schema_version=cast(str, schema),
    )
    return evidence, cast(dict[str, object], semantic_config)


def load_evidence_set(root: str | Path) -> LoadedEvidenceSet:
    """Load and re-verify one complete Study-owned EvidenceSet."""

    evidence_root = Path(root)
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ArtifactIntegrityError("EvidenceSet root must be a regular directory")
    names = {entry.name for entry in evidence_root.iterdir()}
    if names != {"manifest.json", "runs"}:
        raise ArtifactIntegrityError(
            "EvidenceSet root must contain exactly manifest.json and runs"
        )

    manifest = _read_manifest(evidence_root)
    evidence, semantic_config = _parse_evidence(manifest)
    runs_root = evidence_root / "runs"
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise ArtifactIntegrityError("EvidenceSet runs must be a regular directory")
    expected_dirs = {f"seed-{seed}" for seed in evidence.ppo_seeds}
    actual_dirs = {entry.name for entry in runs_root.iterdir()}
    if actual_dirs != expected_dirs:
        raise ArtifactIntegrityError("EvidenceSet seed Run roster mismatch")

    expected_digests = dict(evidence.run_digests)
    runs: dict[int, LoadedCandidateRun] = {}
    for seed in evidence.ppo_seeds:
        run_root = runs_root / f"seed-{seed}"
        try:
            loaded = load_candidate_run_artifact(run_root)
            identity = inspect_candidate_run_artifact(run_root)
        except ValueError as error:
            raise ArtifactIntegrityError(str(error)) from error
        if identity.artifact_digest != expected_digests[seed]:
            raise ArtifactIntegrityError("EvidenceSet Run artifact digest mismatch")
        if loaded.provenance.get("research_context_digest") != evidence.research_context_digest:
            raise ArtifactIntegrityError("EvidenceSet Run research context mismatch")
        if _run_summary_seed(loaded) != seed:
            raise ArtifactIntegrityError("EvidenceSet Run seed mismatch")
        runs[seed] = loaded

    _verify_seed_invariance(
        runs,
        seeds=evidence.ppo_seeds,
        symbols=tuple(
            cast(list[str], runs[evidence.ppo_seeds[0]].summary.get("symbols"))
        ),
        research_context_digest=evidence.research_context_digest,
    )
    return LoadedEvidenceSet(
        evidence=evidence,
        semantic_config=semantic_config,
        runs=runs,
    )


__all__ = [
    "EvidenceSet",
    "LoadedEvidenceSet",
    "execute_evidence_set",
    "load_evidence_set",
]
