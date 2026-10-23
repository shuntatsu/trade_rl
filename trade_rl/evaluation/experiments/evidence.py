"""Study-owned multi-seed candidate evidence sets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
from trade_rl.evaluation.runs import (
    CandidateRunConfig,
    LoadedCandidateRun,
    build_candidate_run_provenance,
    execute_candidate_run,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
    resolve_candidate_run_spec,
)

_EVIDENCE_SCHEMA = "controlled_evidence_set_v1"
_RUN_CONFIG_SCHEMA_BY_RESULT_SCHEMA = {
    "lean_candidate_result_v1": "resolved_run_config_v1",
    "lean_candidate_result_v2": "resolved_run_config_v2",
    "lean_candidate_result_v3": "resolved_run_config_v3",
    "lean_candidate_result_v4": "resolved_run_config_v4",
    "lean_candidate_result_v5": "resolved_run_config_v5",
    "lean_candidate_result_v6": "resolved_run_config_v5",
    "lean_candidate_result_v7": "resolved_run_config_v5",
}
_RUN_RESULT_SCHEMA_VERSION = {
    "lean_candidate_result_v1": 1,
    "lean_candidate_result_v2": 2,
    "lean_candidate_result_v3": 3,
    "lean_candidate_result_v4": 4,
    "lean_candidate_result_v5": 5,
    "lean_candidate_result_v6": 6,
    "lean_candidate_result_v7": 7,
}
_VERSIONED_RUN_CONFIG_FIELDS = {
    2: {"summary": frozenset({"ppo_observation"})},
    3: {
        "candidate_config": frozenset(
            {
                "ppo_training_layout",
                "ppo_rollout_steps_per_env",
                "ppo_training_timesteps",
            }
        )
    },
    4: {
        "candidate_config": frozenset(
            {
                "ppo_minimum_hold_bars",
                "ppo_observation_schema",
                "ppo_settle_terminal_position",
                "ppo_training_minimum_hold_suppressed_count",
            }
        ),
        "evaluation": frozenset({"ppo_settle_terminal_position"}),
        "strategy_diagnostics": frozenset(
            {"minimum_hold_suppressed_count", "minimum_hold_unlocked_count"}
        ),
    },
    5: {
        "candidate_config": frozenset({"pretrade_risk_config"}),
        "evaluation": frozenset({"pretrade_risk_config"}),
        "strategy": frozenset(
            {
                "final_portfolio_value",
                "fill_count",
                "final_quantities",
                "active_order_remainders",
                "terminal_order_reasons",
                "terminal_settlement_complete",
                "minimum_hold_audit",
            }
        ),
    },
    6: {"evaluation": frozenset({"expected_periods"})},
    7: {"summary": frozenset({"shared_cash_ppo"})},
}


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
            raise ArtifactIntegrityError(
                "EvidenceSet seeds must be non-negative integers"
            )
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


def _without_seed(config: ResolvedRunConfig) -> dict[str, object]:
    payload = config.to_payload()
    payload.pop("ppo_seed")
    return payload


def _check_study_fixed_config(plan: StudyPlan, resolved: ResolvedRunConfig) -> None:
    baseline = plan.baseline_config
    for field in plan.FIXED_RESOLVED_FIELDS:
        actual = getattr(resolved, field)
        expected = getattr(baseline, field)
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
    if (
        not isinstance(symbols, Sequence)
        or isinstance(symbols, (str, bytes, bytearray))
        or tuple(symbols) != expected_symbols
    ):
        raise ArtifactIntegrityError("EvidenceSet symbol roster mismatch")
    by_symbol = run.summary.get("by_symbol")
    if (
        not isinstance(by_symbol, Sequence)
        or isinstance(by_symbol, (str, bytes, bytearray))
        or len(by_symbol) != len(expected_symbols)
    ):
        raise ArtifactIntegrityError("EvidenceSet symbol results are incomplete")

    result: dict[tuple[str, str], np.ndarray] = {}
    for expected_symbol, entry in zip(expected_symbols, by_symbol, strict=True):
        if not isinstance(entry, Mapping) or entry.get("symbol") != expected_symbol:
            raise ArtifactIntegrityError("EvidenceSet symbol result ordering mismatch")
        strategies = entry.get("strategies")
        if not isinstance(strategies, Sequence) or isinstance(
            strategies, (str, bytes, bytearray)
        ):
            raise ArtifactIntegrityError("EvidenceSet strategy results are malformed")
        names: set[str] = set()
        for strategy in strategies:
            if not isinstance(strategy, Mapping):
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
        if names != frozenset(StudyPlan.STRATEGY_NAMES):
            raise ArtifactIntegrityError("EvidenceSet strategy roster mismatch")
    return result


def _run_summary_seed(run: LoadedCandidateRun) -> int:
    candidate_config = run.summary.get("candidate_config")
    if not isinstance(candidate_config, Mapping):
        raise ArtifactIntegrityError("EvidenceSet candidate_config is malformed")
    seed = candidate_config.get("ppo_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactIntegrityError("EvidenceSet PPO seed evidence is malformed")
    return seed


def _reject_fields_newer_than_result_schema(
    summary: Mapping[str, object],
    *,
    result_schema: str,
) -> None:
    version = _RUN_RESULT_SCHEMA_VERSION[result_schema]
    mappings: dict[str, list[Mapping[str, object]]] = {"summary": [summary]}
    for location in ("candidate_config", "evaluation"):
        value = summary.get(location)
        if isinstance(value, Mapping):
            mappings[location] = [value]
    by_symbol = summary.get("by_symbol")
    if isinstance(by_symbol, Sequence) and not isinstance(
        by_symbol, (str, bytes, bytearray)
    ):
        for symbol in by_symbol:
            if not isinstance(symbol, Mapping):
                continue
            strategies = symbol.get("strategies")
            if not isinstance(strategies, Sequence) or isinstance(
                strategies, (str, bytes, bytearray)
            ):
                continue
            for strategy in strategies:
                if not isinstance(strategy, Mapping):
                    continue
                mappings.setdefault("strategy", []).append(strategy)
                diagnostics = strategy.get("diagnostics")
                if isinstance(diagnostics, Mapping):
                    mappings.setdefault("strategy_diagnostics", []).append(diagnostics)

    for introduced_version, groups in _VERSIONED_RUN_CONFIG_FIELDS.items():
        if version >= introduced_version:
            continue
        for location, fields in groups.items():
            if any(
                fields.intersection(container)
                for container in mappings.get(location, ())
            ):
                raise ArtifactIntegrityError(
                    "EvidenceSet Run fields exceed the declared result schema"
                )


def _config_identity_matches(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    try:
        return canonical_json_bytes(actual) == canonical_json_bytes(expected)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "EvidenceSet config identity is not canonical JSON"
        ) from error


def _run_semantic_config(run: LoadedCandidateRun) -> dict[str, object]:
    """Project the persisted Run configuration into the Study config schema."""

    summary = run.summary
    candidate_config = summary.get("candidate_config")
    evaluation = summary.get("evaluation")
    if not isinstance(candidate_config, Mapping) or not isinstance(evaluation, Mapping):
        raise ArtifactIntegrityError("EvidenceSet Run config is malformed")
    result_schema = summary.get("schema_version")
    config_schema = _RUN_CONFIG_SCHEMA_BY_RESULT_SCHEMA.get(cast(str, result_schema))
    if config_schema is None:
        raise ArtifactIntegrityError("EvidenceSet Run config schema is unsupported")
    _reject_fields_newer_than_result_schema(
        summary,
        result_schema=cast(str, result_schema),
    )

    resolved: dict[str, object] = {"schema_version": config_schema}
    candidate_fields = (
        "signal_name",
        "signal_index",
        "feature_names",
        "feature_indices",
        "fit_symbol_names",
        "fit_symbol_indices",
        "fit_cutoff",
        "rule_entry_threshold",
        "rule_exit_threshold",
        "forecast_entry_threshold",
        "forecast_exit_threshold",
        "ppo_total_timesteps",
    )
    for name in candidate_fields:
        if name not in candidate_config:
            raise ArtifactIntegrityError(f"EvidenceSet Run config is missing {name}")
        value = candidate_config[name]
        resolved[name] = list(value) if isinstance(value, (list, tuple)) else value

    evaluation_fields = {
        "evaluation_start": "start",
        "evaluation_stop_exclusive": "stop_exclusive",
        "gross_budget": "gross_budget",
        "initial_capital": "initial_capital",
        "execution_overlay": "execution_overlay",
    }
    for name, source_name in evaluation_fields.items():
        if source_name not in evaluation:
            raise ArtifactIntegrityError(
                f"EvidenceSet Run evaluation config is missing {source_name}"
            )
        resolved[name] = evaluation[source_name]

    if config_schema != "resolved_run_config_v1":
        observation = summary.get("ppo_observation")
        if not isinstance(observation, Mapping):
            raise ArtifactIntegrityError(
                "EvidenceSet Run observation config is malformed"
            )
        for name, source_name in (
            ("ppo_observation_schema", "schema_version"),
            ("ppo_global_feature_names", "global_feature_names"),
        ):
            if source_name not in observation:
                raise ArtifactIntegrityError(
                    f"EvidenceSet Run observation config is missing {source_name}"
                )
            value = observation[source_name]
            resolved[name] = list(value) if isinstance(value, (list, tuple)) else value

    if config_schema in {
        "resolved_run_config_v3",
        "resolved_run_config_v4",
        "resolved_run_config_v5",
    }:
        for name in ("ppo_training_layout", "ppo_rollout_steps_per_env"):
            if name not in candidate_config:
                raise ArtifactIntegrityError(
                    f"EvidenceSet Run config is missing {name}"
                )
            resolved[name] = candidate_config[name]

    if config_schema in {
        "resolved_run_config_v4",
        "resolved_run_config_v5",
    }:
        for name in ("ppo_minimum_hold_bars", "ppo_settle_terminal_position"):
            if name not in candidate_config:
                raise ArtifactIntegrityError(
                    f"EvidenceSet Run config is missing {name}"
                )
            resolved[name] = candidate_config[name]

    if config_schema == "resolved_run_config_v5":
        if "pretrade_risk_config" not in candidate_config:
            raise ArtifactIntegrityError(
                "EvidenceSet Run config is missing pretrade_risk_config"
            )
        resolved["pretrade_risk_config"] = candidate_config["pretrade_risk_config"]
    return resolved


def _run_dataset_identity(run: LoadedCandidateRun) -> tuple[str, str, str]:
    dataset_id = run.summary.get("dataset_id")
    dataset_artifact = run.summary.get("dataset_artifact")
    if not isinstance(dataset_id, str) or not isinstance(dataset_artifact, Mapping):
        raise ArtifactIntegrityError("EvidenceSet Run dataset identity is malformed")
    schema = dataset_artifact.get("schema_version")
    digest = dataset_artifact.get("artifact_digest")
    if not isinstance(schema, str) or not isinstance(digest, str):
        raise ArtifactIntegrityError("EvidenceSet Run dataset identity is malformed")
    return dataset_id, schema, digest


def _verify_run_binding(
    run: LoadedCandidateRun,
    *,
    semantic_config: Mapping[str, object],
    expected_plan: StudyPlan | None = None,
) -> tuple[str, str, str]:
    if not _config_identity_matches(
        _run_semantic_config(run),
        semantic_config,
    ):
        raise ArtifactIntegrityError("EvidenceSet Run resolved config mismatch")
    dataset_identity = _run_dataset_identity(run)
    if expected_plan is not None and dataset_identity != (
        expected_plan.dataset_id,
        expected_plan.dataset_artifact_schema,
        expected_plan.dataset_artifact_digest,
    ):
        raise ArtifactIntegrityError("EvidenceSet Run dataset identity mismatch")
    return dataset_identity


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
            for strategy in StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES:
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
    completed: EvidenceSet | None = None

    def builder(staging: Path) -> None:
        nonlocal completed
        loaded_runs: dict[int, LoadedCandidateRun] = {}
        run_digests: list[tuple[int, str]] = []
        normalized_contract: ResolvedRunConfig | None = None
        seedless_contract: dict[str, object] | None = None

        for seed in plan.ppo_seeds:
            seed_config = replace(config, ppo_seed=seed)
            spec = resolve_candidate_run_spec(
                dataset,
                dataset_artifact_schema=artifact.schema_version,
                dataset_artifact_digest=artifact.artifact_digest,
                config=seed_config,
                execution_overlay=plan.baseline_config.execution_overlay,
            )
            resolved = ResolvedRunConfig.from_candidate_spec(spec)
            _check_study_fixed_config(plan, resolved)
            if normalized_contract is None:
                normalized_contract = resolved
                seedless_contract = _without_seed(resolved)
            elif seedless_contract is None or not _config_identity_matches(
                _without_seed(resolved), seedless_contract
            ):
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
            if (
                before.get("runtime_environment_digest")
                != plan.runtime_environment_digest
            ):
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
            if not _config_identity_matches(
                _run_semantic_config(loaded),
                _without_seed(resolved),
            ):
                raise ArtifactIntegrityError("EvidenceSet Run resolved config mismatch")
            if _run_dataset_identity(loaded) != (
                plan.dataset_id,
                plan.dataset_artifact_schema,
                plan.dataset_artifact_digest,
            ):
                raise ArtifactIntegrityError(
                    "EvidenceSet Run dataset identity mismatch"
                )
            identity = inspect_candidate_run_artifact(run_root)
            loaded_runs[seed] = loaded
            run_digests.append((seed, identity.artifact_digest))

        if normalized_contract is None or seedless_contract is None:
            raise ArtifactIntegrityError("EvidenceSet contains no seed Runs")
        _verify_seed_invariance(
            loaded_runs,
            seeds=plan.ppo_seeds,
            symbols=plan.symbols,
            research_context_digest=research_context_digest,
        )
        run_digest_tuple = tuple(run_digests)
        semantic_config_digest = content_digest(seedless_contract)
        fingerprint = _evidence_fingerprint(
            semantic_config_digest=semantic_config_digest,
            ppo_seeds=plan.ppo_seeds,
            run_digests=run_digest_tuple,
            research_context_digest=research_context_digest,
        )
        evidence = EvidenceSet(
            fingerprint=fingerprint,
            semantic_config_digest=semantic_config_digest,
            ppo_seeds=plan.ppo_seeds,
            run_digests=run_digest_tuple,
            research_context_digest=research_context_digest,
        )
        manifest = evidence.to_payload()
        manifest["semantic_config"] = seedless_contract
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
    if "ppo_seed" in semantic_config:
        raise ArtifactIntegrityError(
            "EvidenceSet semantic config must not contain ppo_seed"
        )
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


def load_evidence_set(
    root: str | Path,
    *,
    expected_plan: StudyPlan | None = None,
) -> LoadedEvidenceSet:
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
    dataset_identity: tuple[str, str, str] | None = None
    for seed in evidence.ppo_seeds:
        run_root = runs_root / f"seed-{seed}"
        try:
            loaded = load_candidate_run_artifact(run_root)
            identity = inspect_candidate_run_artifact(run_root)
        except ValueError as error:
            raise ArtifactIntegrityError(str(error)) from error
        if identity.artifact_digest != expected_digests[seed]:
            raise ArtifactIntegrityError("EvidenceSet Run artifact digest mismatch")
        if (
            loaded.provenance.get("research_context_digest")
            != evidence.research_context_digest
        ):
            raise ArtifactIntegrityError("EvidenceSet Run research context mismatch")
        if _run_summary_seed(loaded) != seed:
            raise ArtifactIntegrityError("EvidenceSet Run seed mismatch")
        current_dataset_identity = _verify_run_binding(
            loaded,
            semantic_config=semantic_config,
            expected_plan=expected_plan,
        )
        if dataset_identity is None:
            dataset_identity = current_dataset_identity
        elif current_dataset_identity != dataset_identity:
            raise ArtifactIntegrityError("EvidenceSet Run dataset identity mismatch")
        runs[seed] = loaded

    _verify_seed_invariance(
        runs,
        seeds=evidence.ppo_seeds,
        symbols=tuple(
            cast(
                Sequence[str],
                runs[evidence.ppo_seeds[0]].summary.get("symbols"),
            )
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
