"""Stage orchestration for the hardened artifact-bound causal alpha V3 workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_v3_admission import (
    CausalAlphaV3AdmissionEvidenceV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
    CausalAlphaV3RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateFreeze,
    CausalAlphaV3SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3RunManifestV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_runtime import (
    CausalAlphaV3PreparedResearchData,
)
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    CausalAlphaV3SelectionRejected,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3NestedPartition,
    CausalAlphaV3SignalGateEvidence,
    split_causal_alpha_v3_partitions,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    CausalAlphaV3FitCache,
    CausalAlphaV3SignalScopeUnavailable,
    build_causal_alpha_v3_episode_batch,
    build_causal_alpha_v3_signal_scope_metric,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher_artifacts import (
    UniversalCausalAlphaV3TeacherPackageV2,
)


class CausalAlphaV3SignalRejected(RuntimeError):
    def __init__(self, fit_results: tuple[Mapping[str, object], ...]) -> None:
        self.fit_results = tuple(dict(item) for item in fit_results)
        self.digest = content_digest(
            {
                "fit_results": self.fit_results,
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_signal_rejection_v2",
            }
        )
        super().__init__("no causal alpha V3 fit cleared the signal gate")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "fit_results": self.fit_results,
            "promotion_eligible": False,
            "schema_version": "causal_alpha_v3_signal_rejection_v2",
        }


class CausalAlphaV3AdmissionRejected(RuntimeError):
    def __init__(
        self, *, admission_digest: str, selected_candidate_digest: str
    ) -> None:
        require_sha256(admission_digest, field="V3 rejected admission digest")
        require_sha256(
            selected_candidate_digest, field="V3 rejected selected candidate"
        )
        self.admission_digest = admission_digest
        self.selected_candidate_digest = selected_candidate_digest
        self.digest = content_digest(
            {
                "admission_digest": admission_digest,
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_admission_rejection_v2",
                "selected_candidate_digest": selected_candidate_digest,
            }
        )
        super().__init__("causal alpha V3 teacher admission failed")

    def to_payload(self) -> dict[str, object]:
        return {
            "admission_digest": self.admission_digest,
            "artifact_digest": self.digest,
            "promotion_eligible": False,
            "schema_version": "causal_alpha_v3_admission_rejection_v2",
            "selected_candidate_digest": self.selected_candidate_digest,
        }


def authored_config_payload(config: CausalAlphaV3ResearchConfig) -> dict[str, object]:
    return {
        "candidates": tuple(
            {
                "fit": {"ridge_strength": candidate.fit.ridge_strength},
                "name": candidate.name,
                "target": {
                    "alpha_rebalance_decisions": candidate.target.alpha_rebalance_decisions,
                    "edge_margin": candidate.target.edge_margin,
                    "execution_cost_multiplier": candidate.target.execution_cost_multiplier,
                    "max_target_delta": candidate.target.max_target_delta,
                    "strong_reversal_threshold": candidate.target.strong_reversal_threshold,
                    "target_magnitudes": candidate.target.target_magnitudes,
                    "uncertainty_multiplier": candidate.target.uncertainty_multiplier,
                },
            }
            for candidate in config.candidates
        ),
        "nested_selection": {
            "minimum_economic_contract_count": (
                config.nested_selection.minimum_economic_contract_count
            ),
            "signal_contract_count": config.nested_selection.signal_contract_count,
        },
        "schema_version": config.schema_version,
        "selection_gate": {
            "maximum_mean_turnover_per_day": (
                config.selection_gate.maximum_mean_turnover_per_day
            ),
            "maximum_unexplained_execution_rejections": (
                config.selection_gate.maximum_unexplained_execution_rejections
            ),
            "minimum_mean_gross_return": config.selection_gate.minimum_mean_gross_return,
            "minimum_mean_net_return": config.selection_gate.minimum_mean_net_return,
            "minimum_positive_gross_episode_fraction": (
                config.selection_gate.minimum_positive_gross_episode_fraction
            ),
            "minimum_symbol_episode_net_return": (
                config.selection_gate.minimum_symbol_episode_net_return
            ),
        },
        "signal_gate": {
            "bootstrap_block_size": config.signal_gate.bootstrap_block_size,
            "bootstrap_resamples": config.signal_gate.bootstrap_resamples,
            "bootstrap_seed": config.signal_gate.bootstrap_seed,
            "minimum_direction_accuracy_excess_lower_ci": (
                config.signal_gate.minimum_direction_accuracy_excess_lower_ci
            ),
            "minimum_rank_ic_lower_ci": config.signal_gate.minimum_rank_ic_lower_ci,
            "minimum_scope_count": config.signal_gate.minimum_scope_count,
            "minimum_scope_coverage": config.signal_gate.minimum_scope_coverage,
            "minimum_top_bottom_spread_lower_ci": (
                config.signal_gate.minimum_top_bottom_spread_lower_ci
            ),
        },
    }


def _nested_digest(
    symbols: tuple[str, ...], nested: Mapping[str, CausalAlphaV3NestedPartition]
) -> str:
    return content_digest(
        {
            "partitions": tuple((symbol, nested[symbol].digest) for symbol in symbols),
            "schema_version": "causal_alpha_v3_nested_scope_v2",
        }
    )


def _resolve_selected(
    candidates: tuple[CausalAlphaV3Candidate, ...],
    selection: CausalAlphaV3SelectionEvidence,
) -> CausalAlphaV3Candidate:
    matches = tuple(
        candidate
        for candidate in candidates
        if candidate.digest == selection.selected_candidate_digest
    )
    if len(matches) != 1:
        raise RuntimeError("V3 selected candidate cannot be resolved")
    return matches[0]


def _build_teacher_batches(
    *,
    selected: CausalAlphaV3Candidate,
    prepared: CausalAlphaV3PreparedResearchData,
    teacher_config_digest: str,
    sampling_config_digest: str,
    fit_cache: CausalAlphaV3FitCache,
    episode_batch_builder: Callable[..., Any],
) -> dict[str, EpisodeOracleBatch]:
    batches: dict[str, EpisodeOracleBatch] = {}
    for symbol in prepared.train_symbols:
        environment = prepared.environment_factories[symbol]()
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("V3 teacher environment must be closable")
        try:
            config = getattr(environment, "config", None)
            execution = getattr(config, "execution_cost", None)
            delay = getattr(config, "signal_delay_decisions", None)
            decision_bars = getattr(environment, "decision_bars", None)
            if not isinstance(execution, ExecutionCostConfig):
                raise TypeError("V3 teacher execution cost is unavailable")
            if asdict(execution) != asdict(prepared.execution_costs[symbol]):
                raise ValueError("V3 teacher execution config drifted")
            if delay != prepared.signal_delays[symbol]:
                raise ValueError("V3 teacher signal delay drifted")
            if decision_bars != prepared.decision_bars[symbol]:
                raise ValueError("V3 teacher decision cadence drifted")
            batch = episode_batch_builder(
                symbol=symbol,
                train_symbols=prepared.train_symbols,
                samples=prepared.samples,
                contracts=prepared.partitions[symbol].contracts,
                candidate=selected,
                dataset=environment.dataset,
                execution_cost=execution,
                signal_delay_decisions=delay,
                decision_bars=decision_bars,
                max_position_to_market_notional=(
                    prepared.max_position_to_market_notional
                ),
                teacher_config_digest=teacher_config_digest,
                sampling_config_digest=sampling_config_digest,
                fit_cache=fit_cache,
            )
            if not isinstance(batch, EpisodeOracleBatch):
                raise TypeError("V3 teacher batch builder returned an invalid batch")
            batches[symbol] = batch
        finally:
            close()
    return batches


def _training_batch_without_holdout(batch: EpisodeOracleBatch) -> EpisodeOracleBatch:
    if len(batch.contracts) < 2 or len(batch.targets) != len(batch.contracts):
        raise ValueError(
            "V3 admitted teacher requires training episodes before holdout"
        )
    return EpisodeOracleBatch(
        dataset_id=batch.dataset_id,
        teacher_config_digest=batch.teacher_config_digest,
        sampling_config_digest=batch.sampling_config_digest,
        contracts=batch.contracts[:-1],
        targets=batch.targets[:-1],
        solver_provenance=None,
    )


def run_universal_causal_alpha_v3_research_pipeline(
    *,
    config: CausalAlphaV3ResearchConfig,
    prepared: CausalAlphaV3PreparedResearchData,
    output_root: Path,
    signal_scope_builder: Callable[
        ..., Any
    ] = build_causal_alpha_v3_signal_scope_metric,
    signal_gate_evaluator: Callable[..., CausalAlphaV3SignalGateEvidence],
    selection_evaluator: Callable[..., CausalAlphaV3SelectionEvidence],
    episode_batch_builder: Callable[..., Any] = build_causal_alpha_v3_episode_batch,
    admission_evaluator: Callable[..., CausalAlphaV3AdmissionEvidenceV2],
) -> UniversalCausalAlphaV3TeacherPackageV2:
    if not isinstance(config, CausalAlphaV3ResearchConfig):
        raise TypeError("V3 research runner requires CausalAlphaV3ResearchConfig")
    if not isinstance(prepared, CausalAlphaV3PreparedResearchData):
        raise TypeError("V3 research runner requires prepared research data")
    symbols = prepared.train_symbols
    root = Path(output_root)

    with CausalAlphaV3RunLock(root):
        nested = split_causal_alpha_v3_partitions(
            prepared.partitions,
            train_symbols=symbols,
            signal_contract_count=config.nested_selection.signal_contract_count,
            minimum_economic_contract_count=(
                config.nested_selection.minimum_economic_contract_count
            ),
        )
        nested_digest = _nested_digest(symbols, nested)
        generator_digest = prepared.generator_code_digest
        execution_identity = prepared.execution_identity
        manifest = CausalAlphaV3RunManifestV2(
            train_symbols=symbols,
            config_digest=config.digest,
            catalog_digest=prepared.catalog_digest,
            partition_digest=prepared.partition_digest,
            split_manifest_digest=prepared.split_manifest_digest,
            feature_schema_digest=prepared.feature_schema_digest,
            statistics_digest=prepared.statistics_digest,
            generator_code_digest=generator_digest,
            nested_partition_digest=nested_digest,
            execution_identity_digest=execution_identity.digest,
            training_contract_digest=execution_identity.training_contract_digest,
            instrument_context_schema_digest=(
                execution_identity.instrument_context_schema_digest
            ),
        )
        base_store = CausalAlphaV3ArtifactStore(
            root, run_manifest_digest=manifest.digest
        )
        base_store.write_exact_artifact(
            "execution-identity.json", execution_identity.to_payload()
        )
        base_store.write_exact_artifact("run-manifest.json", manifest.to_payload())
        base_store.write_exact_artifact(
            "authored-config.json", authored_config_payload(config)
        )

        representatives: dict[str, CausalAlphaV3Candidate] = {}
        for candidate in config.candidates:
            representatives.setdefault(candidate.fit.digest, candidate)
        fit_cache = CausalAlphaV3FitCache(
            train_symbols=symbols, samples=prepared.samples
        )
        expected_signal_scopes = sum(
            len(nested[symbol].signal_contracts) for symbol in symbols
        )
        passed_signal: dict[str, CausalAlphaV3SignalGateEvidence] = {}
        fit_results: list[dict[str, object]] = []
        for fit_digest, candidate in representatives.items():
            metrics = []
            unavailable: list[str] = []
            for symbol in symbols:
                for contract in nested[symbol].signal_contracts:
                    try:
                        metric = signal_scope_builder(
                            symbol=symbol,
                            train_symbols=symbols,
                            samples=prepared.samples,
                            contract=contract,
                            candidate=candidate,
                            fit_cache=fit_cache,
                        )
                    except CausalAlphaV3SignalScopeUnavailable:
                        unavailable.append(contract.digest)
                        continue
                    if (
                        metric.fit_config_digest != fit_digest
                        or metric.symbol != symbol
                        or metric.episode_index != contract.episode_index
                        or metric.contract_digest != contract.digest
                    ):
                        raise ValueError("V3 signal scope evidence identity drifted")
                    base_store.write_signal_scope_metric(metric)
                    metrics.append(metric)
            evidence = (
                None
                if not metrics
                else signal_gate_evaluator(
                    tuple(metrics),
                    expected_scope_count=expected_signal_scopes,
                    gate=config.signal_gate,
                )
            )
            result_payload: dict[str, object] = {
                "evidence": None if evidence is None else evidence.to_payload(),
                "fit_config_digest": fit_digest,
                "passed": False if evidence is None else evidence.passed,
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_fit_signal_result_v2",
                "unavailable_scope_contract_digests": tuple(unavailable),
            }
            base_store.write_exact_artifact(
                Path("signal") / f"{fit_digest}.json", result_payload
            )
            fit_results.append(result_payload)
            if evidence is not None and evidence.passed:
                passed_signal[fit_digest] = evidence

        if not passed_signal:
            signal_rejection = CausalAlphaV3SignalRejected(tuple(fit_results))
            base_store.write_exact_artifact(
                "signal/rejection.json", signal_rejection.to_payload()
            )
            raise signal_rejection

        frozen_candidates = tuple(
            candidate
            for candidate in config.candidates
            if candidate.fit.digest in passed_signal
        )
        frozen_fit_digests = tuple(
            dict.fromkeys(candidate.fit.digest for candidate in frozen_candidates)
        )
        freeze = CausalAlphaV3CandidateFreeze(
            run_manifest_digest=manifest.digest,
            config_digest=config.digest,
            generator_code_digest=generator_digest,
            nested_partition_digest=nested_digest,
            candidate_digests=tuple(
                candidate.digest for candidate in frozen_candidates
            ),
            candidate_semantic_digests=tuple(
                candidate.semantic_digest for candidate in frozen_candidates
            ),
            fit_config_digests=frozen_fit_digests,
            signal_evidence_digests=tuple(
                passed_signal[fit_digest].digest for fit_digest in frozen_fit_digests
            ),
        )
        base_store.write_exact_artifact("freeze/candidates.json", freeze.to_payload())
        store = CausalAlphaV3ArtifactStore(
            root,
            run_manifest_digest=manifest.digest,
            freeze_digest=freeze.digest,
        )
        try:
            selection = selection_evaluator(
                train_symbols=symbols,
                samples=prepared.samples,
                nested_partitions=nested,
                candidates=frozen_candidates,
                environment_factories=prepared.environment_factories,
                episode_hours=prepared.episode_hours,
                thresholds=config.selection_gate,
                run_manifest_digest=manifest.digest,
                freeze_digest=freeze.digest,
                store=store,
                max_position_to_market_notional=(
                    prepared.max_position_to_market_notional
                ),
            )
        except CausalAlphaV3SelectionRejected as selection_rejection:
            for candidate_evidence in selection_rejection.candidates:
                store.write_candidate_evidence(candidate_evidence)
            store.write_exact_artifact(
                "selection/rejection.json", selection_rejection.to_payload()
            )
            raise
        for candidate_evidence in selection.candidates:
            store.write_candidate_evidence(candidate_evidence)
        store.write_exact_artifact("selection/evidence.json", selection.to_payload())

        selected = _resolve_selected(frozen_candidates, selection)
        teacher_config_digest = content_digest(
            {
                "freeze_digest": freeze.digest,
                "generator_code_digest": generator_digest,
                "run_manifest_digest": manifest.digest,
                "schema_version": "causal_alpha_v3_teacher_config_v2",
                "selected_candidate_digest": selected.digest,
                "selection_digest": selection.digest,
            }
        )
        sampling_config_digest = content_digest(
            {
                "nested_partition_digest": nested_digest,
                "schema_version": "causal_alpha_v3_sampling_config_v2",
            }
        )
        full_batches = _build_teacher_batches(
            selected=selected,
            prepared=prepared,
            teacher_config_digest=teacher_config_digest,
            sampling_config_digest=sampling_config_digest,
            fit_cache=fit_cache,
            episode_batch_builder=episode_batch_builder,
        )
        admission = admission_evaluator(
            train_symbols=symbols,
            batches=full_batches,
            environment_factories=prepared.environment_factories,
            episode_hours=prepared.episode_hours,
            run_manifest_digest=manifest.digest,
            freeze_digest=freeze.digest,
            selection=selection,
            store=store,
        )
        store.write_exact_artifact("admission/evidence.json", admission.to_payload())
        if not admission.passed:
            admission_rejection = CausalAlphaV3AdmissionRejected(
                admission_digest=admission.digest,
                selected_candidate_digest=selected.digest,
            )
            store.write_exact_artifact(
                "admission/rejection.json", admission_rejection.to_payload()
            )
            raise admission_rejection

        training_batches = {
            symbol: _training_batch_without_holdout(full_batches[symbol])
            for symbol in symbols
        }
        admission_contract_digests = {
            symbol: full_batches[symbol].contracts[-1].digest for symbol in symbols
        }
        package = UniversalCausalAlphaV3TeacherPackageV2(
            train_symbols=symbols,
            batches=training_batches,
            partition_digests={
                symbol: prepared.partitions[symbol].digest for symbol in symbols
            },
            sample_digests={
                symbol: prepared.samples[symbol].digest for symbol in symbols
            },
            admission_contract_digests=admission_contract_digests,
            run_manifest_digest=manifest.digest,
            freeze_digest=freeze.digest,
            selection_digest=selection.digest,
            teacher_admission_digest=admission.digest,
            selected_candidate_digest=selected.digest,
            generator_code_digest=generator_digest,
        )
        store.write_teacher_package(package)
        return package


__all__ = [
    "CausalAlphaV3AdmissionRejected",
    "CausalAlphaV3SignalRejected",
    "authored_config_payload",
    "run_universal_causal_alpha_v3_research_pipeline",
]
