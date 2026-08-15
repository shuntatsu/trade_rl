"""Artifact-bound Causal Alpha V3 selection and admission research route."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

import trade_rl.learning.causal_alpha_v3 as _learning_module
import trade_rl.workflows.universal_causal_alpha_v3 as _fit_module
import trade_rl.workflows.universal_causal_alpha_v3_contracts as _contracts_module
import trade_rl.workflows.universal_causal_alpha_v3_selection as _selection_module
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_costs import (
    causal_alpha_liquidity_weight_caps,
    causal_alpha_one_way_cost_rates,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    build_causal_alpha_symbol_samples,
    build_chronological_episode_partition,
    validate_universal_causal_alpha_partitions,
)
from trade_rl.workflows.universal_causal_alpha_selection import (
    CausalAlphaSelectionThresholds,
)
from trade_rl.workflows.universal_causal_alpha_v3 import (
    CausalAlphaV3FitCache,
    build_causal_alpha_v3_contract_targets,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3SelectionEvidence,
    UniversalCausalAlphaV3TeacherPackage,
)
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    CausalAlphaV3SelectionRejected,
    causal_alpha_v3_grid_digest,
    causal_alpha_v3_metric_from_payload,
    default_causal_alpha_v3_candidate_grid,
    evaluate_causal_alpha_v3_selection,
    load_causal_alpha_v3_selection_checkpoint,
    write_causal_alpha_v3_selection_checkpoint_metric,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    admit_causal_alpha_v3_teacher,
)
from trade_rl.workflows.universal_training_runner import UniversalTrainingRuntime


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ResearchResult:
    phase: Literal["selection_completed", "admission_completed"]
    selection_path: Path
    selection_digest: str
    package_path: Path | None = None
    package_digest: str | None = None


def causal_alpha_v3_generator_code_digest() -> str:
    files = {}
    for module in (
        _learning_module,
        _fit_module,
        _contracts_module,
        _selection_module,
    ):
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str):
            raise RuntimeError("V3 generator source path is unavailable")
        files[module.__name__] = hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    files[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return content_digest(
        {"files": files, "schema_version": "causal_alpha_v3_generator_code_v1"}
    )


def _persist(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_bytes(Path(path), canonical_json_bytes(payload) + b"\n")


def _prepare_train_evidence(
    runtime: UniversalTrainingRuntime,
    *,
    fold_train_range: tuple[int, int],
) -> tuple[
    dict[str, CausalAlphaSymbolSamples],
    dict[str, CausalAlphaEpisodePartition],
    dict[str, object],
    float,
]:
    routed = runtime.routed_environment_factory
    provider = routed.instrument_context_provider
    if provider is None:
        raise ValueError("V3 research requires instrument context evidence")
    binding_by_symbol = {
        binding.concrete_symbol: binding for binding in routed.bindings
    }
    samples: dict[str, CausalAlphaSymbolSamples] = {}
    partitions: dict[str, CausalAlphaEpisodePartition] = {}
    factories: dict[str, object] = {}
    episode_hours: list[float] = []
    for symbol in runtime.train_symbols:
        binding = binding_by_symbol[symbol]
        factory = partial(routed.concrete_environment_factory, binding)
        factories[symbol] = factory
        environment = factory()
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("V3 research environment is not closable")
        try:
            partitions[symbol] = build_chronological_episode_partition(
                environment, train_range=fold_train_range
            )
            samples[symbol] = build_causal_alpha_symbol_samples(
                environment=environment,
                binding=binding,
                instrument_context_provider=provider,
                train_range=fold_train_range,
                feature_schema_digest=runtime.feature_schema_digest,
            )
            risk = getattr(getattr(environment, "portfolio_risk", None), "config", None)
            if getattr(risk, "max_position_to_market_notional", None) != 0.02:
                raise ValueError("V3 hard liquidity contract must remain 0.02")
            value = getattr(getattr(environment, "config", None), "episode_hours", None)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("V3 episode_hours is unavailable")
            episode_hours.append(float(value))
        finally:
            close()
    validate_universal_causal_alpha_partitions(
        train_symbols=runtime.train_symbols, partitions=partitions
    )
    if len(set(episode_hours)) != 1 or not math.isfinite(episode_hours[0]):
        raise ValueError("V3 train-symbol episode horizons differ")
    return samples, partitions, factories, episode_hours[0]


def _holdout_targets(
    *,
    runtime: UniversalTrainingRuntime,
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partitions: Mapping[str, CausalAlphaEpisodePartition],
    factories: Mapping[str, object],
    selection: CausalAlphaV3SelectionEvidence,
    fit_cache: CausalAlphaV3FitCache,
) -> dict[str, np.ndarray]:
    selected = next(
        item.candidate
        for item in selection.candidates
        if item.candidate.digest == selection.selected_candidate_digest
    )
    targets: dict[str, np.ndarray] = {}
    for symbol in runtime.train_symbols:
        factory = factories[symbol]
        if not callable(factory):
            raise TypeError("V3 holdout environment factory is invalid")
        environment = factory()
        try:
            contract = partitions[symbol].holdout_contract
            decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
            execution_cost = getattr(environment.config, "execution_cost", None)
            signal_delay = getattr(environment.config, "signal_delay_decisions", None)
            decision_bars = getattr(environment, "decision_bars", None)
            if not isinstance(execution_cost, ExecutionCostConfig):
                raise TypeError("V3 holdout execution cost is unavailable")
            if not isinstance(signal_delay, int) or not isinstance(decision_bars, int):
                raise ValueError("V3 holdout execution timing is unavailable")
            costs = causal_alpha_one_way_cost_rates(
                environment.dataset,
                execution_cost,
                decision_indices=decisions,
                signal_delay_decisions=signal_delay,
                decision_bars=decision_bars,
            )
            caps = causal_alpha_liquidity_weight_caps(
                environment.dataset,
                decision_indices=decisions,
                reference_portfolio_value=samples[symbol].reference_equity,
                max_position_to_market_notional=0.02,
                lookback_decisions=96,
                lower_quantile=0.10,
                safety_multiplier=0.80,
            )
            targets[symbol] = build_causal_alpha_v3_contract_targets(
                symbol=symbol,
                samples=samples,
                contract=contract,
                candidate=selected,
                fit_cache=fit_cache,
                one_way_cost_rates=costs,
                liquidity_weight_caps=caps,
            ).targets
        finally:
            environment.close()
    return targets


def run_causal_alpha_v3_research(
    *,
    runtime: UniversalTrainingRuntime,
    fold_train_range: tuple[int, int],
    output_root: Path,
    stage_limit: Literal["selection", "admission"] = "admission",
) -> CausalAlphaV3ResearchResult:
    """Run train-only selection and optionally exact-once admission."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    samples, partitions, factories, episode_hours = _prepare_train_evidence(
        runtime, fold_train_range=fold_train_range
    )
    candidates = default_causal_alpha_v3_candidate_grid()
    thresholds = CausalAlphaSelectionThresholds()
    generator_digest = causal_alpha_v3_generator_code_digest()
    fit_cache = CausalAlphaV3FitCache(
        train_symbols=runtime.train_symbols, samples=samples
    )
    grid_digest = causal_alpha_v3_grid_digest(candidates, thresholds)
    checkpoint_path = root / "causal-alpha-v3-selection-checkpoint.jsonl"
    progress_path = root / "causal-alpha-v3-progress.json"
    selection_path = root / "causal-alpha-v3-selection.json"
    initial = load_causal_alpha_v3_selection_checkpoint(
        checkpoint_path,
        expected_grid_digest=grid_digest,
        expected_generator_code_digest=generator_digest,
        expected_sample_scope_digest=fit_cache.sample_scope_digest,
    )
    latest: dict[str, object] = {}

    def progress(payload: Mapping[str, object]) -> None:
        raw_metric = payload.get("episode_metric")
        if isinstance(raw_metric, Mapping):
            write_causal_alpha_v3_selection_checkpoint_metric(
                checkpoint_path,
                causal_alpha_v3_metric_from_payload(raw_metric),
                grid_digest=grid_digest,
                generator_code_digest=generator_digest,
                sample_scope_digest=fit_cache.sample_scope_digest,
            )
        latest.clear()
        latest.update(payload)
        _persist(progress_path, latest)

    try:
        selection = evaluate_causal_alpha_v3_selection(
            train_symbols=runtime.train_symbols,
            samples=samples,
            partitions=partitions,
            candidates=candidates,
            environment_factories=factories,
            episode_hours=episode_hours,
            thresholds=thresholds,
            generator_code_digest=generator_digest,
            fit_cache=fit_cache,
            progress_callback=progress,
            initial_metrics=initial,
        )
    except CausalAlphaV3SelectionRejected as rejection:
        _persist(
            root / "causal-alpha-v3-selection-rejected.json", rejection.to_payload()
        )
        _persist(
            progress_path,
            {
                **latest,
                "phase": "causal_alpha_v3_selection_rejected",
                "selection_rejection_digest": rejection.digest,
            },
        )
        raise
    _persist(selection_path, selection.to_payload())
    if stage_limit == "selection":
        return CausalAlphaV3ResearchResult(
            phase="selection_completed",
            selection_path=selection_path,
            selection_digest=selection.digest,
        )
    holdout_targets = _holdout_targets(
        runtime=runtime,
        samples=samples,
        partitions=partitions,
        factories=factories,
        selection=selection,
        fit_cache=fit_cache,
    )
    teacher_config_digest = content_digest(
        {
            "generator_code_digest": generator_digest,
            "schema_version": "universal_causal_alpha_v3_teacher_config_v1",
            "selected_candidate_digest": selection.selected_candidate_digest,
            "selection_digest": selection.digest,
        }
    )
    package_path = root / "causal-alpha-v3-package.json"
    package: UniversalCausalAlphaV3TeacherPackage = admit_causal_alpha_v3_teacher(
        selection=selection,
        selection_evidence_path=selection_path,
        holdout_contracts={
            symbol: partitions[symbol].holdout_contract
            for symbol in runtime.train_symbols
        },
        holdout_targets=holdout_targets,
        environment_factories=factories,
        episode_hours=episode_hours,
        teacher_config_digest=teacher_config_digest,
        admission_evidence_path=root / "causal-alpha-v3-admission.json",
        package_evidence_path=package_path,
    )
    _persist(
        progress_path,
        {
            **latest,
            "package_digest": package.digest,
            "phase": "causal_alpha_v3_admission_completed",
            "teacher_admission_passed": True,
        },
    )
    return CausalAlphaV3ResearchResult(
        phase="admission_completed",
        selection_path=selection_path,
        selection_digest=selection.digest,
        package_path=package_path,
        package_digest=package.digest,
    )


__all__ = [
    "CausalAlphaV3ResearchResult",
    "causal_alpha_v3_generator_code_digest",
    "run_causal_alpha_v3_research",
]
