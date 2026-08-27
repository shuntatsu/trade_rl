"""Restart-safe DB-backed Signal and Selection execution for Causal Alpha V9."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.learning.causal_alpha_v8 import (
    CausalAlphaV8Candidate,
    CausalAlphaV8TargetConfig,
    CausalAlphaV8TargetPath,
)
from trade_rl.learning.causal_alpha_v8_target import (
    causal_alpha_v8_target_paths_from_v7,
)
from trade_rl.learning.causal_alpha_v9 import (
    CausalAlphaV9Candidate,
    CausalAlphaV9Config,
    CausalAlphaV9TargetPath,
)
from trade_rl.learning.causal_alpha_v9_wave import (
    CausalAlphaV9TrainingRows,
    CausalAlphaV9WaveFit,
    causal_alpha_v9_wave_target_path,
    fit_causal_alpha_v9_wave,
)
from trade_rl.learning.rollout_evaluation import evaluate_action_path
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
    CausalAlphaV4RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    prepare_causal_alpha_v4_stage_data,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_science import (
    resolve_causal_alpha_v4_contract_rows,
)
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    build_causal_alpha_v6_replay_metric,
)
from trade_rl.workflows.universal_causal_alpha_v6_stage_entry import (
    _contract_column,
    _environment,
    _InitialStateEnvironment,
    _progress,
    _reward_scale,
)
from trade_rl.workflows.universal_causal_alpha_v7_runner import (
    CausalAlphaV7ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v7_stage_entry import (
    _fit_one_calibration,
    _target_bundle,
    signal_stage,
)
from trade_rl.workflows.universal_causal_alpha_v8_attribution import (
    CausalAlphaV8AttributionEvidence,
    build_causal_alpha_v8_attribution,
)
from trade_rl.workflows.universal_causal_alpha_v8_gates import (
    CausalAlphaV8SignalEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v9_gates import (
    V8_CANDIDATE_BY_V9,
    CausalAlphaV9SelectionEvidence,
    CausalAlphaV9SignalEvidence,
    evaluate_causal_alpha_v9_selection,
)

_REPLAY_LEAF_SCHEMA: Final = "causal_alpha_v9_replay_leaf_v1"
_RESULT_SCHEMA: Final = "causal_alpha_v9_terminal_result_v1"


class CausalAlphaV9RunLock(CausalAlphaV4RunLock):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.path = self.root / ".causal-alpha-v9.lock"


def causal_alpha_v9_stage_config_digest(
    source: CausalAlphaV7ResearchConfig,
    v8_target: CausalAlphaV8TargetConfig,
    wave: CausalAlphaV9Config,
) -> str:
    return content_digest(
        {
            "calibration_config_digest": source.calibration.digest,
            "schema_version": "causal_alpha_v9_stage_config_v1",
            "v8_target_config_digest": v8_target.digest,
            "wave_config_digest": wave.digest,
        }
    )


def _artifact(body: dict[str, object]) -> dict[str, object]:
    return {**body, "artifact_digest": content_digest(body)}


def _feature_surface(sample: Any) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    names = (
        *(f"local:{name}" for name in sample.local_context.feature_names),
        *(f"global:{name}" for name in sample.global_context.feature_names),
    )
    features = np.column_stack(
        (sample.local_context.values, sample.global_context.values)
    )
    available = np.column_stack(
        (sample.local_context.available, sample.global_context.available)
    )
    return names, features, available


def _wave_rows(prepared: Any) -> dict[str, CausalAlphaV9TrainingRows]:
    result: dict[str, CausalAlphaV9TrainingRows] = {}
    for symbol in prepared.train_symbols:
        sample = prepared.samples[symbol]
        names, features, available = _feature_surface(sample)
        result[symbol] = CausalAlphaV9TrainingRows(
            symbol=symbol,
            decision_indices=sample.decision_indices,
            label_end_indices=sample.label_end_indices_4h,
            features=features,
            feature_available=available,
            labels=sample.labels_4h,
            feature_names=names,
        )
    return result


def _fit_wave(
    training_rows: dict[str, CausalAlphaV9TrainingRows],
    *,
    cutoff: int,
    config: CausalAlphaV9Config,
) -> CausalAlphaV9WaveFit:
    return fit_causal_alpha_v9_wave(
        training_rows,
        knowledge_cutoff=cutoff,
        config=config,
    )


def _write_evidence(
    store: CausalAlphaV4ArtifactStore,
    stage: str,
    evidence: object,
) -> None:
    store.write_leaf(
        Path(stage) / "evidence.json",
        store.envelope(
            schema_version=f"causal_alpha_v9_{stage}_envelope_v1",
            evidence_digest=getattr(evidence, "digest"),
            payload=getattr(evidence, "to_payload")(),
        ),
    )


def _signal_evidence(
    prepared: Any,
    *,
    source_config: CausalAlphaV7ResearchConfig,
    wave_config: CausalAlphaV9Config,
    config_digest: str,
) -> CausalAlphaV9SignalEvidence:
    source = CausalAlphaV8SignalEvidence(
        signal_stage(
            prepared,
            calibration_config=source_config.calibration,
            target_config=source_config.target,
            v7_config_digest=config_digest,
        )
    )
    fit_digests: list[str] = []
    scope_count = 0
    qualified_count = 0
    training_rows = _wave_rows(prepared)
    count = len(prepared.nested_partitions[prepared.train_symbols[0]].signal_contracts)
    for index in range(count):
        contracts = _contract_column(prepared, "signal_contracts", index)
        cutoff = int(contracts[0].start)
        fit = _fit_wave(training_rows, cutoff=cutoff, config=wave_config)
        fit_digests.append(fit.digest)
        for symbol, contract in zip(prepared.train_symbols, contracts, strict=True):
            sample = prepared.samples[symbol]
            rows = resolve_causal_alpha_v4_contract_rows(
                sample,
                start=contract.start,
                stop=contract.stop,
            )
            _names, features, _available = _feature_surface(sample)
            heads = fit.predict_heads(features[rows])
            mean = np.mean(heads, axis=0)
            std = np.std(heads, axis=0)
            qualified = np.all(np.sign(heads) == np.sign(heads[0]), axis=0) & (
                np.abs(mean) > std + wave_config.edge_margin
            )
            scope_count += 1
            qualified_count += int(np.any(qualified))
    return CausalAlphaV9SignalEvidence(
        source_v8=source,
        wave_scope_count=scope_count,
        qualified_wave_scope_count=qualified_count,
        wave_fit_digests=tuple(fit_digests),
    )


def _target_paths(
    *,
    sample: Any,
    rows: np.ndarray,
    forecast: Any,
    wave_fit: CausalAlphaV9WaveFit,
    v7_targets: Any,
    v8_config: CausalAlphaV8TargetConfig,
    wave_config: CausalAlphaV9Config,
) -> dict[CausalAlphaV9Candidate, CausalAlphaV9TargetPath]:
    v8_targets = causal_alpha_v8_target_paths_from_v7(
        forecast=forecast,
        v7_paths=v7_targets,
        config=v8_config,
    )
    control = v7_targets[CausalAlphaV7Candidate.V6_CONTROL].v6_target_path
    robust = v8_targets[CausalAlphaV8Candidate.ROBUST_CALIBRATED].v6_target_path
    _names, features, _available = _feature_surface(sample)
    wave = causal_alpha_v9_wave_target_path(
        decision_indices=forecast.decision_indices,
        head_predictions=wave_fit.predict_heads(features[rows]),
        one_way_cost_rates=control.one_way_cost_rates,
        liquidity_weight_caps=control.liquidity_weight_caps,
        risk_weight_caps=control.risk_weight_caps,
        actionable_mask=control.actionable_mask,
        source_forecast_digest=forecast.digest,
        config=wave_config,
        initial_weight=control.initial_weight,
    )
    base = {
        CausalAlphaV9Candidate.V7_CONTROL: control,
        CausalAlphaV9Candidate.V8_ROBUST_CONTROL: robust,
        CausalAlphaV9Candidate.NONLINEAR_WAVE: wave,
    }
    return {
        candidate: CausalAlphaV9TargetPath(
            candidate=candidate,
            v6_target_path=path,
            source_forecast_digest=forecast.digest,
            wave_fit_digest=wave_fit.digest,
            v9_config_digest=wave_config.digest,
        )
        for candidate, path in base.items()
    }


def _build_replay(
    *,
    prepared: Any,
    resolved: Any,
    symbol: str,
    contract: Any,
    forecast: Any,
    state: Any,
    rows: np.ndarray,
    target: CausalAlphaV9TargetPath,
    config_digest: str,
    boundaries: Any,
) -> CausalAlphaV8ReplayMetric:
    environment = _environment(prepared, symbol)
    try:
        evaluation = evaluate_action_path(
            _InitialStateEnvironment(environment, contract.initial_state_mode),
            evaluation_range=(contract.start, contract.stop),
            actions=target.v6_target_path.targets[:, None].astype(np.float32),
        )
        base = build_causal_alpha_v6_replay_metric(
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            symbol=symbol,
            episode_index=contract.episode_index,
            contract_digest=contract.digest,
            fit_digest=resolved.base_fit.digest,
            forecast_digest=target.v6_target_path.forecast_digest,
            target_path=target.v6_target_path,
            evaluation=evaluation,
            episode_hours=float(prepared.prepared_v3.episode_hours),
            reward_scale=_reward_scale(environment),
        )
        mapped = V8_CANDIDATE_BY_V9[target.candidate]
        compatibility = CausalAlphaV8TargetPath(
            candidate=mapped,
            v6_target_path=target.v6_target_path,
            source_forecast_digest=forecast.digest,
            calibration_fit_digest=target.wave_fit_digest,
            v8_config_digest=target.v9_config_digest,
        )
        source = build_causal_alpha_v8_attribution(
            target_path=compatibility,
            evaluation=evaluation,
            confidence=np.abs(target.v6_target_path.direction_scores_4h),
            realized_volatility=np.asarray(state.realized_volatility)[rows],
            liquidity=np.asarray(state.liquidity)[rows],
            boundaries=boundaries,
            step_hours=float(prepared.prepared_v3.episode_hours) / len(rows),
        )
        attribution = CausalAlphaV8AttributionEvidence(
            candidate=mapped,
            target_path_digest=target.digest,
            boundaries_digest=source.boundaries_digest,
            step_economics_digest=source.step_economics_digest,
            decision_count=source.decision_count,
            gross_log_return=source.gross_log_return,
            net_log_return=source.net_log_return,
            total_execution_cost=source.total_execution_cost,
            total_exposure_hours=source.total_exposure_hours,
            cells=source.cells,
        )
        return CausalAlphaV8ReplayMetric(
            candidate=mapped,
            v6_metric=base,
            attribution=attribution,
            v8_target_path_digest=target.digest,
            source_forecast_digest=forecast.digest,
            calibration_fit_digest=target.wave_fit_digest,
            v8_config_digest=config_digest,
        )
    finally:
        environment.close()


def _leaf(
    store: CausalAlphaV4ArtifactStore,
    candidate: CausalAlphaV9Candidate,
    metric: CausalAlphaV8ReplayMetric,
) -> dict[str, object]:
    base = metric.v6_metric
    body: dict[str, object] = {
        "candidate": candidate.value,
        "config_digest": store.config_digest,
        "contract_digest": base.contract_digest,
        "episode_index": base.episode_index,
        "generator_code_digest": store.generator_code_digest,
        "replay": metric.to_payload(),
        "replay_digest": metric.digest,
        "run_manifest_digest": store.run_manifest_digest,
        "schema_version": _REPLAY_LEAF_SCHEMA,
        "symbol": base.symbol,
        "target_path_digest": metric.v8_target_path_digest,
        "v4_context_manifest_digest": store.v4_context_manifest_digest,
    }
    return _artifact(body)


def _path(candidate: CausalAlphaV9Candidate, symbol: str, episode: int) -> Path:
    return (
        Path("selection")
        / "replays"
        / f"{episode:02d}"
        / symbol
        / f"{candidate.value}.json"
    )


def _load(
    store: CausalAlphaV4ArtifactStore,
    *,
    path: Path,
    candidate: CausalAlphaV9Candidate,
    target: CausalAlphaV9TargetPath,
    symbol: str,
    episode: int,
    contract_digest: str,
) -> CausalAlphaV8ReplayMetric | None:
    leaf = store.load_leaf(path, expected_schema=_REPLAY_LEAF_SCHEMA)
    if leaf is None:
        return None
    metric = CausalAlphaV8ReplayMetric.from_payload(leaf["replay"])
    if (
        leaf["candidate"] != candidate.value
        or metric.candidate is not V8_CANDIDATE_BY_V9[candidate]
        or metric.v8_target_path_digest != target.digest
        or metric.v8_config_digest != store.config_digest
        or metric.v6_metric.symbol != symbol
        or metric.v6_metric.episode_index != episode
        or metric.v6_metric.contract_digest != contract_digest
        or metric.calibration_fit_digest != target.wave_fit_digest
        or leaf["replay_digest"] != metric.digest
    ):
        raise ValueError("V9 resumed replay identity drifted")
    return metric


def selection_stage(
    prepared: Any,
    signal: CausalAlphaV9SignalEvidence,
    *,
    source_config: CausalAlphaV7ResearchConfig,
    v8_config: CausalAlphaV8TargetConfig,
    wave_config: CausalAlphaV9Config,
    config_digest: str,
    store: CausalAlphaV4ArtifactStore,
) -> CausalAlphaV9SelectionEvidence:
    if not signal.passed:
        raise ValueError("V9 Selection cannot bypass Signal")
    records: list[CausalAlphaV8ReplayMetric] = []
    training_rows = _wave_rows(prepared)
    count = len(
        prepared.nested_partitions[prepared.train_symbols[0]].economic_contracts
    )
    for index in range(count):
        contracts = _contract_column(prepared, "economic_contracts", index)
        cutoff = int(contracts[0].start)
        resolved = _fit_one_calibration(
            prepared,
            train_stop=cutoff,
            config=source_config.calibration,
        )
        wave_fit = _fit_wave(training_rows, cutoff=cutoff, config=wave_config)
        try:
            for symbol, contract in zip(prepared.train_symbols, contracts, strict=True):
                rows, forecast, state, _features, v7_targets = _target_bundle(
                    prepared,
                    resolved,
                    symbol,
                    contract,
                    source_config.target,
                )
                targets = _target_paths(
                    sample=prepared.samples[symbol],
                    rows=rows,
                    forecast=forecast,
                    wave_fit=wave_fit,
                    v7_targets=v7_targets,
                    v8_config=v8_config,
                    wave_config=wave_config,
                )
                for candidate in CausalAlphaV9Candidate:
                    target = targets[candidate]
                    relative = _path(candidate, symbol, contract.episode_index)
                    metric = _load(
                        store,
                        path=relative,
                        candidate=candidate,
                        target=target,
                        symbol=symbol,
                        episode=contract.episode_index,
                        contract_digest=contract.digest,
                    )
                    if metric is None:
                        metric = _build_replay(
                            prepared=prepared,
                            resolved=resolved,
                            symbol=symbol,
                            contract=contract,
                            forecast=forecast,
                            state=state,
                            rows=rows,
                            target=target,
                            config_digest=config_digest,
                            boundaries=resolved.boundaries,
                        )
                        store.write_leaf(relative, _leaf(store, candidate, metric))
                    records.append(metric)
            for candidate in CausalAlphaV9Candidate:
                _progress(stage="selection", cutoff=cutoff, candidate=candidate.value)
        finally:
            del resolved, wave_fit
            gc.collect()
    return evaluate_causal_alpha_v9_selection(
        tuple(records), expected_symbols=prepared.train_symbols
    )


def run_causal_alpha_v9_selection(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV9SelectionEvidence:
    from trade_rl.workflows.universal_causal_alpha_v4_runtime_adapter import (
        prepare_causal_alpha_v4_runtime_adapter,
    )

    source_config = CausalAlphaV7ResearchConfig.from_json(config_path)
    v8_config = CausalAlphaV8TargetConfig(base=source_config.target)
    wave_config = CausalAlphaV9Config()
    config_digest = causal_alpha_v9_stage_config_digest(
        source_config, v8_config, wave_config
    )
    context, runtime, prepared_v3 = prepare_causal_alpha_v4_runtime_adapter(
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
    )
    generator_digest = content_digest(
        {
            "schema_version": "causal_alpha_v9_generator_code_v1",
            "source_tree_digest": prepared_v3.execution_identity.source_tree_digest,
        }
    )
    prepared = prepare_causal_alpha_v4_stage_data(
        config_digest=config_digest,
        generator_code_digest=generator_digest,
        runtime_context=context,
        runtime=runtime,
        prepared_v3=prepared_v3,
    )
    del context, runtime, prepared_v3
    gc.collect()
    root = Path(output_root)
    with CausalAlphaV9RunLock(root):
        store = CausalAlphaV4ArtifactStore(
            root,
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=config_digest,
            generator_code_digest=prepared.generator_code_digest,
        )
        signal = _signal_evidence(
            prepared,
            source_config=source_config,
            wave_config=wave_config,
            config_digest=config_digest,
        )
        _write_evidence(store, "signal", signal)
        selection = selection_stage(
            prepared,
            signal,
            source_config=source_config,
            v8_config=v8_config,
            wave_config=wave_config,
            config_digest=config_digest,
            store=store,
        )
        _write_evidence(store, "selection", selection)
        status = "selection_passed" if selection.passed else "selection_rejected"
        store.write_leaf(
            "result.json",
            _artifact(
                {
                    "evidence_digest": selection.digest,
                    "promotion_eligible": False,
                    "schema_version": _RESULT_SCHEMA,
                    "status": status,
                }
            ),
        )
        return selection


__all__ = [
    "causal_alpha_v9_stage_config_digest",
    "run_causal_alpha_v9_selection",
    "selection_stage",
]
