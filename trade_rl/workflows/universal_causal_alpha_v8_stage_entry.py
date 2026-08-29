"""Restart-safe DB-backed Signal and Selection execution for Causal Alpha V8."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v8 import (
    CausalAlphaV8Candidate,
    CausalAlphaV8TargetConfig,
    CausalAlphaV8TargetPath,
)
from trade_rl.learning.causal_alpha_v8_target import (
    causal_alpha_v8_target_paths_from_v7,
)
from trade_rl.learning.rollout_evaluation import evaluate_action_path
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
    CausalAlphaV4RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    prepare_causal_alpha_v4_stage_data,
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
    build_causal_alpha_v8_attribution,
)
from trade_rl.workflows.universal_causal_alpha_v8_gates import (
    CausalAlphaV8SelectionEvidence,
    CausalAlphaV8SignalEvidence,
    evaluate_causal_alpha_v8_selection,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)

_REPLAY_LEAF_SCHEMA: Final = "causal_alpha_v8_replay_leaf_v1"
_RESULT_SCHEMA: Final = "causal_alpha_v8_terminal_result_v1"


class CausalAlphaV8RunLock(CausalAlphaV4RunLock):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.path = self.root / ".causal-alpha-v8.lock"


def causal_alpha_v8_stage_config_digest(
    source: CausalAlphaV7ResearchConfig,
    target: CausalAlphaV8TargetConfig,
) -> str:
    return content_digest(
        {
            "calibration_config_digest": source.calibration.digest,
            "schema_version": "causal_alpha_v8_stage_config_v1",
            "target_config_digest": target.digest,
        }
    )


def _artifact(body: dict[str, object]) -> dict[str, object]:
    return {**body, "artifact_digest": content_digest(body)}


def _write_evidence(
    store: CausalAlphaV4ArtifactStore,
    stage: str,
    evidence: object,
) -> None:
    digest = getattr(evidence, "digest")
    payload = getattr(evidence, "to_payload")()
    store.write_leaf(
        Path(stage) / "evidence.json",
        store.envelope(
            schema_version=f"causal_alpha_v8_{stage}_envelope_v1",
            evidence_digest=digest,
            payload=payload,
        ),
    )


def _replay_leaf(
    *,
    store: CausalAlphaV4ArtifactStore,
    metric: CausalAlphaV8ReplayMetric,
) -> dict[str, object]:
    base = metric.v6_metric
    body: dict[str, object] = {
        "candidate": metric.candidate.value,
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


def _replay_path(metric: CausalAlphaV8ReplayMetric) -> Path:
    base = metric.v6_metric
    return (
        Path("selection")
        / "replays"
        / f"{base.episode_index:02d}"
        / base.symbol
        / f"{metric.candidate.value}.json"
    )


def _load_replay(
    *,
    store: CausalAlphaV4ArtifactStore,
    path: Path,
    expected: CausalAlphaV8TargetPath,
    symbol: str,
    episode_index: int,
    contract_digest: str,
    calibration_fit_digest: str,
) -> CausalAlphaV8ReplayMetric | None:
    leaf = store.load_leaf(path, expected_schema=_REPLAY_LEAF_SCHEMA)
    if leaf is None:
        return None
    metric = CausalAlphaV8ReplayMetric.from_payload(leaf["replay"])
    if (
        metric.candidate is not expected.candidate
        or metric.v8_target_path_digest != expected.digest
        or metric.v8_config_digest != store.config_digest
        or metric.v6_metric.symbol != symbol
        or metric.v6_metric.episode_index != episode_index
        or metric.v6_metric.contract_digest != contract_digest
        or metric.calibration_fit_digest != calibration_fit_digest
        or leaf["replay_digest"] != metric.digest
    ):
        raise ValueError("V8 resumed replay identity drifted")
    return metric


def _build_replay(
    *,
    prepared: Any,
    resolved: Any,
    symbol: str,
    contract: Any,
    forecast: Any,
    state: Any,
    rows: np.ndarray,
    target: CausalAlphaV8TargetPath,
    config_digest: str,
) -> CausalAlphaV8ReplayMetric:
    environment = _environment(prepared, symbol)
    try:
        reward_scale = _reward_scale(environment)
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
            reward_scale=reward_scale,
        )
        attribution = build_causal_alpha_v8_attribution(
            target_path=target,
            evaluation=evaluation,
            confidence=np.abs(target.v6_target_path.direction_scores_4h),
            realized_volatility=np.asarray(state.realized_volatility)[rows],
            liquidity=np.asarray(state.liquidity)[rows],
            boundaries=resolved.boundaries,
            step_hours=float(prepared.prepared_v3.episode_hours) / len(rows),
        )
        return CausalAlphaV8ReplayMetric(
            candidate=target.candidate,
            v6_metric=base,
            attribution=attribution,
            v8_target_path_digest=target.digest,
            source_forecast_digest=forecast.digest,
            calibration_fit_digest=resolved.calibration_fit.digest,
            v8_config_digest=config_digest,
            step_trace=evaluation.step_trace,
        )
    finally:
        environment.close()


def selection_stage(
    prepared: Any,
    signal: CausalAlphaV8SignalEvidence,
    *,
    source_config: CausalAlphaV7ResearchConfig,
    target_config: CausalAlphaV8TargetConfig,
    config_digest: str,
    store: CausalAlphaV4ArtifactStore,
) -> CausalAlphaV8SelectionEvidence:
    if not signal.passed:
        raise ValueError("V8 Selection cannot bypass Signal")
    records: list[CausalAlphaV8ReplayMetric] = []
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
        try:
            for symbol, contract in zip(prepared.train_symbols, contracts, strict=True):
                rows, forecast, state, _features, v7_targets = _target_bundle(
                    prepared,
                    resolved,
                    symbol,
                    contract,
                    source_config.target,
                )
                targets = causal_alpha_v8_target_paths_from_v7(
                    forecast=forecast,
                    v7_paths=v7_targets,
                    config=target_config,
                )
                for candidate in CausalAlphaV8Candidate:
                    target = targets[candidate]
                    relative = (
                        Path("selection")
                        / "replays"
                        / f"{contract.episode_index:02d}"
                        / symbol
                        / f"{candidate.value}.json"
                    )
                    metric = _load_replay(
                        store=store,
                        path=relative,
                        expected=target,
                        symbol=symbol,
                        episode_index=contract.episode_index,
                        contract_digest=contract.digest,
                        calibration_fit_digest=resolved.calibration_fit.digest,
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
                        )
                        store.write_leaf(
                            _replay_path(metric),
                            _replay_leaf(store=store, metric=metric),
                        )
                    records.append(metric)
            for candidate in CausalAlphaV8Candidate:
                _progress(stage="selection", cutoff=cutoff, candidate=candidate.value)
        finally:
            del resolved
            gc.collect()
    return evaluate_causal_alpha_v8_selection(
        tuple(records), expected_symbols=prepared.train_symbols
    )


def run_causal_alpha_v8_selection(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV8SelectionEvidence:
    """Run immutable Signal then restart-safe Selection on real DB-backed data."""

    from trade_rl.workflows.universal_causal_alpha_v4_runtime_adapter import (
        prepare_causal_alpha_v4_runtime_adapter,
    )

    source_config = CausalAlphaV7ResearchConfig.from_json(config_path)
    target_config = CausalAlphaV8TargetConfig(base=source_config.target)
    config_digest = causal_alpha_v8_stage_config_digest(source_config, target_config)
    context, runtime, prepared_v3 = prepare_causal_alpha_v4_runtime_adapter(
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
    )
    generator_digest = content_digest(
        {
            "schema_version": "causal_alpha_v8_generator_code_v1",
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
    with CausalAlphaV8RunLock(root):
        store = CausalAlphaV4ArtifactStore(
            root,
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=config_digest,
            generator_code_digest=prepared.generator_code_digest,
        )
        source_signal = signal_stage(
            prepared,
            calibration_config=source_config.calibration,
            target_config=source_config.target,
            v7_config_digest=config_digest,
        )
        signal = CausalAlphaV8SignalEvidence(source_signal)
        _write_evidence(store, "signal", signal)
        if not signal.passed:
            body = {
                "evidence_digest": signal.digest,
                "promotion_eligible": False,
                "schema_version": _RESULT_SCHEMA,
                "status": "signal_rejected",
            }
            store.write_leaf("result.json", _artifact(body))
            raise RuntimeError("causal alpha V8 Signal rejected")
        selection = selection_stage(
            prepared,
            signal,
            source_config=source_config,
            target_config=target_config,
            config_digest=config_digest,
            store=store,
        )
        _write_evidence(store, "selection", selection)
        body = {
            "evidence_digest": selection.digest,
            "promotion_eligible": False,
            "schema_version": _RESULT_SCHEMA,
            "status": "selection_passed" if selection.passed else "selection_rejected",
        }
        store.write_leaf("result.json", _artifact(body))
        return selection


__all__ = [
    "causal_alpha_v8_stage_config_digest",
    "run_causal_alpha_v8_selection",
    "selection_stage",
]
