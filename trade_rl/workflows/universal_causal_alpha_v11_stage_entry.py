"""Restart-safe DB-backed study-arm execution for Causal Alpha V11."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.learning.causal_alpha_v8 import (
    CausalAlphaV8Candidate,
    CausalAlphaV8TargetConfig,
)
from trade_rl.learning.causal_alpha_v8_target import (
    causal_alpha_v8_target_paths_from_v7,
)
from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Config
from trade_rl.learning.causal_alpha_v10 import (
    CausalAlphaV10Candidate,
    CausalAlphaV10Config,
    CausalAlphaV10TargetPath,
)
from trade_rl.learning.causal_alpha_v11 import (
    CausalAlphaV11Candidate,
    CausalAlphaV11Config,
    CausalAlphaV11StudyArm,
    evaluate_v11_sizing_feasibility,
)
from trade_rl.learning.causal_alpha_v11_calibration import (
    fit_causal_alpha_v11_sign_calibration,
)
from trade_rl.learning.causal_alpha_v11_diagnostics import (
    CausalAlphaV11DiagnosticEvidence,
    build_causal_alpha_v11_diagnostics,
)
from trade_rl.learning.causal_alpha_v11_policy import (
    CausalAlphaV11CompiledTarget,
    CausalAlphaV11TracePolicy,
    compile_causal_alpha_v11_target,
)
from trade_rl.learning.rollout_evaluation import evaluate_action_path
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
    CausalAlphaV4RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v6_stage_entry import (
    _contract_column,
    _environment,
    _InitialStateEnvironment,
    _progress,
)
from trade_rl.workflows.universal_causal_alpha_v7_runner import (
    CausalAlphaV7ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v7_stage_entry import (
    _fit_one_calibration,
    _target_bundle,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v9_stage_entry import (
    _fit_wave as _fit_v9_wave,
)
from trade_rl.workflows.universal_causal_alpha_v9_stage_entry import (
    _wave_rows as _v9_wave_rows,
)
from trade_rl.workflows.universal_causal_alpha_v10_stage_entry import (
    _execution_rebalance_contract,
    _feature_surface,
    _metric_from_evaluation,
    _prepare_causal_alpha_v10_stage_data,
    _require_execution_rebalance_contract,
)
from trade_rl.workflows.universal_causal_alpha_v11_gates import (
    CausalAlphaV11SelectionEvidence,
    evaluate_causal_alpha_v11_selection,
)

_REPLAY_LEAF_SCHEMA: Final = "causal_alpha_v11_replay_leaf_v1"
_RESULT_SCHEMA: Final = "causal_alpha_v11_terminal_result_v1"
_CONTROL_TOLERANCE: Final = 1e-12


def causal_alpha_v11_stage_config_digest(
    *, source_config_digest: str, study_arm: CausalAlphaV11StudyArm
) -> str:
    """Bind one output root to one and only one V11 treatment arm."""

    require_sha256(source_config_digest, field="V11 source config digest")
    arm = CausalAlphaV11StudyArm(study_arm)
    return content_digest(
        {
            "schema_version": "causal_alpha_v11_stage_config_v1",
            "source_config_digest": source_config_digest,
            "study_arm": arm.value,
            "v11_config_digest": CausalAlphaV11Config().digest,
        }
    )


def _require_control_equivalence(observed: object, r21: object) -> None:
    """Fail closed on any behavior-neutral V9 control economic drift."""

    observed_base = getattr(observed, "v6_metric", None)
    r21_base = getattr(r21, "v6_metric", None)
    observed_trace = getattr(observed, "step_trace", None)
    r21_trace = getattr(r21, "step_trace", None)
    observed_lifecycle = getattr(observed, "lifecycle_trace", None)
    r21_lifecycle = getattr(r21, "lifecycle_trace", None)
    if any(
        value is None
        for value in (
            observed_base,
            r21_base,
            observed_trace,
            r21_trace,
            observed_lifecycle,
            r21_lifecycle,
        )
    ):
        raise ValueError("V11 behavior-neutral control drifted: missing evidence")
    assert observed_base is not None
    assert r21_base is not None
    assert observed_trace is not None
    assert r21_trace is not None
    assert observed_lifecycle is not None
    assert r21_lifecycle is not None
    scalar_fields = (
        "gross_return",
        "net_return",
        "total_execution_cost",
        "turnover_per_day",
    )
    for field_name in scalar_fields:
        if not math.isclose(
            float(getattr(observed_base, field_name)),
            float(getattr(r21_base, field_name)),
            rel_tol=0.0,
            abs_tol=_CONTROL_TOLERANCE,
        ):
            raise ValueError(f"V11 behavior-neutral control drifted: {field_name}")
    trace_fields = (
        "requested_targets",
        "projected_targets",
        "realized_weights",
        "gross_returns",
        "net_returns",
        "costs",
        "turnovers",
    )
    for field_name in trace_fields:
        if not np.allclose(
            np.asarray(getattr(observed_trace, field_name)),
            np.asarray(getattr(r21_trace, field_name)),
            rtol=0.0,
            atol=_CONTROL_TOLERANCE,
        ):
            raise ValueError(
                f"V11 behavior-neutral control drifted: trace {field_name}"
            )
    if tuple(observed_lifecycle.transition_classes) != tuple(
        r21_lifecycle.transition_classes
    ):
        raise ValueError("V11 behavior-neutral control drifted: lifecycle transitions")
    for field_name in ("execution_intent_targets", "final_risk_targets"):
        if not np.allclose(
            np.asarray(getattr(observed_lifecycle, field_name)),
            np.asarray(getattr(r21_lifecycle, field_name)),
            rtol=0.0,
            atol=_CONTROL_TOLERANCE,
        ):
            raise ValueError(
                f"V11 behavior-neutral control drifted: lifecycle {field_name}"
            )


class CausalAlphaV11RunLock(CausalAlphaV4RunLock):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.path = self.root / ".causal-alpha-v11.lock"


def _artifact(body: dict[str, object]) -> dict[str, object]:
    return {**body, "artifact_digest": content_digest(body)}


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"V11 {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V11 {field} must be finite")
    return result


def _r21_path(root: Path, candidate: str, symbol: str, episode: int) -> Path:
    return (
        root / "selection" / "replays" / f"{episode:02d}" / symbol / f"{candidate}.json"
    )


def _load_r21_leaf(
    root: Path, *, candidate: str, symbol: str, episode: int, contract_digest: str
) -> tuple[dict[str, object], CausalAlphaV8ReplayMetric]:
    path = _r21_path(root, candidate, symbol, episode)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"V11 r21 replay is unreadable: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("V11 r21 replay payload is invalid")
    payload = dict(raw)
    digest = str(payload.pop("artifact_digest", ""))
    if content_digest(payload) != digest:
        raise ValueError("V11 r21 replay outer digest mismatch")
    if (
        raw.get("schema_version") != "causal_alpha_v10_replay_leaf_v3"
        or raw.get("candidate") != candidate
        or raw.get("symbol") != symbol
        or raw.get("episode_index") != episode
        or raw.get("contract_digest") != contract_digest
    ):
        raise ValueError("V11 r21 replay identity drifted")
    metric = CausalAlphaV8ReplayMetric.from_payload(raw["replay"])
    if metric.digest != raw.get("replay_digest"):
        raise ValueError("V11 r21 replay digest binding drifted")
    return raw, metric


def _compatibility_target(
    *,
    candidate: CausalAlphaV10Candidate,
    path: Any,
    source_forecast_digest: str,
    fit_digest: str,
    hierarchy_input_digest: str | None = None,
    hierarchy_reasons: tuple[str, ...] = (),
) -> CausalAlphaV10TargetPath:
    mapped_reasons = tuple(
        "neutral_fast_expiry" if reason == "neutral_expiry_2" else reason
        for reason in hierarchy_reasons
    )
    return CausalAlphaV10TargetPath(
        candidate=candidate,
        v6_target_path=path,
        source_forecast_digest=source_forecast_digest,
        fast_fit_digest=fit_digest,
        slow_fit_digest=fit_digest,
        v10_config_digest=CausalAlphaV10Config().digest,
        hierarchy_input_digest=hierarchy_input_digest,
        hierarchy_reasons=mapped_reasons,
        hierarchy_reason_counts=tuple(
            sorted(
                (reason, mapped_reasons.count(reason)) for reason in set(mapped_reasons)
            )
        ),
    )


def _replay(
    *,
    prepared: Any,
    resolved: Any,
    symbol: str,
    contract: Any,
    forecast: Any,
    state: Any,
    rows: np.ndarray,
    target: CausalAlphaV10TargetPath,
    model: object | None,
    config_digest: str,
    boundaries: Any,
    execution_contract: Any,
) -> CausalAlphaV8ReplayMetric:
    environment = _environment(prepared, symbol)
    try:
        _require_execution_rebalance_contract(environment, execution_contract)
        wrapped = _InitialStateEnvironment(environment, contract.initial_state_mode)
        if model is None:
            evaluation = evaluate_action_path(
                wrapped,
                evaluation_range=(contract.start, contract.stop),
                actions=target.v6_target_path.targets[:, None].astype(np.float32),
            )
        else:
            evaluation = evaluate_action_path(
                wrapped,
                evaluation_range=(contract.start, contract.stop),
                model=model,
            )
        return _metric_from_evaluation(
            prepared=prepared,
            resolved=resolved,
            symbol=symbol,
            contract=contract,
            forecast=forecast,
            state=state,
            rows=rows,
            target=target,
            evaluation=evaluation,
            config_digest=config_digest,
            boundaries=boundaries,
            environment=environment,
        )
    finally:
        environment.close()


def _leaf_path(candidate: CausalAlphaV11Candidate, symbol: str, episode: int) -> Path:
    return (
        Path("selection")
        / "replays"
        / f"{episode:02d}"
        / symbol
        / f"{candidate.value}.json"
    )


def _load_metric(
    store: CausalAlphaV4ArtifactStore,
    *,
    path: Path,
    candidate: CausalAlphaV11Candidate,
    study_arm: CausalAlphaV11StudyArm,
    symbol: str,
    episode: int,
    contract_digest: str,
    candidate_input_digest: str,
    r21_replay_digest: str | None,
) -> CausalAlphaV8ReplayMetric | None:
    leaf = store.load_leaf(path, expected_schema=_REPLAY_LEAF_SCHEMA)
    if leaf is None:
        return None
    metric = CausalAlphaV8ReplayMetric.from_payload(leaf["replay"])
    expected_mapped = {
        CausalAlphaV11Candidate.V8_CASH_SANITY: CausalAlphaV8Candidate.V7_CONTROL,
        CausalAlphaV11Candidate.V9_CONTROL: CausalAlphaV8Candidate.ROBUST_CONTRARIAN,
        CausalAlphaV11Candidate.TREATMENT: CausalAlphaV8Candidate.ROBUST_CALIBRATED,
    }[candidate]
    if (
        leaf.get("candidate") != candidate.value
        or leaf.get("study_arm") != study_arm.value
        or leaf.get("candidate_input_digest") != candidate_input_digest
        or leaf.get("config_digest") != store.config_digest
        or metric.candidate is not expected_mapped
        or metric.v8_config_digest != store.config_digest
        or metric.v6_metric.symbol != symbol
        or metric.v6_metric.episode_index != episode
        or metric.v6_metric.contract_digest != contract_digest
        or metric.step_trace is None
        or metric.lifecycle_trace is None
        or leaf.get("step_trace_digest") != metric.step_trace.digest
        or leaf.get("lifecycle_trace_digest") != metric.lifecycle_trace.digest
        or leaf.get("replay_digest") != metric.digest
        or leaf.get("r21_replay_digest") != r21_replay_digest
    ):
        raise ValueError("V11 resumed replay identity drifted")
    return metric


def _write_metric(
    store: CausalAlphaV4ArtifactStore,
    *,
    candidate: CausalAlphaV11Candidate,
    study_arm: CausalAlphaV11StudyArm,
    symbol: str,
    episode: int,
    contract_digest: str,
    candidate_input_digest: str,
    compatibility_target: CausalAlphaV10TargetPath,
    v11_target: CausalAlphaV11CompiledTarget | None,
    metric: CausalAlphaV8ReplayMetric,
    r21_replay_digest: str | None,
) -> None:
    if metric.step_trace is None or metric.lifecycle_trace is None:
        raise ValueError("V11 replay is missing trace evidence")
    body: dict[str, object] = {
        "candidate": candidate.value,
        "candidate_input_digest": candidate_input_digest,
        "compatibility_target": compatibility_target.to_payload(),
        "config_digest": store.config_digest,
        "contract_digest": contract_digest,
        "episode_index": episode,
        "generator_code_digest": store.generator_code_digest,
        "lifecycle_trace_digest": metric.lifecycle_trace.digest,
        "replay": metric.to_payload(),
        "replay_digest": metric.digest,
        "r21_replay_digest": r21_replay_digest,
        "run_manifest_digest": store.run_manifest_digest,
        "schema_version": _REPLAY_LEAF_SCHEMA,
        "step_trace_digest": metric.step_trace.digest,
        "study_arm": study_arm.value,
        "symbol": symbol,
        "v4_context_manifest_digest": store.v4_context_manifest_digest,
    }
    if v11_target is not None:
        body["policy_digest"] = v11_target.digest
        body["v11_target"] = v11_target.target.to_payload()
    store.write_leaf(_leaf_path(candidate, symbol, episode), _artifact(body))


def _diagnostic_path(symbol: str, episode: int) -> Path:
    return Path("diagnostics") / "scopes" / f"{episode:02d}" / f"{symbol}.json"


def _write_diagnostic(
    store: CausalAlphaV4ArtifactStore,
    evidence: CausalAlphaV11DiagnosticEvidence,
    *,
    study_arm: CausalAlphaV11StudyArm,
    r21_replay_digest: str,
) -> None:
    store.write_leaf(
        _diagnostic_path(evidence.symbol, int(evidence.episode_id)),
        _artifact(
            {
                "config_digest": store.config_digest,
                "diagnostic": evidence.to_payload(),
                "diagnostic_digest": evidence.digest,
                "episode_index": int(evidence.episode_id),
                "r21_replay_digest": r21_replay_digest,
                "schema_version": "causal_alpha_v11_diagnostic_leaf_v1",
                "study_arm": study_arm.value,
                "symbol": evidence.symbol,
            }
        ),
    )


def _load_diagnostic_digest(
    store: CausalAlphaV4ArtifactStore,
    *,
    symbol: str,
    episode: int,
    study_arm: CausalAlphaV11StudyArm,
    r21_replay_digest: str,
) -> str | None:
    leaf = store.load_leaf(
        _diagnostic_path(symbol, episode),
        expected_schema="causal_alpha_v11_diagnostic_leaf_v1",
    )
    if leaf is None:
        return None
    diagnostic = leaf.get("diagnostic")
    if not isinstance(diagnostic, dict):
        raise ValueError("V11 resumed diagnostic payload is invalid")
    digest = str(diagnostic.get("artifact_digest", ""))
    payload = dict(diagnostic)
    payload.pop("artifact_digest", None)
    if (
        content_digest(payload) != digest
        or leaf.get("diagnostic_digest") != digest
        or leaf.get("study_arm") != study_arm.value
        or leaf.get("symbol") != symbol
        or leaf.get("episode_index") != episode
        or leaf.get("r21_replay_digest") != r21_replay_digest
        or leaf.get("config_digest") != store.config_digest
    ):
        raise ValueError("V11 resumed diagnostic identity drifted")
    return digest


def _diagnostic_index(
    store: CausalAlphaV4ArtifactStore,
    *,
    study_arm: CausalAlphaV11StudyArm,
    scope_paths: tuple[Path, ...],
) -> dict[str, object]:
    payloads: list[dict[str, object]] = []
    for path in scope_paths:
        leaf = store.load_leaf(
            path, expected_schema="causal_alpha_v11_diagnostic_leaf_v1"
        )
        if leaf is None or not isinstance(leaf.get("diagnostic"), dict):
            raise ValueError("V11 diagnostic scope is incomplete")
        payloads.append(cast(dict[str, object], leaf["diagnostic"]))
    trades: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    for payload in payloads:
        trades.extend(cast(list[dict[str, object]], payload["trades"]))
        entries.extend(cast(list[dict[str, object]], payload["entries"]))
    before = [
        _number(
            trade["entry_to_neutral_net_log_return"],
            field="entry-to-neutral return",
        )
        for trade in trades
    ]
    after = [
        _number(
            trade["neutral_to_exit_net_log_return"],
            field="neutral-to-exit return",
        )
        for trade in trades
    ]
    entry_edges = [
        _number(entry["entry_edge"], field="entry edge") for entry in entries
    ]
    body: dict[str, object] = {
        "config_digest": store.config_digest,
        "diagnostic_digests": tuple(
            str(payload["artifact_digest"]) for payload in payloads
        ),
        "entry_count": len(entries),
        "mean_entry_edge": float(np.mean(entry_edges)) if entry_edges else 0.0,
        "mean_entry_to_neutral_net_log_return": (
            float(np.mean(before)) if before else 0.0
        ),
        "mean_neutral_to_exit_net_log_return": (
            float(np.mean(after)) if after else 0.0
        ),
        "neutral_observed_trade_count": sum(
            trade["first_neutral_index"] is not None for trade in trades
        ),
        "right_censored_trade_count": sum(
            bool(trade["right_censored"]) for trade in trades
        ),
        "schema_version": "causal_alpha_v11_diagnostic_index_v1",
        "scope_count": len(payloads),
        "study_arm": study_arm.value,
        "trade_count": len(trades),
    }
    artifact = _artifact(body)
    store.write_leaf("diagnostics/evidence.json", artifact)
    return artifact


def _r21_signal_digest(root: Path) -> str:
    path = root / "signal" / "evidence.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("V11 r21 Signal evidence is unreadable") from error
    if not isinstance(raw, dict):
        raise ValueError("V11 r21 Signal evidence is invalid")
    outer = dict(raw)
    digest = str(outer.pop("artifact_digest", ""))
    if content_digest(outer) != digest:
        raise ValueError("V11 r21 Signal outer digest mismatch")
    payload = raw.get("payload")
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        raise ValueError("V11 cannot bypass rejected r21 Signal")
    evidence_digest = str(raw.get("evidence_digest", ""))
    if payload.get("artifact_digest") != evidence_digest:
        raise ValueError("V11 r21 Signal digest binding drifted")
    return evidence_digest


def run_causal_alpha_v11_selection(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    r21_output_root: Path,
    output_root: Path,
    study_arm: CausalAlphaV11StudyArm,
) -> CausalAlphaV11SelectionEvidence:
    """Run one V11 study arm on DB-backed r21-equivalent economic scopes."""

    arm = CausalAlphaV11StudyArm(study_arm)
    source_config = CausalAlphaV7ResearchConfig.from_json(config_path)
    v9_config = CausalAlphaV9Config()
    v11_config = CausalAlphaV11Config()
    config_digest = causal_alpha_v11_stage_config_digest(
        source_config_digest=source_config.digest,
        study_arm=arm,
    )
    r21_root = Path(r21_output_root)
    signal_digest = _r21_signal_digest(r21_root)
    prepared = _prepare_causal_alpha_v10_stage_data(
        run_config_path=Path(run_config_path),
        runtime_manifest_path=Path(runtime_manifest_path),
        v4_context_manifest_path=Path(v4_context_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
        config_digest=config_digest,
    )
    root = Path(output_root)
    with CausalAlphaV11RunLock(root):
        store = CausalAlphaV4ArtifactStore(
            root,
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=config_digest,
            generator_code_digest=prepared.generator_code_digest,
        )
        store.write_leaf(
            "signal-reference.json",
            _artifact(
                {
                    "r21_output_root": str(r21_root),
                    "r21_signal_evidence_digest": signal_digest,
                    "schema_version": "causal_alpha_v11_signal_reference_v1",
                    "study_arm": arm.value,
                }
            ),
        )
        cash_metrics: list[CausalAlphaV8ReplayMetric] = []
        control_metrics: list[CausalAlphaV8ReplayMetric] = []
        treatment_metrics: list[CausalAlphaV8ReplayMetric] = []
        diagnostic_digests: list[str] = []
        diagnostic_paths: list[Path] = []
        sizing_targets: list[np.ndarray] = []
        v9_rows = _v9_wave_rows(prepared)
        execution_contracts = {
            symbol: _execution_rebalance_contract(prepared, symbol)
            for symbol in prepared.train_symbols
        }
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
            v9_fit = _fit_v9_wave(v9_rows, cutoff=cutoff, config=v9_config)
            sign_calibration = None
            if arm in (
                CausalAlphaV11StudyArm.SIGN_CALIBRATED_ENTRY,
                CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING,
            ):
                calibration_start = (
                    cutoff - v11_config.calibration_hours * v9_config.decisions_per_hour
                )
                source_fit = _fit_v9_wave(
                    v9_rows, cutoff=calibration_start, config=v9_config
                )
                sign_calibration = fit_causal_alpha_v11_sign_calibration(
                    v9_rows,
                    source_fit=source_fit,
                    outer_cutoff=cutoff,
                    config=v11_config,
                )
            try:
                for symbol, contract in zip(
                    prepared.train_symbols, contracts, strict=True
                ):
                    rows, forecast, state, _features, v7_targets = _target_bundle(
                        prepared,
                        resolved,
                        symbol,
                        contract,
                        source_config.target,
                    )
                    sample = prepared.samples[symbol]
                    _names, full_features, _available = _feature_surface(sample)
                    heads = v9_fit.predict_heads(full_features[rows])
                    control_path = v7_targets[
                        CausalAlphaV7Candidate.V6_CONTROL
                    ].v6_target_path
                    control = compile_causal_alpha_v11_target(
                        study_arm=None,
                        decision_indices=forecast.decision_indices,
                        head_predictions=heads,
                        one_way_cost_rates=control_path.one_way_cost_rates,
                        liquidity_weight_caps=control_path.liquidity_weight_caps,
                        risk_weight_caps=control_path.risk_weight_caps,
                        actionable_mask=control_path.actionable_mask,
                        source_forecast_digest=forecast.digest,
                        wave_fit_digest=v9_fit.digest,
                        v9_config=v9_config,
                        v11_config=v11_config,
                        initial_weight=control_path.initial_weight,
                    )
                    treatment = compile_causal_alpha_v11_target(
                        study_arm=arm,
                        decision_indices=forecast.decision_indices,
                        head_predictions=heads,
                        one_way_cost_rates=control_path.one_way_cost_rates,
                        liquidity_weight_caps=control_path.liquidity_weight_caps,
                        risk_weight_caps=control_path.risk_weight_caps,
                        actionable_mask=control_path.actionable_mask,
                        source_forecast_digest=forecast.digest,
                        wave_fit_digest=v9_fit.digest,
                        v9_config=v9_config,
                        v11_config=v11_config,
                        initial_weight=control_path.initial_weight,
                        sign_calibration=sign_calibration,
                    )
                    r21_control_leaf, r21_control = _load_r21_leaf(
                        r21_root,
                        candidate="v9_nonlinear_control",
                        symbol=symbol,
                        episode=contract.episode_index,
                        contract_digest=contract.digest,
                    )
                    r21_target = r21_control_leaf.get("target")
                    if not isinstance(r21_target, dict):
                        raise ValueError("V11 r21 control target is invalid")
                    expected_v6_digest = str(
                        r21_target.get("v6_target_path_digest", "")
                    )
                    if control.target.v6_target_path.digest != expected_v6_digest:
                        raise ValueError("V11 regenerated V9 target digest mismatch")
                    diagnostic_digest = _load_diagnostic_digest(
                        store,
                        symbol=symbol,
                        episode=contract.episode_index,
                        study_arm=arm,
                        r21_replay_digest=r21_control.digest,
                    )
                    if diagnostic_digest is None:
                        if (
                            r21_control.step_trace is None
                            or r21_control.lifecycle_trace is None
                        ):
                            raise ValueError("V11 r21 control lacks D1 trace evidence")
                        diagnostic = build_causal_alpha_v11_diagnostics(
                            symbol=symbol,
                            episode_id=str(contract.episode_index),
                            step_trace=r21_control.step_trace,
                            lifecycle_trace=r21_control.lifecycle_trace,
                            qualified_directions=control.fast_qualified_directions,
                            actionable_mask=control_path.actionable_mask,
                            labels_4h=np.asarray(sample.labels_4h)[rows],
                            one_way_cost_rates=control_path.one_way_cost_rates,
                            expected_target_digest=expected_v6_digest,
                            regenerated_target_digest=control.target.v6_target_path.digest,
                            policy_input_digest=control.digest,
                        )
                        _write_diagnostic(
                            store,
                            diagnostic,
                            study_arm=arm,
                            r21_replay_digest=r21_control.digest,
                        )
                        diagnostic_digest = diagnostic.digest
                    diagnostic_digests.append(diagnostic_digest)
                    diagnostic_paths.append(
                        _diagnostic_path(symbol, contract.episode_index)
                    )
                    if arm is CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING:
                        sizing_targets.append(treatment.target.v6_target_path.targets)
                        continue

                    v8_paths = causal_alpha_v8_target_paths_from_v7(
                        forecast=forecast,
                        v7_paths=v7_targets,
                        config=CausalAlphaV8TargetConfig(base=source_config.target),
                    )
                    cash_compat = _compatibility_target(
                        candidate=CausalAlphaV10Candidate.V8_ROBUST_CONTROL,
                        path=v8_paths[
                            CausalAlphaV8Candidate.ROBUST_CALIBRATED
                        ].v6_target_path,
                        source_forecast_digest=forecast.digest,
                        fit_digest=v9_fit.digest,
                    )
                    control_compat = _compatibility_target(
                        candidate=CausalAlphaV10Candidate.V9_NONLINEAR_CONTROL,
                        path=control.target.v6_target_path,
                        source_forecast_digest=forecast.digest,
                        fit_digest=v9_fit.digest,
                    )
                    if control_compat.digest != r21_target.get("artifact_digest"):
                        raise ValueError("V11 regenerated V9 wrapper digest mismatch")
                    treatment_compat = _compatibility_target(
                        candidate=CausalAlphaV10Candidate.HIERARCHICAL_WAVE,
                        path=treatment.target.v6_target_path,
                        source_forecast_digest=forecast.digest,
                        fit_digest=v9_fit.digest,
                        hierarchy_input_digest=treatment.digest,
                        hierarchy_reasons=treatment.policy_reasons,
                    )
                    candidate_specs = (
                        (
                            CausalAlphaV11Candidate.V8_CASH_SANITY,
                            cash_compat,
                            None,
                            None,
                            None,
                        ),
                        (
                            CausalAlphaV11Candidate.V9_CONTROL,
                            control_compat,
                            control,
                            CausalAlphaV11TracePolicy(control),
                            r21_control,
                        ),
                        (
                            CausalAlphaV11Candidate.TREATMENT,
                            treatment_compat,
                            treatment,
                            CausalAlphaV11TracePolicy(treatment),
                            None,
                        ),
                    )
                    for (
                        candidate,
                        compatibility,
                        compiled,
                        model,
                        reference,
                    ) in candidate_specs:
                        input_digest = (
                            compatibility.digest
                            if compiled is None
                            else compiled.digest
                        )
                        r21_digest = None if reference is None else reference.digest
                        metric = _load_metric(
                            store,
                            path=_leaf_path(candidate, symbol, contract.episode_index),
                            candidate=candidate,
                            study_arm=arm,
                            symbol=symbol,
                            episode=contract.episode_index,
                            contract_digest=contract.digest,
                            candidate_input_digest=input_digest,
                            r21_replay_digest=r21_digest,
                        )
                        if metric is None:
                            metric = _replay(
                                prepared=prepared,
                                resolved=resolved,
                                symbol=symbol,
                                contract=contract,
                                forecast=forecast,
                                state=state,
                                rows=rows,
                                target=compatibility,
                                model=model,
                                config_digest=config_digest,
                                boundaries=resolved.boundaries,
                                execution_contract=execution_contracts[symbol],
                            )
                            if reference is not None:
                                _require_control_equivalence(metric, reference)
                            _write_metric(
                                store,
                                candidate=candidate,
                                study_arm=arm,
                                symbol=symbol,
                                episode=contract.episode_index,
                                contract_digest=contract.digest,
                                candidate_input_digest=input_digest,
                                compatibility_target=compatibility,
                                v11_target=compiled,
                                metric=metric,
                                r21_replay_digest=r21_digest,
                            )
                        if candidate is CausalAlphaV11Candidate.V8_CASH_SANITY:
                            cash_metrics.append(metric)
                        elif candidate is CausalAlphaV11Candidate.V9_CONTROL:
                            control_metrics.append(metric)
                        else:
                            treatment_metrics.append(metric)
                _progress(stage="v11-selection", cutoff=cutoff, candidate=arm.value)
            finally:
                del resolved, v9_fit, sign_calibration
                gc.collect()

        diagnostic_index = _diagnostic_index(
            store,
            study_arm=arm,
            scope_paths=tuple(diagnostic_paths),
        )
        sizing = None
        if arm is CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING:
            sizing = evaluate_v11_sizing_feasibility(
                targets=np.concatenate(sizing_targets),
                entry_threshold=0.1,
                no_trade_band=0.05,
            )
            store.write_leaf(
                "sizing-feasibility.json",
                _artifact(
                    {
                        "feasibility": sizing.to_payload(),
                        "feasibility_digest": sizing.digest,
                        "schema_version": "causal_alpha_v11_sizing_feasibility_leaf_v1",
                        "study_arm": arm.value,
                    }
                ),
            )
        selection = evaluate_causal_alpha_v11_selection(
            study_arm=arm,
            cash_metrics=tuple(cash_metrics),
            control_metrics=tuple(control_metrics),
            treatment_metrics=tuple(treatment_metrics),
            expected_symbols=prepared.train_symbols,
            v11_config_digest=v11_config.digest,
            diagnostic_digests=tuple(diagnostic_digests),
            sizing_feasibility=sizing,
        )
        store.write_leaf(
            "selection/evidence.json",
            store.envelope(
                schema_version="causal_alpha_v11_selection_envelope_v1",
                evidence_digest=selection.digest,
                payload=selection.to_payload(),
            ),
        )
        status = (
            "preflight_stopped"
            if selection.source_v8 is None
            else "selection_passed"
            if selection.passed
            else "selection_rejected"
        )
        store.write_leaf(
            "result.json",
            _artifact(
                {
                    "diagnostic_index_digest": diagnostic_index["artifact_digest"],
                    "evidence_digest": selection.digest,
                    "promotion_eligible": False,
                    "r21_signal_evidence_digest": signal_digest,
                    "schema_version": _RESULT_SCHEMA,
                    "status": status,
                    "study_arm": arm.value,
                }
            ),
        )
        return selection


__all__ = [
    "causal_alpha_v11_stage_config_digest",
    "run_causal_alpha_v11_selection",
]
