"""Evaluation-only C3 lifecycle over published walk-forward evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.run_manifest import (
    WalkForwardRunManifest,
    validate_walk_forward_run_directory,
)
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.causal_scenario_artifact import (
    load_causal_scenario_value_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_adverse import C3AdverseFoldEvidence
from trade_rl.evaluation.causal_scenario_c3_adverse_source import (
    load_c3_source_adverse_evidence,
)
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    C3ReplayIdentity,
    CausalScenarioC3Config,
    PerfectInformationComparison,
    PerfectInformationComparisonReason,
    PerfectInformationComparisonStatus,
    RealizedPolicyOutcome,
)
from trade_rl.evaluation.causal_scenario_c3_decision_artifact import (
    write_c3_decision_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_prediction import (
    build_c3_prediction_evidence,
)
from trade_rl.evaluation.causal_scenario_c3_runner import (
    build_persisted_scenario_decision,
)
from trade_rl.workflows.causal_scenario.c3 import (
    C3BatchQuery,
    C3BatchResult,
    execute_c3_batch,
)
from trade_rl.workflows.causal_scenario.library_artifact import (
    load_causal_scenario_library_artifact,
)

C3_EVALUATION_REQUEST_SCHEMA: Final = "causal_scenario_c3_evaluation_request_v2"
C3_EVALUATION_RESULT_SCHEMA: Final = "causal_scenario_c3_evaluation_result_v2"
PRODUCTION_STATUS: Final = "NO-GO"


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _fields(payload: dict[str, object], expected: set[str], *, field: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{field} field closure mismatch")


def _relative_path(root: Path, value: object, *, field: str) -> Path:
    raw = _string(value, field=field)
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{field} must be a safe relative path")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} must not contain symbolic links")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError(f"{field} escapes the request root")
    return resolved


def _float_vector(value: object, *, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).copy(order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _load_outcome(value: object, *, field: str) -> RealizedPolicyOutcome:
    payload = _object(value, field=field)
    _fields(
        payload,
        {
            "borrow_paid",
            "cancel_replace_events",
            "fees",
            "fill_count",
            "fill_ratio",
            "filled_turnover",
            "funding_paid",
            "gross_log_return",
            "impact_cost",
            "max_drawdown",
            "outcome_digest",
            "pending_order_events",
            "policy_kind",
            "schema_version",
            "spread_cost",
            "terminal_equity",
            "termination_reason",
        },
        field=field,
    )
    return RealizedPolicyOutcome(
        policy_kind=_string(payload["policy_kind"], field=f"{field}.policy_kind"),
        gross_log_return=_number(
            payload["gross_log_return"], field=f"{field}.gross_log_return"
        ),
        filled_turnover=_number(
            payload["filled_turnover"], field=f"{field}.filled_turnover"
        ),
        fees=_number(payload["fees"], field=f"{field}.fees"),
        spread_cost=_number(payload["spread_cost"], field=f"{field}.spread_cost"),
        impact_cost=_number(payload["impact_cost"], field=f"{field}.impact_cost"),
        funding_paid=_number(payload["funding_paid"], field=f"{field}.funding_paid"),
        borrow_paid=_number(payload["borrow_paid"], field=f"{field}.borrow_paid"),
        fill_ratio=_number(payload["fill_ratio"], field=f"{field}.fill_ratio"),
        fill_count=_integer(payload["fill_count"], field=f"{field}.fill_count"),
        pending_order_events=_integer(
            payload["pending_order_events"], field=f"{field}.pending_order_events"
        ),
        cancel_replace_events=_integer(
            payload["cancel_replace_events"], field=f"{field}.cancel_replace_events"
        ),
        max_drawdown=_number(payload["max_drawdown"], field=f"{field}.max_drawdown"),
        terminal_equity=_number(
            payload["terminal_equity"], field=f"{field}.terminal_equity"
        ),
        termination_reason=_string(
            payload["termination_reason"], field=f"{field}.termination_reason"
        ),
        outcome_digest=_string(
            payload["outcome_digest"], field=f"{field}.outcome_digest"
        ),
        schema_version=_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
    )


def _load_perfect_information(
    value: object, *, field: str
) -> PerfectInformationComparison:
    payload = _object(value, field=field)
    _fields(
        payload,
        {
            "bound_log_return",
            "causal_log_return",
            "compatibility_evidence_digest",
            "gap",
            "reason",
            "status",
        },
        field=field,
    )
    status = PerfectInformationComparisonStatus(
        _string(payload["status"], field=f"{field}.status")
    )
    reason = PerfectInformationComparisonReason(
        _string(payload["reason"], field=f"{field}.reason")
    )
    if status is PerfectInformationComparisonStatus.COMPARABLE:
        if reason is not PerfectInformationComparisonReason.DOMINANCE_VERIFIED:
            raise ValueError(f"{field} comparable reason mismatch")
        evidence = _string(
            payload["compatibility_evidence_digest"],
            field=f"{field}.compatibility_evidence_digest",
        )
        bound = _number(payload["bound_log_return"], field=f"{field}.bound_log_return")
        causal = _number(
            payload["causal_log_return"], field=f"{field}.causal_log_return"
        )
        gap = _number(payload["gap"], field=f"{field}.gap")
        if not math.isclose(gap, bound - causal, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{field} gap does not match comparable returns")
        return PerfectInformationComparison.comparable(
            bound_log_return=bound,
            causal_log_return=causal,
            compatibility_evidence_digest=evidence,
        )
    if any(
        payload[name] is not None
        for name in ("bound_log_return", "causal_log_return", "gap")
    ):
        raise ValueError(f"{field} non-comparable returns must be null")
    evidence_raw = payload["compatibility_evidence_digest"]
    optional_evidence = (
        None
        if evidence_raw is None
        else _string(evidence_raw, field=f"{field}.compatibility_evidence_digest")
    )
    if status is PerfectInformationComparisonStatus.NOT_EVALUATED:
        if reason is not PerfectInformationComparisonReason.NOT_EVALUATED:
            raise ValueError(f"{field} not-evaluated reason mismatch")
        if optional_evidence is not None:
            raise ValueError(f"{field} not-evaluated evidence must be null")
        return PerfectInformationComparison.not_evaluated()
    return PerfectInformationComparison.not_comparable(
        reason,
        compatibility_evidence_digest=optional_evidence,
    )


def _action_key(policy_kind: str, raw_residual: np.ndarray) -> str:
    return content_digest(
        {
            "policy_kind": policy_kind,
            "raw_residual": raw_residual.tolist(),
            "schema_version": "causal_scenario_c3_artifact_replay_key_v1",
        }
    )


class ArtifactBackedC3Replay:
    """Replay capability backed by immutable stateful-execution outcomes."""

    def __init__(
        self,
        identity: C3ReplayIdentity,
        outcomes: dict[str, RealizedPolicyOutcome],
    ) -> None:
        self.identity = identity
        self._outcomes = dict(outcomes)

    def clone_for_replay(self) -> ArtifactBackedC3Replay:
        return ArtifactBackedC3Replay(self.identity, self._outcomes)

    def run(
        self,
        raw_residual: np.ndarray,
        *,
        horizon_decisions: int,
        zero_residual_after_first: bool,
        policy_kind: str,
    ) -> RealizedPolicyOutcome:
        if (
            horizon_decisions
            != self.identity.realized_stop_index - self.identity.query_index
        ):
            raise ValueError("artifact replay horizon does not match identity")
        if zero_residual_after_first is not True:
            raise ValueError("artifact replay requires one-shot residual semantics")
        action = _float_vector(raw_residual, field="raw_residual")
        key = _action_key(policy_kind, action)
        outcome = self._outcomes.get(key)
        if outcome is None:
            raise ValueError("artifact replay outcome is missing")
        if outcome.policy_kind != policy_kind:
            raise ValueError("artifact replay policy identity mismatch")
        return outcome


def _load_replay_outcomes(
    value: object,
    *,
    field: str,
    action_dimension: int,
) -> dict[str, RealizedPolicyOutcome]:
    result: dict[str, RealizedPolicyOutcome] = {}
    for index, item in enumerate(_list(value, field=field)):
        item_field = f"{field}[{index}]"
        payload = _object(item, field=item_field)
        _fields(payload, {"outcome", "raw_residual"}, field=item_field)
        raw = _float_vector(payload["raw_residual"], field=f"{item_field}.raw_residual")
        if raw.shape != (action_dimension,) or np.any(np.abs(raw) > 1.0):
            raise ValueError(f"{item_field}.raw_residual has an invalid action shape")
        outcome = _load_outcome(payload["outcome"], field=f"{item_field}.outcome")
        key = _action_key(outcome.policy_kind, raw)
        if key in result:
            raise ValueError("artifact replay outcome keys must be unique")
        result[key] = outcome
    if not result:
        raise ValueError("artifact replay outcomes must not be empty")
    return result


def _load_config(value: object) -> CausalScenarioC3Config:
    payload = _object(value, field="config")
    _fields(
        payload,
        {
            "bootstrap_block_days",
            "bootstrap_resamples",
            "horizon_decisions",
            "random_comparator_count",
            "ranking_tolerance",
            "required_folds",
            "required_selection_days",
            "scenario_count",
            "schema_version",
        },
        field="config",
    )
    return CausalScenarioC3Config(
        horizon_decisions=_integer(
            payload["horizon_decisions"],
            field="config.horizon_decisions",
            positive=True,
        ),
        scenario_count=_integer(
            payload["scenario_count"], field="config.scenario_count", positive=True
        ),
        random_comparator_count=_integer(
            payload["random_comparator_count"],
            field="config.random_comparator_count",
            positive=True,
        ),
        bootstrap_block_days=_integer(
            payload["bootstrap_block_days"],
            field="config.bootstrap_block_days",
            positive=True,
        ),
        ranking_tolerance=_number(
            payload["ranking_tolerance"], field="config.ranking_tolerance"
        ),
        required_folds=_integer(
            payload["required_folds"], field="config.required_folds", positive=True
        ),
        required_selection_days=_integer(
            payload["required_selection_days"],
            field="config.required_selection_days",
            positive=True,
        ),
        bootstrap_resamples=_integer(
            payload["bootstrap_resamples"],
            field="config.bootstrap_resamples",
            positive=True,
        ),
        schema_version=_string(
            payload["schema_version"], field="config.schema_version"
        ),
    )


def _canonical_object(path: Path, *, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} file is missing: {path}")
    raw_bytes = path.read_bytes()
    try:
        raw = _object(json.loads(raw_bytes.decode("utf-8")), field=field)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is invalid JSON") from error
    if canonical_json_bytes(raw) != raw_bytes:
        raise ValueError(f"{field} must be canonical JSON")
    return raw


def _walk_forward_payload(
    source_root: Path, manifest: WalkForwardRunManifest
) -> dict[str, object]:
    raw = _canonical_object(source_root / "walk-forward.json", field="walk-forward")
    if raw.get("dataset_id") != manifest.dataset_id:
        raise ValueError("walk-forward dataset identity does not match manifest")
    if raw.get("evaluation_digest") != manifest.evaluation_digest:
        raise ValueError("walk-forward evaluation identity does not match manifest")
    folds = raw.get("folds")
    if not isinstance(folds, list) or len(folds) != manifest.fold_count:
        raise ValueError("walk-forward fold count does not match manifest")
    return raw


def _walk_forward_config_payload(
    source_root: Path, manifest: WalkForwardRunManifest
) -> dict[str, object]:
    raw = _canonical_object(
        source_root / "walk-forward-config.json",
        field="walk-forward-config",
    )
    if content_digest(raw) != manifest.workflow_config_digest:
        raise ValueError("walk-forward config identity does not match manifest")
    return raw


def _source_fold_map(walk_forward: dict[str, object]) -> dict[int, dict[str, object]]:
    source_folds: dict[int, dict[str, object]] = {}
    for position, value in enumerate(
        _list(walk_forward["folds"], field="walk-forward.folds")
    ):
        item = _object(value, field=f"walk-forward.folds[{position}]")
        fold_index = _integer(
            item.get("fold_index"), field=f"walk-forward.folds[{position}].fold_index"
        )
        if fold_index in source_folds:
            raise ValueError("source walk-forward fold indices must be unique")
        source_folds[fold_index] = item
    return source_folds


def _require_execution_scenarios(
    scenarios: set[str], *, required_scenario: str
) -> None:
    if "nominal" not in scenarios:
        raise ValueError("each C3 request fold requires a nominal execution scenario")
    if required_scenario not in scenarios:
        raise ValueError(
            "each C3 request fold requires the source-declared required adverse scenario"
        )


@dataclass(frozen=True, slots=True)
class C3EvaluationResult:
    source_run_digest: str
    request_digest: str
    batch: C3BatchResult
    schema_version: str = C3_EVALUATION_RESULT_SCHEMA
    production_status: str = PRODUCTION_STATUS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_run_digest",
            require_sha256(self.source_run_digest, field="source_run_digest"),
        )
        object.__setattr__(
            self,
            "request_digest",
            require_sha256(self.request_digest, field="request_digest"),
        )
        if not isinstance(self.batch, C3BatchResult):
            raise ValueError("batch must be C3BatchResult")
        if self.schema_version != C3_EVALUATION_RESULT_SCHEMA:
            raise ValueError("unsupported C3 evaluation result schema")
        if self.production_status != PRODUCTION_STATUS:
            raise ValueError("C3 evaluation production status must remain NO-GO")


def execute_c3_evaluation_request(
    request_path: str | Path,
    *,
    output_root: str | Path,
) -> C3EvaluationResult:
    """Execute C3 from a published walk-forward run and frozen C1/C2 evidence."""

    request_file = Path(request_path)
    raw = _canonical_object(request_file, field="request")
    _fields(
        raw,
        {"config", "folds", "schema_version", "source_walk_forward_run"},
        field="request",
    )
    if raw["schema_version"] != C3_EVALUATION_REQUEST_SCHEMA:
        raise ValueError("unsupported C3 evaluation request schema")
    request_root = request_file.parent
    config = _load_config(raw["config"])
    source_root = _relative_path(
        request_root,
        raw["source_walk_forward_run"],
        field="source_walk_forward_run",
    )
    source_manifest = validate_walk_forward_run_directory(source_root)
    walk_forward = _walk_forward_payload(source_root, source_manifest)
    walk_forward_config = _walk_forward_config_payload(source_root, source_manifest)
    source_folds = _source_fold_map(walk_forward)
    source_adverse = load_c3_source_adverse_evidence(
        source_root,
        walk_forward_config=walk_forward_config,
        source_folds=source_folds,
        dataset_id=source_manifest.dataset_id,
    )

    destination = Path(output_root)
    batch_queries: list[C3BatchQuery] = []
    fold_selection_days: dict[str, int] = {}
    required_adverse_evidence: dict[str, C3AdverseFoldEvidence] = {}
    seen_fold_indices: set[int] = set()
    seen_fold_ids: set[str] = set()
    for fold_position, raw_fold in enumerate(_list(raw["folds"], field="folds")):
        field = f"folds[{fold_position}]"
        fold = _object(raw_fold, field=field)
        _fields(
            fold,
            {"fold_digest", "fold_id", "fold_index", "library", "queries"},
            field=field,
        )
        fold_id = _string(fold["fold_id"], field=f"{field}.fold_id")
        fold_index = _integer(fold["fold_index"], field=f"{field}.fold_index")
        if fold_id in seen_fold_ids:
            raise ValueError("C3 request fold IDs must be unique")
        if fold_index in seen_fold_indices:
            raise ValueError("C3 request fold indices must be unique")
        seen_fold_ids.add(fold_id)
        seen_fold_indices.add(fold_index)
        source_fold = source_folds.get(fold_index)
        if source_fold is None:
            raise ValueError("C3 request fold is absent from source walk-forward run")
        train_range = source_fold.get("train_range")
        test_range = source_fold.get("test_range")
        if (
            not isinstance(train_range, list)
            or len(train_range) != 2
            or not isinstance(test_range, list)
            or len(test_range) != 2
        ):
            raise ValueError("source walk-forward fold ranges are invalid")
        train_start = _integer(train_range[0], field="source.train_start")
        train_stop = _integer(train_range[1], field="source.train_stop", positive=True)
        test_start = _integer(test_range[0], field="source.test_start")
        test_stop = _integer(test_range[1], field="source.test_stop", positive=True)
        library_root = _relative_path(
            request_root, fold["library"], field=f"{field}.library"
        )
        library = load_causal_scenario_library_artifact(library_root)
        if library.dataset_id != source_manifest.dataset_id:
            raise ValueError("C2 library dataset does not match source run")
        if (library.train_start, library.train_stop) != (train_start, train_stop):
            raise ValueError("C2 library train range does not match source fold")
        if (
            library.config.horizon_decisions != config.horizon_decisions
            or library.config.scenario_count != config.scenario_count
        ):
            raise ValueError("C2 library configuration does not match C3")
        fold_digest = require_sha256(
            _string(fold["fold_digest"], field=f"{field}.fold_digest"),
            field=f"{field}.fold_digest",
        )
        try:
            fold_selection_days[fold_id] = source_adverse.selection_days_by_fold[
                fold_index
            ]
            required_adverse_evidence[fold_id] = source_adverse.by_fold_index[
                fold_index
            ]
        except KeyError as error:
            raise ValueError(
                "source adverse evidence is missing for a C3 fold"
            ) from error
        scenarios: set[str] = set()
        for query_position, raw_query in enumerate(
            _list(fold["queries"], field=f"{field}.queries")
        ):
            query_field = f"{field}.queries[{query_position}]"
            query = _object(raw_query, field=query_field)
            _fields(
                query,
                {
                    "execution_scenario",
                    "outcomes",
                    "perfect_information",
                    "ppo_mean_action",
                    "value_artifact",
                },
                field=query_field,
            )
            value_root = _relative_path(
                request_root,
                query["value_artifact"],
                field=f"{query_field}.value_artifact",
            )
            value_result = load_causal_scenario_value_artifact(value_root)
            if value_result.dataset_id != source_manifest.dataset_id:
                raise ValueError("C1 value dataset does not match source run")
            if value_result.fold_digest != fold_digest:
                raise ValueError("C1 value fold identity does not match request")
            if (value_result.train_start, value_result.train_stop) != (
                train_start,
                train_stop,
            ):
                raise ValueError("C1 value train range does not match source fold")
            if not test_start <= value_result.query_index < test_stop:
                raise ValueError("C1 query lies outside the source fold test range")
            if value_result.query_index + config.horizon_decisions > test_stop:
                raise ValueError(
                    "C1 realized horizon exceeds the source fold test range"
                )
            if value_result.scenario_library_digest != library.library_digest:
                raise ValueError("C1 value does not bind the frozen C2 library")
            if (
                value_result.config.scenario_count != config.scenario_count
                or value_result.config.horizon_decisions != config.horizon_decisions
            ):
                raise ValueError("C1 evaluator configuration does not match C3")
            decision = build_persisted_scenario_decision(value_result)
            decision_root = destination / "decisions" / decision.decision_digest
            write_c3_decision_artifact(decision_root, decision)
            prediction = build_c3_prediction_evidence(value_result)
            action = _float_vector(
                query["ppo_mean_action"], field=f"{query_field}.ppo_mean_action"
            )
            if action.shape != (value_result.config.action_dimension,):
                raise ValueError("PPO mean action dimension does not match C1 value")
            scenario = _string(
                query["execution_scenario"],
                field=f"{query_field}.execution_scenario",
            )
            scenarios.add(scenario)
            outcomes = _load_replay_outcomes(
                query["outcomes"],
                field=f"{query_field}.outcomes",
                action_dimension=value_result.config.action_dimension,
            )
            replay = ArtifactBackedC3Replay(decision.replay_identity, outcomes)
            batch_queries.append(
                C3BatchQuery(
                    fold_id=fold_id,
                    decision_root=decision_root,
                    replay=replay,
                    ppo_mean_action=action,
                    prediction_evidence=prediction,
                    execution_scenario=scenario,
                    perfect_information=_load_perfect_information(
                        query["perfect_information"],
                        field=f"{query_field}.perfect_information",
                    ),
                )
            )
        _require_execution_scenarios(
            scenarios,
            required_scenario=source_adverse.required_scenario,
        )

    if not batch_queries:
        raise ValueError("C3 evaluation request contains no queries")
    batch = execute_c3_batch(
        tuple(batch_queries),
        output_root=destination,
        fold_selection_days=fold_selection_days,
        required_adverse_evidence=required_adverse_evidence,
        config=config,
    )
    return C3EvaluationResult(
        source_run_digest=source_manifest.digest,
        request_digest=content_digest(raw),
        batch=batch,
    )


__all__ = [
    "ArtifactBackedC3Replay",
    "C3_EVALUATION_REQUEST_SCHEMA",
    "C3_EVALUATION_RESULT_SCHEMA",
    "C3EvaluationResult",
    "execute_c3_evaluation_request",
]
