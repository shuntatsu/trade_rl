"""Sealed pre-result contract for calibrated causal-capacity robustness."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from trade_rl._validation import (
    require_aware_datetime,
    require_git_sha,
    require_non_empty,
    require_sha256,
)
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "calibrated_capacity_robustness_prereg_v2"
_SOURCE_BASELINE_ARTIFACT_ID = 10301701698
_SOURCE_BASELINE_ARTIFACT_DIGEST = (
    "f814fe4e205f8714c4344238911aae16e89ce0279908265feb1fdc85069b2a0a"
)
_SOURCE_BASELINE_EVIDENCE_FINGERPRINT = (
    "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
)
_SOURCE_DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
_SOURCE_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
_SOURCE_STUDY_DIGEST = (
    "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
)
_SOURCE_IMPLEMENTATION_SHA = "5605bfee8df4e38b6c9a26621a132a1e5453b9ff"
_SOURCE_IMPLEMENTATION_DIGEST = (
    "4e74c99ea86f347e024d701396ac58f24eb49affb30a6c283f385de2a839f8f7"
)
_SOURCE_RUNTIME_ENVIRONMENT_DIGEST = (
    "6dbc9cffd844837e17741ae30681f09a6d84b0a49dec011a4fc328f97ade1d85"
)
_SOURCE_ELIGIBILITY_RUN_ID = 34794576828
_SOURCE_ELIGIBILITY_ARTIFACT_ID = 10329551558
_SOURCE_ELIGIBILITY_ARTIFACT_DIGEST = (
    "907599e2330e52fe832872b3b11a5f0ad62316fe4a170acd4453ab39ba1a0800"
)
_SOURCE_ELIGIBILITY_DIGEST = (
    "cb92548157200067907abdb221bcfb9cea5fa527dd22193ae065fcad9c4fdbe7"
)

_CAPACITY_PROTOCOL_DIGEST = (
    "5fb013fb0a3d717846a701b23d2f4bfca8e742ebaef0e6f564a331053f2ed071"
)
_CAPACITY_RESULT_DIGEST = (
    "a5490f9209b16a8ba94f11506b83aa45ce44c7d61d13234480e2362f6df89809"
)
_CAPACITY_RESULT_JSON_SHA256 = (
    "397004e6a1a9c3b30b32212066ac41366f7af74edc0ae23372f847e50c6c5a4e"
)
_CAPACITY_RESULT_ARTIFACT_ID = 10320428830
_CAPACITY_RESULT_ARTIFACT_DIGEST = (
    "810e792636a696a0bc11e22ab53a21f378aaa0bfaeab8c3411123e9c6d9a5c35"
)
_CAPACITY_VERIFIER_RUN_ID = 34767391268
_CAPACITY_VERIFIER_ARTIFACT_ID = 10320394746
_CAPACITY_VERIFIER_ARTIFACT_DIGEST = (
    "31702e68dddd9b4844eee6dc3b5b47ab4aa824d50200509a7bcdbf009c652bd1"
)

_CAUSAL_CAPACITY_PR = 531
_CAUSAL_CAPACITY_HEAD_SHA = "db8193c81dd0ca334a15ffc1895d75884238f8a9"
_IDENTITY_LAYER_PR = 540
_IDENTITY_LAYER_HEAD_SHA = "15bf6546996dc969d4d6e946261d73e3d43336e6"
_IDENTITY_LAYER_CI_RUN_ID = 34793925990
_SUCCESSOR_BUNDLE_RUN_ID = 34794384559
_SUCCESSOR_BUNDLE_ARTIFACT_ID = 10329107309
_SUCCESSOR_BUNDLE_ARTIFACT_DIGEST = (
    "6e8bff792fb905bf93fbf591e2c9cf18c2b28e6a1396f4e1a88562e83a65986d"
)
_SUCCESSOR_DATASET_ID = (
    "6a9d6066fe8f92d46fe57a1d8172a60092568e625b2e843dc7a0c3930d79de86"
)
_SUCCESSOR_DATASET_ARTIFACT_DIGEST = (
    "9a3f01bc4608c256b5c448e5cee123c4653fe02060040dc7909b4eb1c6c3e30f"
)
_SUCCESSOR_STUDY_DIGEST = (
    "baaf25c93f37e409220d4413ddd9216c2afbf29ba1ce344310d43d7efa74e7eb"
)
_SUCCESSOR_DATASET_TREE_DIGEST = (
    "3bd6964426f983c8726a26dd1298df785726c9b66771e445253b38a8bb3dc9af"
)
_SUCCESSOR_STUDY_TREE_DIGEST = (
    "a579b72e93d39b59fe1e232fd658bbc016fdecf4b5da6c683080ed2b5ea49bcc"
)
_SUCCESSOR_MATERIALIZATION_INDEX_SHA256 = (
    "8fae0e0e9ae0fdcc706d63b5159d825b052592b427d483fa8cf423c8dd880efc"
)
_SUCCESSOR_RUNTIME_ENVIRONMENT_DIGEST = (
    "6dbc9cffd844837e17741ae30681f09a6d84b0a49dec011a4fc328f97ade1d85"
)
_SUCCESSOR_FRESH_VERIFICATION_ARTIFACT_ID = 10329905111
_SUCCESSOR_FRESH_VERIFICATION_ARTIFACT_DIGEST = (
    "9769715e5d776d01791c6a7b2acbdfd4a52c97030ddb57875f448f0705b36f0f"
)
_SUCCESSOR_EXECUTION_OVERLAY = (
    "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity"
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_CAPACITY_CAPS = (
    0.0021629560553901974,
    0.002044685341258238,
    0.002184898995567895,
    0.0020480213652913385,
    0.002346308308284808,
)
_FIT_CUTOFF = datetime(2023, 1, 1, tzinfo=UTC)
_EVALUATION_START = datetime(2023, 1, 1, tzinfo=UTC)
_EVALUATION_STOP_EXCLUSIVE = datetime(2025, 1, 1, tzinfo=UTC)
_PPO_SEEDS = (0, 1, 2, 3, 4)
_CORE_STRATEGIES = ("trend", "mean_reversion", "ridge24", "lightgbm24")
_CONTROL_STRATEGIES = ("cash", "constant_long", "constant_short")
_SOURCE_CORE_TOTAL_RETURNS = (
    (
        "trend",
        (
            -0.4693596383984787,
            -0.6745119541152158,
            -0.6985296047775885,
            -0.6610604847435877,
            -0.6995926449531766,
        ),
    ),
    (
        "mean_reversion",
        (
            -0.3865837097474639,
            -0.19975352370855237,
            -0.08364864809522887,
            -0.3984753309722041,
            -0.3792367092172487,
        ),
    ),
    (
        "ridge24",
        (
            -0.3817900601958202,
            -0.28212108986352513,
            -0.2640984345173577,
            -0.7338319054368865,
            0.2262786724769308,
        ),
    ),
    (
        "lightgbm24",
        (
            -0.54697432035348,
            -0.26915263313070426,
            0.10253875097178011,
            -0.11188398892435136,
            0.8215258151078024,
        ),
    ),
)
_SOURCE_PROFITABLE_CORE_STRATEGIES: tuple[str, ...] = ()
_PRE_SUCCESSOR_SUITE_STATUS = "NO_PROFITABLE_CORE_BASELINE"
_STAGE_B_ROLE = "diagnostic_only"
_INTEGRATION_GATES = (
    "pr_531_merged",
    "pr_540_merged_after_531",
    "post_merge_main_ci_green",
    "merged_tree_matches_verified_identity_head",
    "successor_bundle_reverified",
)

_SOURCE_PROFITABILITY_REQUIRED_POSITIVE_SYMBOLS = 5
_ROBUST_POSITIVE_REQUIRED_POSITIVE_SYMBOLS = 5
_EXECUTION_FRAGILE_MAX_POSITIVE_SYMBOLS = 3
_STAGE_A_FAILURE_STATUS = "INVALID_IMPLEMENTATION_DRIFT"
_EXPERIMENT_0004_ISSUE = 529


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _datetime(value: datetime, *, field: str) -> datetime:
    return require_aware_datetime(value, field=field).astimezone(UTC)


def _parse_datetime(value: object, *, field: str) -> datetime:
    text = (
        require_non_empty(cast(str, value), field=field)
        if isinstance(value, str)
        else ""
    )
    if not text:
        raise ValueError(f"{field} must be an ISO datetime")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO datetime") from error
    return _datetime(parsed, field=field)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result = tuple(
        require_non_empty(item, field=field) for item in value if isinstance(item, str)
    )
    if len(result) != len(value):
        raise ValueError(f"{field} must contain only strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _int_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result = tuple(_positive_or_zero_int(item, field=field) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _positive_or_zero_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _float_tuple(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return tuple(_finite_float(item, field=field) for item in value)


def _parse_source_returns(value: object) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if not isinstance(value, list):
        raise ValueError("source_core_total_returns must be an array")
    result: list[tuple[str, tuple[float, ...]]] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"strategy", "total_returns"}:
            raise ValueError(
                "source_core_total_returns entry keys differ from contract"
            )
        strategy = entry.get("strategy")
        if not isinstance(strategy, str):
            raise ValueError("source return strategy must be a string")
        result.append(
            (
                require_non_empty(strategy, field="source return strategy"),
                _float_tuple(entry.get("total_returns"), field="source total returns"),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CapacityRobustnessProtocol:
    """Immutable pre-result authority for the one-shot capacity robustness test."""

    source_baseline_artifact_id: int
    source_baseline_artifact_digest: str
    source_baseline_evidence_fingerprint: str
    source_dataset_id: str
    source_dataset_artifact_digest: str
    source_study_digest: str
    source_implementation_sha: str
    source_implementation_digest: str
    source_runtime_environment_digest: str
    source_eligibility_run_id: int
    source_eligibility_artifact_id: int
    source_eligibility_artifact_digest: str
    source_eligibility_digest: str
    capacity_protocol_digest: str
    capacity_result_digest: str
    capacity_result_json_sha256: str
    capacity_result_artifact_id: int
    capacity_result_artifact_digest: str
    capacity_verifier_run_id: int
    capacity_verifier_artifact_id: int
    capacity_verifier_artifact_digest: str
    causal_capacity_pr: int
    causal_capacity_head_sha: str
    identity_layer_pr: int
    identity_layer_head_sha: str
    identity_layer_ci_run_id: int
    successor_bundle_run_id: int
    successor_bundle_artifact_id: int
    successor_bundle_artifact_digest: str
    successor_dataset_id: str
    successor_dataset_artifact_digest: str
    successor_study_digest: str
    successor_dataset_tree_digest: str
    successor_study_tree_digest: str
    successor_materialization_index_sha256: str
    successor_runtime_environment_digest: str
    successor_fresh_verification_artifact_id: int
    successor_fresh_verification_artifact_digest: str
    successor_execution_overlay: str
    symbols: tuple[str, ...]
    capacity_caps: tuple[float, ...]
    fit_cutoff: datetime
    evaluation_start: datetime
    evaluation_stop_exclusive: datetime
    ppo_seeds: tuple[int, ...]
    core_strategies: tuple[str, ...]
    control_strategies: tuple[str, ...]
    ppo_disclosed: bool
    ppo_in_formal_decision: bool
    source_core_total_returns: tuple[tuple[str, tuple[float, ...]], ...]
    source_profitable_core_strategies: tuple[str, ...]
    pre_successor_suite_status: str
    stage_b_role: str
    integration_gates: tuple[str, ...]
    stage_a_required: bool
    stage_a_failure_status: str
    source_profitability_required_positive_symbols: int
    robust_positive_required_positive_symbols: int
    execution_fragile_max_positive_symbols: int
    robust_positive_requires_positive_median: bool
    integration_gate_satisfied: bool
    execution_authorized: bool
    stage_a_executed: bool
    stage_b_executed: bool
    successor_pnl_inspected: bool
    experiment_0004_issue: int
    experiment_0004_must_not_influence: bool
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _positive_int(
            self.source_baseline_artifact_id,
            field="source_baseline_artifact_id",
        )
        require_sha256(
            self.source_baseline_artifact_digest,
            field="source_baseline_artifact_digest",
        )
        require_sha256(
            self.source_baseline_evidence_fingerprint,
            field="source_baseline_evidence_fingerprint",
        )
        require_sha256(self.source_dataset_id, field="source_dataset_id")
        require_sha256(
            self.source_dataset_artifact_digest,
            field="source_dataset_artifact_digest",
        )
        require_sha256(
            self.source_study_digest,
            field="source_study_digest",
        )
        require_git_sha(
            self.source_implementation_sha,
            field="source_implementation_sha",
        )
        require_sha256(
            self.source_implementation_digest,
            field="source_implementation_digest",
        )
        require_sha256(
            self.source_runtime_environment_digest,
            field="source_runtime_environment_digest",
        )
        _positive_int(
            self.source_eligibility_run_id,
            field="source_eligibility_run_id",
        )
        _positive_int(
            self.source_eligibility_artifact_id,
            field="source_eligibility_artifact_id",
        )
        require_sha256(
            self.source_eligibility_artifact_digest,
            field="source_eligibility_artifact_digest",
        )
        require_sha256(
            self.source_eligibility_digest,
            field="source_eligibility_digest",
        )
        require_sha256(
            self.capacity_protocol_digest,
            field="capacity_protocol_digest",
        )
        require_sha256(
            self.capacity_result_digest,
            field="capacity_result_digest",
        )
        require_sha256(
            self.capacity_result_json_sha256,
            field="capacity_result_json_sha256",
        )
        _positive_int(
            self.capacity_result_artifact_id,
            field="capacity_result_artifact_id",
        )
        require_sha256(
            self.capacity_result_artifact_digest,
            field="capacity_result_artifact_digest",
        )
        _positive_int(
            self.capacity_verifier_run_id,
            field="capacity_verifier_run_id",
        )
        _positive_int(
            self.capacity_verifier_artifact_id,
            field="capacity_verifier_artifact_id",
        )
        require_sha256(
            self.capacity_verifier_artifact_digest,
            field="capacity_verifier_artifact_digest",
        )
        _positive_int(self.causal_capacity_pr, field="causal_capacity_pr")
        require_git_sha(
            self.causal_capacity_head_sha,
            field="causal_capacity_head_sha",
        )
        _positive_int(self.identity_layer_pr, field="identity_layer_pr")
        require_git_sha(
            self.identity_layer_head_sha,
            field="identity_layer_head_sha",
        )
        _positive_int(
            self.identity_layer_ci_run_id,
            field="identity_layer_ci_run_id",
        )
        _positive_int(
            self.successor_bundle_run_id,
            field="successor_bundle_run_id",
        )
        _positive_int(
            self.successor_bundle_artifact_id,
            field="successor_bundle_artifact_id",
        )
        require_sha256(
            self.successor_bundle_artifact_digest,
            field="successor_bundle_artifact_digest",
        )
        require_sha256(
            self.successor_dataset_id,
            field="successor_dataset_id",
        )
        require_sha256(
            self.successor_dataset_artifact_digest,
            field="successor_dataset_artifact_digest",
        )
        require_sha256(
            self.successor_study_digest,
            field="successor_study_digest",
        )
        require_sha256(
            self.successor_dataset_tree_digest,
            field="successor_dataset_tree_digest",
        )
        require_sha256(
            self.successor_study_tree_digest,
            field="successor_study_tree_digest",
        )
        require_sha256(
            self.successor_materialization_index_sha256,
            field="successor_materialization_index_sha256",
        )
        require_sha256(
            self.successor_runtime_environment_digest,
            field="successor_runtime_environment_digest",
        )
        _positive_int(
            self.successor_fresh_verification_artifact_id,
            field="successor_fresh_verification_artifact_id",
        )
        require_sha256(
            self.successor_fresh_verification_artifact_digest,
            field="successor_fresh_verification_artifact_digest",
        )
        require_non_empty(
            self.successor_execution_overlay,
            field="successor_execution_overlay",
        )
        symbols = tuple(self.symbols)
        caps = tuple(
            _finite_float(value, field="capacity_caps") for value in self.capacity_caps
        )
        fit_cutoff = _datetime(self.fit_cutoff, field="fit_cutoff")
        evaluation_start = _datetime(self.evaluation_start, field="evaluation_start")
        evaluation_stop = _datetime(
            self.evaluation_stop_exclusive,
            field="evaluation_stop_exclusive",
        )
        ppo_seeds = tuple(self.ppo_seeds)
        core_strategies = tuple(self.core_strategies)
        control_strategies = tuple(self.control_strategies)
        source_returns = tuple(
            (
                strategy,
                tuple(
                    _finite_float(value, field="source total returns")
                    for value in values
                ),
            )
            for strategy, values in self.source_core_total_returns
        )
        source_profitable = tuple(self.source_profitable_core_strategies)
        pre_successor_status = require_non_empty(
            self.pre_successor_suite_status,
            field="pre_successor_suite_status",
        )
        stage_b_role = require_non_empty(self.stage_b_role, field="stage_b_role")
        integration_gates = tuple(self.integration_gates)
        stage_a_failure_status = require_non_empty(
            self.stage_a_failure_status,
            field="stage_a_failure_status",
        )

        if len(symbols) != len(set(symbols)) or any(not item for item in symbols):
            raise ValueError("symbols must contain unique non-empty values")
        if len(caps) != len(symbols) or any(
            value <= 0.0 or value > 1.0 for value in caps
        ):
            raise ValueError(
                "capacity_caps must align with symbols and remain within (0, 1]"
            )
        if not fit_cutoff <= evaluation_start < evaluation_stop:
            raise ValueError("evaluation clock contract is invalid")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in ppo_seeds
        ):
            raise ValueError("ppo_seeds must be non-negative integers")
        if len(set(ppo_seeds)) != len(ppo_seeds):
            raise ValueError("ppo_seeds must be unique")
        if len(core_strategies) != len(set(core_strategies)) or any(
            not item for item in core_strategies
        ):
            raise ValueError("core_strategies must contain unique non-empty values")
        if len(control_strategies) != len(set(control_strategies)) or any(
            not item for item in control_strategies
        ):
            raise ValueError("control_strategies must contain unique non-empty values")
        if tuple(strategy for strategy, _ in source_returns) != core_strategies:
            raise ValueError(
                "source return strategy roster differs from preregistered core_strategies"
            )
        if any(len(values) != len(symbols) for _, values in source_returns):
            raise ValueError("source total returns must align with symbols")
        if any(item not in core_strategies for item in source_profitable):
            raise ValueError("source_profitable_core_strategies must be a core subset")
        if len(integration_gates) != len(set(integration_gates)) or any(
            not item for item in integration_gates
        ):
            raise ValueError("integration_gates must contain unique non-empty values")

        bool_fields = (
            ("ppo_disclosed", self.ppo_disclosed),
            ("ppo_in_formal_decision", self.ppo_in_formal_decision),
            ("stage_a_required", self.stage_a_required),
            (
                "robust_positive_requires_positive_median",
                self.robust_positive_requires_positive_median,
            ),
            ("integration_gate_satisfied", self.integration_gate_satisfied),
            ("execution_authorized", self.execution_authorized),
            ("stage_a_executed", self.stage_a_executed),
            ("stage_b_executed", self.stage_b_executed),
            ("successor_pnl_inspected", self.successor_pnl_inspected),
            (
                "experiment_0004_must_not_influence",
                self.experiment_0004_must_not_influence,
            ),
        )
        for bool_field, bool_value in bool_fields:
            _boolean(bool_value, field=bool_field)

        count_fields = (
            (
                "source_profitability_required_positive_symbols",
                self.source_profitability_required_positive_symbols,
            ),
            (
                "robust_positive_required_positive_symbols",
                self.robust_positive_required_positive_symbols,
            ),
            (
                "execution_fragile_max_positive_symbols",
                self.execution_fragile_max_positive_symbols,
            ),
            ("experiment_0004_issue", self.experiment_0004_issue),
        )
        for count_field, count_value in count_fields:
            _positive_int(count_value, field=count_field)

        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "capacity_caps", caps)
        object.__setattr__(self, "fit_cutoff", fit_cutoff)
        object.__setattr__(self, "evaluation_start", evaluation_start)
        object.__setattr__(self, "evaluation_stop_exclusive", evaluation_stop)
        object.__setattr__(self, "ppo_seeds", ppo_seeds)
        object.__setattr__(self, "core_strategies", core_strategies)
        object.__setattr__(self, "control_strategies", control_strategies)
        object.__setattr__(self, "source_core_total_returns", source_returns)
        object.__setattr__(self, "source_profitable_core_strategies", source_profitable)
        object.__setattr__(self, "pre_successor_suite_status", pre_successor_status)
        object.__setattr__(self, "stage_b_role", stage_b_role)
        object.__setattr__(self, "integration_gates", integration_gates)
        object.__setattr__(self, "stage_a_failure_status", stage_a_failure_status)

        if self.to_payload() != _registered_payload():
            raise ValueError(
                "capacity robustness contract differs from preregistered authority"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_baseline_artifact_id": self.source_baseline_artifact_id,
            "source_baseline_artifact_digest": self.source_baseline_artifact_digest,
            "source_baseline_evidence_fingerprint": self.source_baseline_evidence_fingerprint,
            "source_dataset_id": self.source_dataset_id,
            "source_dataset_artifact_digest": self.source_dataset_artifact_digest,
            "source_study_digest": self.source_study_digest,
            "source_implementation_sha": self.source_implementation_sha,
            "source_implementation_digest": self.source_implementation_digest,
            "source_runtime_environment_digest": self.source_runtime_environment_digest,
            "source_eligibility_run_id": self.source_eligibility_run_id,
            "source_eligibility_artifact_id": self.source_eligibility_artifact_id,
            "source_eligibility_artifact_digest": self.source_eligibility_artifact_digest,
            "source_eligibility_digest": self.source_eligibility_digest,
            "capacity_protocol_digest": self.capacity_protocol_digest,
            "capacity_result_digest": self.capacity_result_digest,
            "capacity_result_json_sha256": self.capacity_result_json_sha256,
            "capacity_result_artifact_id": self.capacity_result_artifact_id,
            "capacity_result_artifact_digest": self.capacity_result_artifact_digest,
            "capacity_verifier_run_id": self.capacity_verifier_run_id,
            "capacity_verifier_artifact_id": self.capacity_verifier_artifact_id,
            "capacity_verifier_artifact_digest": self.capacity_verifier_artifact_digest,
            "causal_capacity_pr": self.causal_capacity_pr,
            "causal_capacity_head_sha": self.causal_capacity_head_sha,
            "identity_layer_pr": self.identity_layer_pr,
            "identity_layer_head_sha": self.identity_layer_head_sha,
            "identity_layer_ci_run_id": self.identity_layer_ci_run_id,
            "successor_bundle_run_id": self.successor_bundle_run_id,
            "successor_bundle_artifact_id": self.successor_bundle_artifact_id,
            "successor_bundle_artifact_digest": self.successor_bundle_artifact_digest,
            "successor_dataset_id": self.successor_dataset_id,
            "successor_dataset_artifact_digest": self.successor_dataset_artifact_digest,
            "successor_study_digest": self.successor_study_digest,
            "successor_dataset_tree_digest": self.successor_dataset_tree_digest,
            "successor_study_tree_digest": self.successor_study_tree_digest,
            "successor_materialization_index_sha256": self.successor_materialization_index_sha256,
            "successor_runtime_environment_digest": self.successor_runtime_environment_digest,
            "successor_fresh_verification_artifact_id": self.successor_fresh_verification_artifact_id,
            "successor_fresh_verification_artifact_digest": self.successor_fresh_verification_artifact_digest,
            "successor_execution_overlay": self.successor_execution_overlay,
            "symbols": list(self.symbols),
            "capacity_caps": list(self.capacity_caps),
            "fit_cutoff": _iso(self.fit_cutoff),
            "evaluation_start": _iso(self.evaluation_start),
            "evaluation_stop_exclusive": _iso(self.evaluation_stop_exclusive),
            "ppo_seeds": list(self.ppo_seeds),
            "core_strategies": list(self.core_strategies),
            "control_strategies": list(self.control_strategies),
            "ppo_disclosed": self.ppo_disclosed,
            "ppo_in_formal_decision": self.ppo_in_formal_decision,
            "source_core_total_returns": [
                {"strategy": strategy, "total_returns": list(values)}
                for strategy, values in self.source_core_total_returns
            ],
            "source_profitable_core_strategies": list(
                self.source_profitable_core_strategies
            ),
            "pre_successor_suite_status": self.pre_successor_suite_status,
            "stage_b_role": self.stage_b_role,
            "integration_gates": list(self.integration_gates),
            "stage_a_required": self.stage_a_required,
            "stage_a_failure_status": self.stage_a_failure_status,
            "source_profitability_required_positive_symbols": (
                self.source_profitability_required_positive_symbols
            ),
            "robust_positive_required_positive_symbols": (
                self.robust_positive_required_positive_symbols
            ),
            "execution_fragile_max_positive_symbols": (
                self.execution_fragile_max_positive_symbols
            ),
            "robust_positive_requires_positive_median": (
                self.robust_positive_requires_positive_median
            ),
            "integration_gate_satisfied": self.integration_gate_satisfied,
            "execution_authorized": self.execution_authorized,
            "stage_a_executed": self.stage_a_executed,
            "stage_b_executed": self.stage_b_executed,
            "successor_pnl_inspected": self.successor_pnl_inspected,
            "experiment_0004_issue": self.experiment_0004_issue,
            "experiment_0004_must_not_influence": (
                self.experiment_0004_must_not_influence
            ),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _registered_payload() -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "source_baseline_artifact_id": _SOURCE_BASELINE_ARTIFACT_ID,
        "source_baseline_artifact_digest": _SOURCE_BASELINE_ARTIFACT_DIGEST,
        "source_baseline_evidence_fingerprint": _SOURCE_BASELINE_EVIDENCE_FINGERPRINT,
        "source_dataset_id": _SOURCE_DATASET_ID,
        "source_dataset_artifact_digest": _SOURCE_DATASET_ARTIFACT_DIGEST,
        "source_study_digest": _SOURCE_STUDY_DIGEST,
        "source_implementation_sha": _SOURCE_IMPLEMENTATION_SHA,
        "source_implementation_digest": _SOURCE_IMPLEMENTATION_DIGEST,
        "source_runtime_environment_digest": _SOURCE_RUNTIME_ENVIRONMENT_DIGEST,
        "source_eligibility_run_id": _SOURCE_ELIGIBILITY_RUN_ID,
        "source_eligibility_artifact_id": _SOURCE_ELIGIBILITY_ARTIFACT_ID,
        "source_eligibility_artifact_digest": _SOURCE_ELIGIBILITY_ARTIFACT_DIGEST,
        "source_eligibility_digest": _SOURCE_ELIGIBILITY_DIGEST,
        "capacity_protocol_digest": _CAPACITY_PROTOCOL_DIGEST,
        "capacity_result_digest": _CAPACITY_RESULT_DIGEST,
        "capacity_result_json_sha256": _CAPACITY_RESULT_JSON_SHA256,
        "capacity_result_artifact_id": _CAPACITY_RESULT_ARTIFACT_ID,
        "capacity_result_artifact_digest": _CAPACITY_RESULT_ARTIFACT_DIGEST,
        "capacity_verifier_run_id": _CAPACITY_VERIFIER_RUN_ID,
        "capacity_verifier_artifact_id": _CAPACITY_VERIFIER_ARTIFACT_ID,
        "capacity_verifier_artifact_digest": _CAPACITY_VERIFIER_ARTIFACT_DIGEST,
        "causal_capacity_pr": _CAUSAL_CAPACITY_PR,
        "causal_capacity_head_sha": _CAUSAL_CAPACITY_HEAD_SHA,
        "identity_layer_pr": _IDENTITY_LAYER_PR,
        "identity_layer_head_sha": _IDENTITY_LAYER_HEAD_SHA,
        "identity_layer_ci_run_id": _IDENTITY_LAYER_CI_RUN_ID,
        "successor_bundle_run_id": _SUCCESSOR_BUNDLE_RUN_ID,
        "successor_bundle_artifact_id": _SUCCESSOR_BUNDLE_ARTIFACT_ID,
        "successor_bundle_artifact_digest": _SUCCESSOR_BUNDLE_ARTIFACT_DIGEST,
        "successor_dataset_id": _SUCCESSOR_DATASET_ID,
        "successor_dataset_artifact_digest": _SUCCESSOR_DATASET_ARTIFACT_DIGEST,
        "successor_study_digest": _SUCCESSOR_STUDY_DIGEST,
        "successor_dataset_tree_digest": _SUCCESSOR_DATASET_TREE_DIGEST,
        "successor_study_tree_digest": _SUCCESSOR_STUDY_TREE_DIGEST,
        "successor_materialization_index_sha256": _SUCCESSOR_MATERIALIZATION_INDEX_SHA256,
        "successor_runtime_environment_digest": _SUCCESSOR_RUNTIME_ENVIRONMENT_DIGEST,
        "successor_fresh_verification_artifact_id": _SUCCESSOR_FRESH_VERIFICATION_ARTIFACT_ID,
        "successor_fresh_verification_artifact_digest": _SUCCESSOR_FRESH_VERIFICATION_ARTIFACT_DIGEST,
        "successor_execution_overlay": _SUCCESSOR_EXECUTION_OVERLAY,
        "symbols": list(_SYMBOLS),
        "capacity_caps": list(_CAPACITY_CAPS),
        "fit_cutoff": _iso(_FIT_CUTOFF),
        "evaluation_start": _iso(_EVALUATION_START),
        "evaluation_stop_exclusive": _iso(_EVALUATION_STOP_EXCLUSIVE),
        "ppo_seeds": list(_PPO_SEEDS),
        "core_strategies": list(_CORE_STRATEGIES),
        "control_strategies": list(_CONTROL_STRATEGIES),
        "ppo_disclosed": True,
        "ppo_in_formal_decision": False,
        "source_core_total_returns": [
            {"strategy": strategy, "total_returns": list(values)}
            for strategy, values in _SOURCE_CORE_TOTAL_RETURNS
        ],
        "source_profitable_core_strategies": list(_SOURCE_PROFITABLE_CORE_STRATEGIES),
        "pre_successor_suite_status": _PRE_SUCCESSOR_SUITE_STATUS,
        "stage_b_role": _STAGE_B_ROLE,
        "integration_gates": list(_INTEGRATION_GATES),
        "stage_a_required": True,
        "stage_a_failure_status": _STAGE_A_FAILURE_STATUS,
        "source_profitability_required_positive_symbols": (
            _SOURCE_PROFITABILITY_REQUIRED_POSITIVE_SYMBOLS
        ),
        "robust_positive_required_positive_symbols": (
            _ROBUST_POSITIVE_REQUIRED_POSITIVE_SYMBOLS
        ),
        "execution_fragile_max_positive_symbols": _EXECUTION_FRAGILE_MAX_POSITIVE_SYMBOLS,
        "robust_positive_requires_positive_median": True,
        "integration_gate_satisfied": False,
        "execution_authorized": False,
        "stage_a_executed": False,
        "stage_b_executed": False,
        "successor_pnl_inspected": False,
        "experiment_0004_issue": _EXPERIMENT_0004_ISSUE,
        "experiment_0004_must_not_influence": True,
    }


def _from_payload(raw: dict[str, object]) -> CapacityRobustnessProtocol:
    expected = set(_registered_payload())
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            "capacity robustness preregistration keys differ from contract: "
            f"missing={missing}, unknown={unknown}"
        )
    schema = raw.get("schema_version")
    if schema != _SCHEMA_VERSION:
        raise ValueError("schema_version differs from preregistered authority")

    return CapacityRobustnessProtocol(
        source_baseline_artifact_id=_positive_int(
            raw.get("source_baseline_artifact_id"), field="source_baseline_artifact_id"
        ),
        source_baseline_artifact_digest=cast(
            str, raw.get("source_baseline_artifact_digest")
        ),
        source_baseline_evidence_fingerprint=cast(
            str, raw.get("source_baseline_evidence_fingerprint")
        ),
        source_dataset_id=cast(str, raw.get("source_dataset_id")),
        source_dataset_artifact_digest=cast(
            str, raw.get("source_dataset_artifact_digest")
        ),
        source_study_digest=cast(str, raw.get("source_study_digest")),
        source_implementation_sha=cast(str, raw.get("source_implementation_sha")),
        source_implementation_digest=cast(str, raw.get("source_implementation_digest")),
        source_runtime_environment_digest=cast(
            str, raw.get("source_runtime_environment_digest")
        ),
        source_eligibility_run_id=_positive_int(
            raw.get("source_eligibility_run_id"), field="source_eligibility_run_id"
        ),
        source_eligibility_artifact_id=_positive_int(
            raw.get("source_eligibility_artifact_id"),
            field="source_eligibility_artifact_id",
        ),
        source_eligibility_artifact_digest=cast(
            str, raw.get("source_eligibility_artifact_digest")
        ),
        source_eligibility_digest=cast(str, raw.get("source_eligibility_digest")),
        capacity_protocol_digest=cast(str, raw.get("capacity_protocol_digest")),
        capacity_result_digest=cast(str, raw.get("capacity_result_digest")),
        capacity_result_json_sha256=cast(str, raw.get("capacity_result_json_sha256")),
        capacity_result_artifact_id=_positive_int(
            raw.get("capacity_result_artifact_id"), field="capacity_result_artifact_id"
        ),
        capacity_result_artifact_digest=cast(
            str, raw.get("capacity_result_artifact_digest")
        ),
        capacity_verifier_run_id=_positive_int(
            raw.get("capacity_verifier_run_id"), field="capacity_verifier_run_id"
        ),
        capacity_verifier_artifact_id=_positive_int(
            raw.get("capacity_verifier_artifact_id"),
            field="capacity_verifier_artifact_id",
        ),
        capacity_verifier_artifact_digest=cast(
            str, raw.get("capacity_verifier_artifact_digest")
        ),
        causal_capacity_pr=_positive_int(
            raw.get("causal_capacity_pr"), field="causal_capacity_pr"
        ),
        causal_capacity_head_sha=cast(str, raw.get("causal_capacity_head_sha")),
        identity_layer_pr=_positive_int(
            raw.get("identity_layer_pr"), field="identity_layer_pr"
        ),
        identity_layer_head_sha=cast(str, raw.get("identity_layer_head_sha")),
        identity_layer_ci_run_id=_positive_int(
            raw.get("identity_layer_ci_run_id"), field="identity_layer_ci_run_id"
        ),
        successor_bundle_run_id=_positive_int(
            raw.get("successor_bundle_run_id"), field="successor_bundle_run_id"
        ),
        successor_bundle_artifact_id=_positive_int(
            raw.get("successor_bundle_artifact_id"),
            field="successor_bundle_artifact_id",
        ),
        successor_bundle_artifact_digest=cast(
            str, raw.get("successor_bundle_artifact_digest")
        ),
        successor_dataset_id=cast(str, raw.get("successor_dataset_id")),
        successor_dataset_artifact_digest=cast(
            str, raw.get("successor_dataset_artifact_digest")
        ),
        successor_study_digest=cast(str, raw.get("successor_study_digest")),
        successor_dataset_tree_digest=cast(
            str, raw.get("successor_dataset_tree_digest")
        ),
        successor_study_tree_digest=cast(str, raw.get("successor_study_tree_digest")),
        successor_materialization_index_sha256=cast(
            str, raw.get("successor_materialization_index_sha256")
        ),
        successor_runtime_environment_digest=cast(
            str, raw.get("successor_runtime_environment_digest")
        ),
        successor_fresh_verification_artifact_id=_positive_int(
            raw.get("successor_fresh_verification_artifact_id"),
            field="successor_fresh_verification_artifact_id",
        ),
        successor_fresh_verification_artifact_digest=cast(
            str, raw.get("successor_fresh_verification_artifact_digest")
        ),
        successor_execution_overlay=cast(str, raw.get("successor_execution_overlay")),
        symbols=_string_tuple(raw.get("symbols"), field="symbols"),
        capacity_caps=_float_tuple(raw.get("capacity_caps"), field="capacity_caps"),
        fit_cutoff=_parse_datetime(raw.get("fit_cutoff"), field="fit_cutoff"),
        evaluation_start=_parse_datetime(
            raw.get("evaluation_start"), field="evaluation_start"
        ),
        evaluation_stop_exclusive=_parse_datetime(
            raw.get("evaluation_stop_exclusive"), field="evaluation_stop_exclusive"
        ),
        ppo_seeds=_int_tuple(raw.get("ppo_seeds"), field="ppo_seeds"),
        core_strategies=_string_tuple(
            raw.get("core_strategies"), field="core_strategies"
        ),
        control_strategies=_string_tuple(
            raw.get("control_strategies"), field="control_strategies"
        ),
        ppo_disclosed=_boolean(raw.get("ppo_disclosed"), field="ppo_disclosed"),
        ppo_in_formal_decision=_boolean(
            raw.get("ppo_in_formal_decision"), field="ppo_in_formal_decision"
        ),
        source_core_total_returns=_parse_source_returns(
            raw.get("source_core_total_returns")
        ),
        source_profitable_core_strategies=_string_tuple_allow_empty(
            raw.get("source_profitable_core_strategies"),
            field="source_profitable_core_strategies",
        ),
        pre_successor_suite_status=cast(str, raw.get("pre_successor_suite_status")),
        stage_b_role=cast(str, raw.get("stage_b_role")),
        integration_gates=_string_tuple(
            raw.get("integration_gates"), field="integration_gates"
        ),
        stage_a_required=_boolean(
            raw.get("stage_a_required"), field="stage_a_required"
        ),
        stage_a_failure_status=cast(str, raw.get("stage_a_failure_status")),
        source_profitability_required_positive_symbols=_positive_int(
            raw.get("source_profitability_required_positive_symbols"),
            field="source_profitability_required_positive_symbols",
        ),
        robust_positive_required_positive_symbols=_positive_int(
            raw.get("robust_positive_required_positive_symbols"),
            field="robust_positive_required_positive_symbols",
        ),
        execution_fragile_max_positive_symbols=_positive_int(
            raw.get("execution_fragile_max_positive_symbols"),
            field="execution_fragile_max_positive_symbols",
        ),
        robust_positive_requires_positive_median=_boolean(
            raw.get("robust_positive_requires_positive_median"),
            field="robust_positive_requires_positive_median",
        ),
        integration_gate_satisfied=_boolean(
            raw.get("integration_gate_satisfied"), field="integration_gate_satisfied"
        ),
        execution_authorized=_boolean(
            raw.get("execution_authorized"), field="execution_authorized"
        ),
        stage_a_executed=_boolean(
            raw.get("stage_a_executed"), field="stage_a_executed"
        ),
        stage_b_executed=_boolean(
            raw.get("stage_b_executed"), field="stage_b_executed"
        ),
        successor_pnl_inspected=_boolean(
            raw.get("successor_pnl_inspected"), field="successor_pnl_inspected"
        ),
        experiment_0004_issue=_positive_int(
            raw.get("experiment_0004_issue"), field="experiment_0004_issue"
        ),
        experiment_0004_must_not_influence=_boolean(
            raw.get("experiment_0004_must_not_influence"),
            field="experiment_0004_must_not_influence",
        ),
        schema_version=cast(str, schema),
    )


def _string_tuple_allow_empty(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} must contain only strings")
        result.append(require_non_empty(item, field=field))
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return tuple(result)


def canonical_capacity_robustness_protocol() -> CapacityRobustnessProtocol:
    """Return the exact pre-result capacity-robustness preregistration."""

    return _from_payload(_registered_payload())


def load_capacity_robustness_protocol(
    path: str | Path,
) -> CapacityRobustnessProtocol:
    """Load one strict regular-file copy of the sealed preregistration."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("capacity robustness preregistration must be a regular file")
    try:
        decoded = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "capacity robustness preregistration is not valid JSON"
        ) from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("capacity robustness preregistration must be a JSON object")
    return _from_payload(cast(dict[str, object], decoded))


__all__ = [
    "CapacityRobustnessProtocol",
    "canonical_capacity_robustness_protocol",
    "load_capacity_robustness_protocol",
]
