"""Strict write-once result identity and artifact for scenario-v2 attempts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes, freeze_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs.portfolio_family_comparison import (
    _FAMILY_NAMES,
    FourFamilyPortfolioReplayResult,
    PortfolioFamilyReplayOutcome,
)
from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
    FourFamilyPortfolioAttemptIdentity,
    _family_payload,
    _read_file,
    _require_object,
    _strict_json,
    _validate_family_payload,
    _validate_provenance_record,
    four_family_portfolio_attempt_identity_payload,
)
from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
    _publish_record as _publish_v1_record,
)
from trade_rl.evaluation.runs.portfolio_family_scenario_replay_v2 import (
    FourFamilyPortfolioScenarioReplayV2Result,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.rl.portfolio_runtime_scenario import (
    PortfolioRuntimeExecutionScenarioV2,
)
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioScenarioObservationRuntimeBindingV2,
    _datetime_ns,
    _execution_payload,
)

_IDENTITY_SCHEMA = "four_family_portfolio_scenario_attempt_identity_v2"
_SOURCE_SCHEMA = "four_family_portfolio_scenario_durable_sources_v2"
_IDENTITY_DIGEST_DOMAIN = b"trade_rl.four_family_portfolio_scenario_identity_v2\0"
_RESULT_SCHEMA = "four_family_portfolio_scenario_result_v2"
_RESULT_DIGEST_DOMAIN = b"trade_rl.four_family_portfolio_scenario_result_v2\0"
_ATTEMPT_FILES = frozenset(
    {
        "started/record.json",
        "provenance/before/record.json",
        "provenance/after/record.json",
        "scenario_source/before/record.json",
        "scenario_source/after/record.json",
        "result/manifest.json",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "schema",
        "training_context_sha256",
        "transform_digest",
        "checkpoint_sha256",
        "scenario_sha256",
        "scenario_inference_source_sha256",
        "base_execution_policy_digest",
        "training_execution_config_digest",
        "runtime_execution_policy_digest",
        "training_bound_digest",
        "runtime_bound_digest",
        "dataset_id",
        "training_start_index",
        "training_stop_index",
        "training_agent_stop_index",
        "runtime_start_index",
        "runtime_stop_index",
        "runtime_agent_stop_index",
        "runtime_start_ns",
        "runtime_horizon_seconds",
        "native_interval_seconds",
    }
)


def _require_digest(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    require_sha256(value, field=field)
    return value


def _require_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _execution_cost_bytes(value: object, *, field: str) -> bytes:
    if type(value) is not ExecutionCostConfig:
        raise TypeError(f"{field} must be exactly ExecutionCostConfig")
    return canonical_json_bytes(value.execution_policy_payload())


@dataclass(frozen=True, slots=True)
class FourFamilyPortfolioScenarioAttemptIdentityV2:
    """External v2 identity; the nested v1 identity is only common inputs."""

    common_input_declaration: FourFamilyPortfolioAttemptIdentity
    expected_scenario: PortfolioRuntimeExecutionScenarioV2
    expected_scenario_sha256: str
    expected_scenario_inference_source_sha256: str
    expected_durable_source_sha256: str
    schema_version: str = _IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != _IDENTITY_SCHEMA
        ):
            raise ValueError("scenario attempt identity schema is unsupported")
        if (
            type(self.common_input_declaration)
            is not FourFamilyPortfolioAttemptIdentity
        ):
            raise TypeError(
                "common_input_declaration must be exactly FourFamilyPortfolioAttemptIdentity"
            )
        if type(self.expected_scenario) is not PortfolioRuntimeExecutionScenarioV2:
            raise TypeError(
                "expected_scenario must be exactly PortfolioRuntimeExecutionScenarioV2"
            )
        require_sha256(self.expected_scenario_sha256, field="expected_scenario_sha256")
        require_sha256(
            self.expected_scenario_inference_source_sha256,
            field="expected_scenario_inference_source_sha256",
        )
        require_sha256(
            self.expected_durable_source_sha256,
            field="expected_durable_source_sha256",
        )
        self.expected_scenario.to_bytes()
        if self.expected_scenario.digest != self.expected_scenario_sha256:
            raise ValueError("scenario digest differs from its external expected pin")
        common = self.common_input_declaration
        scenario = self.expected_scenario
        if scenario.training_context_sha256 != common.expected_training_context_sha256:
            raise ValueError("scenario training context differs from common input pin")
        if scenario.checkpoint_sha256 != common.expected_checkpoint_sha256:
            raise ValueError("scenario checkpoint differs from common input pin")
        if _execution_cost_bytes(
            common.execution_cost, field="common_input_declaration.execution_cost"
        ) != _execution_cost_bytes(
            scenario.runtime_execution_cost, field="scenario.runtime_execution_cost"
        ):
            raise ValueError(
                "common input execution cost differs from scenario runtime cost"
            )


def four_family_portfolio_scenario_attempt_identity_v2_payload(
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> dict[str, object]:
    """Return the portable v2 identity payload; input roots are excluded."""
    if type(identity) is not FourFamilyPortfolioScenarioAttemptIdentityV2:
        raise TypeError(
            "identity must be exactly FourFamilyPortfolioScenarioAttemptIdentityV2"
        )
    identity._validate()
    return {
        "common_input_declaration": four_family_portfolio_attempt_identity_payload(
            identity.common_input_declaration
        ),
        "expected_durable_source_sha256": identity.expected_durable_source_sha256,
        "expected_scenario": identity.expected_scenario.to_payload(),
        "expected_scenario_inference_source_sha256": (
            identity.expected_scenario_inference_source_sha256
        ),
        "expected_scenario_sha256": identity.expected_scenario_sha256,
        "schema_version": _IDENTITY_SCHEMA,
    }


def four_family_portfolio_scenario_attempt_identity_v2_digest(
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> str:
    payload = four_family_portfolio_scenario_attempt_identity_v2_payload(identity)
    return hashlib.sha256(
        _IDENTITY_DIGEST_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()


def capture_four_family_scenario_durable_source_digest_v2() -> str:
    """Digest both v2 durable owners using checkout-independent source bytes."""
    package_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "evaluation/runs/portfolio_family_scenario_attempt_v2.py",
        "evaluation/runs/portfolio_family_scenario_result_artifact_v2.py",
    )
    files: list[dict[str, str]] = []
    for relative in relative_paths:
        path = package_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"durable source owner is missing or not regular: {relative}"
            )
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        if b"\r" in raw:
            raise ValueError(
                f"durable source owner has unsupported CR bytes: {relative}"
            )
        files.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
    return content_digest({"files": files, "schema_version": _SOURCE_SCHEMA})


@dataclass(frozen=True, slots=True)
class LoadedFourFamilyPortfolioScenarioResultArtifactV2:
    root: Path
    artifact_digest: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        require_sha256(self.artifact_digest, field="artifact_digest")
        frozen = freeze_json_value(self.payload)
        if not isinstance(frozen, Mapping):
            raise TypeError("result payload must be an object")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "payload", cast(Mapping[str, object], frozen))


def _publish_record(path: Path, record: object) -> bytes:
    raw = canonical_json_bytes(record)
    _publish_v1_record(path, raw)
    return raw


def _attempt_file_inventory(root: Path) -> set[str]:
    """Inventory only the exact V2 directories without widening the V1 owner."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("scenario attempt root must be a regular directory")
    found: set[str] = set()
    allowed_directories = {
        "families",
        "provenance",
        "result",
        "started",
        "complete",
        "failure",
        "scenario_source",
        "provenance/before",
        "provenance/after",
        "scenario_source/before",
        "scenario_source/after",
        *(f"families/{name}" for name in _FAMILY_NAMES),
    }
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("scenario attempt tree must not contain symlinks")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            if relative not in allowed_directories:
                raise ValueError(
                    "scenario attempt tree contains an unexpected directory"
                )
        elif path.is_file():
            found.add(relative)
        else:
            raise ValueError("scenario attempt tree contains a non-regular entry")
    return found


def _validate_started_record(
    raw: bytes, *, expected_identity: FourFamilyPortfolioScenarioAttemptIdentityV2
) -> dict[str, object]:
    record = _require_object(
        _strict_json(raw, field="scenario started record"),
        field="scenario started record",
        keys={"declared_roots", "identity", "schema_version"},
    )
    if record["schema_version"] != "four_family_portfolio_scenario_attempt_started_v2":
        raise ValueError("scenario started record schema is unsupported")
    declared_roots = _require_object(
        record["declared_roots"],
        field="scenario started declared roots",
        keys={"bundle_root", "dataset_artifact_root", "ridge_artifact_root"},
    )
    if any(type(value) is not str or not value for value in declared_roots.values()):
        raise ValueError("scenario started roots must be non-empty strings")
    expected = four_family_portfolio_scenario_attempt_identity_v2_payload(
        expected_identity
    )
    if canonical_json_bytes(record["identity"]) != canonical_json_bytes(expected):
        raise ValueError("started identity differs from the external v2 identity")
    return record


def _validate_scenario_source_records(
    root: Path,
    *,
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
    expected_before: Mapping[str, object],
    expected_after: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    validated: list[dict[str, object]] = []
    for position, supplied in (("before", expected_before), ("after", expected_after)):
        raw = _read_file(
            root / "scenario_source" / position / "record.json",
            field=f"{position} scenario source record",
        )
        record = _require_object(
            _strict_json(raw, field=f"{position} scenario source record"),
            field=f"{position} scenario source record",
            keys={
                "durable_source_sha256",
                "scenario_inference_source_sha256",
                "schema_version",
            },
        )
        if record["schema_version"] != "four_family_portfolio_scenario_source_v2":
            raise ValueError("scenario source record schema is unsupported")
        if (
            record["durable_source_sha256"] != identity.expected_durable_source_sha256
            or record["scenario_inference_source_sha256"]
            != identity.expected_scenario_inference_source_sha256
        ):
            raise ValueError("scenario source record differs from external pins")
        if canonical_json_bytes(record) != canonical_json_bytes(supplied):
            raise ValueError(f"{position} scenario source record differs from supplied")
        validated.append(record)
    if canonical_json_bytes(validated[0]) != canonical_json_bytes(validated[1]):
        raise ValueError("before/after scenario source records differ")
    return validated[0], validated[1]


def _validate_runtime_binding(
    value: object,
    *,
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> dict[str, object]:
    if type(value) is PortfolioScenarioObservationRuntimeBindingV2:
        payload = value.to_payload()
    elif isinstance(value, dict):
        payload = value
    else:
        raise TypeError("scenario replay runtime binding has an unsupported type")
    binding = _require_object(
        payload, field="scenario runtime binding", keys=set(_BINDING_FIELDS)
    )
    common = identity.common_input_declaration
    scenario = identity.expected_scenario
    for name in (
        "training_context_sha256",
        "transform_digest",
        "checkpoint_sha256",
        "scenario_sha256",
        "scenario_inference_source_sha256",
        "base_execution_policy_digest",
        "training_execution_config_digest",
        "runtime_execution_policy_digest",
        "training_bound_digest",
        "runtime_bound_digest",
        "dataset_id",
    ):
        _require_digest(binding[name], field=f"scenario runtime binding.{name}")
    if binding["schema"] != "shared_portfolio_scenario_observation_runtime_binding_v2":
        raise ValueError("scenario runtime binding schema is unsupported")
    expected_digests = {
        "training_context_sha256": common.expected_training_context_sha256,
        "transform_digest": scenario.transform_digest,
        "checkpoint_sha256": common.expected_checkpoint_sha256,
        "scenario_sha256": identity.expected_scenario_sha256,
        "scenario_inference_source_sha256": (
            identity.expected_scenario_inference_source_sha256
        ),
        "base_execution_policy_digest": (
            scenario.base_execution_cost.execution_policy_digest
        ),
        "training_execution_config_digest": content_digest(
            _execution_payload(scenario.base_execution_cost)
        ),
        "runtime_execution_policy_digest": (
            scenario.runtime_execution_cost.execution_policy_digest
        ),
        "runtime_bound_digest": common.bound.digest,
        "dataset_id": common.expected_ridge_binding.dataset_id,
    }
    for name, expected in expected_digests.items():
        if binding[name] != expected:
            raise ValueError(f"scenario runtime binding {name} differs from identity")
    int_fields = (
        "training_start_index",
        "training_stop_index",
        "training_agent_stop_index",
        "runtime_start_index",
        "runtime_stop_index",
        "runtime_agent_stop_index",
        "runtime_start_ns",
        "runtime_horizon_seconds",
        "native_interval_seconds",
    )
    integers = {
        name: _require_integer(binding[name], field=f"scenario runtime binding.{name}")
        for name in int_fields
    }
    if (
        integers["runtime_start_index"] != common.start_index
        or integers["runtime_stop_index"] != common.stop_index
        or integers["runtime_start_ns"]
        != _datetime_ns(common.bound.objective.evaluation_start)
        or integers["runtime_horizon_seconds"]
        != common.bound.clock.economic_horizon_seconds
        or integers["native_interval_seconds"]
        != common.bound.clock.execution_interval_seconds
    ):
        raise ValueError("scenario runtime binding window differs from identity")
    if (
        integers["training_agent_stop_index"]
        != integers["training_stop_index"]
        - scenario.base_execution_cost.order_latency_bars
        - 1
    ):
        raise ValueError(
            "scenario runtime binding training agent stop differs from base latency"
        )
    if (
        integers["runtime_agent_stop_index"]
        != integers["runtime_stop_index"]
        - scenario.runtime_execution_cost.order_latency_bars
        - 1
    ):
        raise ValueError(
            "scenario runtime binding agent stop differs from runtime latency"
        )
    if not (
        0
        <= integers["training_start_index"]
        < integers["training_agent_stop_index"]
        <= integers["training_stop_index"]
        <= integers["runtime_start_index"]
        < integers["runtime_agent_stop_index"]
        <= integers["runtime_stop_index"]
    ):
        raise ValueError("scenario runtime binding windows are inconsistent")
    if (
        integers["runtime_horizon_seconds"]
        != (integers["runtime_stop_index"] - integers["runtime_start_index"])
        * integers["native_interval_seconds"]
    ):
        raise ValueError("scenario runtime binding horizon is inconsistent")
    return cast(dict[str, object], binding)


def _validate_scenario_replay_metadata(
    value: object,
    *,
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> dict[str, object]:
    replay = _require_object(
        value,
        field="scenario replay metadata",
        keys={
            "runtime_binding",
            "scenario",
            "scenario_inference_source_sha256",
            "scenario_sha256",
            "schema_version",
        },
    )
    if replay["schema_version"] != "four_family_portfolio_scenario_replay_metadata_v2":
        raise ValueError("scenario replay metadata schema is unsupported")
    if replay["scenario_sha256"] != identity.expected_scenario_sha256:
        raise ValueError("actual scenario replay digest differs from identity")
    if (
        replay["scenario_inference_source_sha256"]
        != identity.expected_scenario_inference_source_sha256
    ):
        raise ValueError("actual scenario inference source differs from identity")
    if (
        canonical_json_bytes(replay["scenario"])
        != identity.expected_scenario.to_bytes()
    ):
        raise ValueError("actual scenario replay declaration differs from identity")
    replay["runtime_binding"] = _validate_runtime_binding(
        replay["runtime_binding"], identity=identity
    )
    return replay


def _dataset_payload(
    result: FourFamilyPortfolioReplayResult,
    *,
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> dict[str, object]:
    common = identity.common_input_declaration
    if (
        result.dataset_id != common.expected_ridge_binding.dataset_id
        or result.dataset_artifact.artifact_digest
        != common.expected_dataset_artifact_sha256
        or result.dataset_artifact.schema_version
        != common.expected_ridge_binding.dataset_schema_version
        or tuple(result.symbols) != common.expected_symbols
        or (result.start_index, result.stop_index)
        != (common.start_index, common.stop_index)
        or result.ridge_artifact_digest != common.ridge_artifact_sha256
        or result.ppo_bundle_sha256 != common.expected_bundle_sha256
    ):
        raise ValueError("scenario replay outputs differ from common input identity")
    return {
        "artifact_digest": result.dataset_artifact.artifact_digest,
        "dataset_id": result.dataset_id,
        "schema_version": result.dataset_artifact.schema_version,
        "symbols": list(result.symbols),
        "start_index": result.start_index,
        "stop_index": result.stop_index,
    }


def _validate_dataset_payload(
    value: object,
    *,
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> dict[str, object]:
    dataset = _require_object(
        value,
        field="scenario result dataset",
        keys={
            "artifact_digest",
            "dataset_id",
            "schema_version",
            "symbols",
            "start_index",
            "stop_index",
        },
    )
    common = identity.common_input_declaration
    expected = {
        "artifact_digest": common.expected_dataset_artifact_sha256,
        "dataset_id": common.expected_ridge_binding.dataset_id,
        "schema_version": common.expected_ridge_binding.dataset_schema_version,
        "symbols": list(common.expected_symbols),
        "start_index": common.start_index,
        "stop_index": common.stop_index,
    }
    if canonical_json_bytes(dataset) != canonical_json_bytes(expected):
        raise ValueError("scenario result Dataset differs from common input identity")
    return dataset


def _row_manifest(
    root: Path,
    *,
    outcomes: tuple[PortfolioFamilyReplayOutcome, ...],
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outcome, family in zip(outcomes, _FAMILY_NAMES, strict=True):
        path = root / "families" / family / "record.json"
        raw = _read_file(path, field=f"{family} scenario family record")
        row = _validate_family_payload(
            _strict_json(raw, field=f"{family} scenario family record"),
            expected_family=family,
            identity=identity.common_input_declaration,
        )
        if row != _family_payload(outcome):
            raise ValueError("durable family row differs from returned native outcome")
        rows.append(
            {
                "family": family,
                "path": f"families/{family}/record.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    return rows


def publish_four_family_portfolio_scenario_result_artifact_v2(
    attempt_root: str | Path,
    result: FourFamilyPortfolioScenarioReplayV2Result,
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
    *,
    before_provenance: Mapping[str, object],
    after_provenance: Mapping[str, object],
    before_scenario_source: Mapping[str, object],
    after_scenario_source: Mapping[str, object],
) -> str:
    """Publish a strict v2 manifest over actual scenario metadata and native rows."""
    if type(result) is not FourFamilyPortfolioScenarioReplayV2Result:
        raise TypeError(
            "result must be exactly FourFamilyPortfolioScenarioReplayV2Result"
        )
    if type(identity) is not FourFamilyPortfolioScenarioAttemptIdentityV2:
        raise TypeError(
            "identity must be exactly FourFamilyPortfolioScenarioAttemptIdentityV2"
        )
    identity._validate()
    root = Path(attempt_root)
    common = identity.common_input_declaration
    identity_payload = four_family_portfolio_scenario_attempt_identity_v2_payload(
        identity
    )
    premanifest = set(_ATTEMPT_FILES) - {"result/manifest.json"}
    premanifest.update(f"families/{name}/record.json" for name in _FAMILY_NAMES)
    if _attempt_file_inventory(root) != premanifest:
        raise ValueError("scenario attempt files differ from pre-result inventory")
    if type(result.common_replay) is not FourFamilyPortfolioReplayResult:
        raise TypeError(
            "scenario common replay must be exactly FourFamilyPortfolioReplayResult"
        )
    outcomes = tuple(result.common_replay.families)
    if tuple(item.family for item in outcomes) != _FAMILY_NAMES:
        raise ValueError("scenario result must contain the fixed four-family order")
    if type(result.scenario) is not PortfolioRuntimeExecutionScenarioV2:
        raise TypeError("scenario replay returned an invalid scenario declaration")
    replay_metadata = _validate_scenario_replay_metadata(
        {
            "runtime_binding": result.runtime_binding,
            "scenario": result.scenario.to_payload(),
            "scenario_inference_source_sha256": result.scenario_inference_source_sha256,
            "scenario_sha256": result.scenario_sha256,
            "schema_version": "four_family_portfolio_scenario_replay_metadata_v2",
        },
        identity=identity,
    )
    dataset = _dataset_payload(result.common_replay, identity=identity)
    started_raw = _read_file(
        root / "started" / "record.json", field="scenario started record"
    )
    _validate_started_record(started_raw, expected_identity=identity)
    before_digest = content_digest(before_provenance)
    after_digest = content_digest(after_provenance)
    before_record = _validate_provenance_record(
        root,
        "before",
        identity=common,
        expected_digest=before_digest,
    )
    after_record = _validate_provenance_record(
        root,
        "after",
        identity=common,
        expected_digest=after_digest,
    )
    if canonical_json_bytes(before_record) != canonical_json_bytes(before_provenance):
        raise ValueError("before provenance record differs from supplied provenance")
    if canonical_json_bytes(after_record) != canonical_json_bytes(after_provenance):
        raise ValueError("after provenance record differs from supplied provenance")
    if canonical_json_bytes(before_record) != canonical_json_bytes(after_record):
        raise ValueError("before/after provenance records differ")
    before_source, after_source = _validate_scenario_source_records(
        root,
        identity=identity,
        expected_before=before_scenario_source,
        expected_after=after_scenario_source,
    )
    rows = _row_manifest(root, outcomes=outcomes, identity=identity)
    payload: dict[str, object] = {
        "after_provenance_digest": content_digest(after_record),
        "after_scenario_source_digest": content_digest(after_source),
        "before_provenance_digest": content_digest(before_record),
        "before_scenario_source_digest": content_digest(before_source),
        "dataset": dataset,
        "families": rows,
        "identity": identity_payload,
        "identity_digest": four_family_portfolio_scenario_attempt_identity_v2_digest(
            identity
        ),
        "scenario_replay": replay_metadata,
        "schema_version": _RESULT_SCHEMA,
        "started_record": {
            "path": "started/record.json",
            "sha256": hashlib.sha256(started_raw).hexdigest(),
            "size_bytes": len(started_raw),
        },
    }
    result_digest = hashlib.sha256(
        _RESULT_DIGEST_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()
    _publish_record(
        root / "result" / "manifest.json",
        {"artifact_digest": result_digest, "payload": payload},
    )
    return result_digest


def _validate_family_rows(
    root: Path,
    value: object,
    *,
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> list[dict[str, object]]:
    rows = value
    if type(rows) is not list or len(rows) != len(_FAMILY_NAMES):
        raise ValueError("scenario result family manifest has invalid length")
    validated: list[dict[str, object]] = []
    for row_value, family in zip(rows, _FAMILY_NAMES, strict=True):
        row = _require_object(
            row_value,
            field="scenario result family entry",
            keys={"family", "path", "sha256", "size_bytes"},
        )
        if row["family"] != family or row["path"] != f"families/{family}/record.json":
            raise ValueError("scenario result family order or path differs")
        _require_digest(row["sha256"], field=f"{family} row digest")
        if type(row["size_bytes"]) is not int or row["size_bytes"] <= 0:
            raise ValueError(f"{family} row size must be a positive integer")
        raw = _read_file(
            root / str(row["path"]), field=f"{family} scenario family record"
        )
        if (
            len(raw) != row["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != row["sha256"]
        ):
            raise ValueError(
                f"{family} scenario family record differs from its manifest"
            )
        _validate_family_payload(
            _strict_json(raw, field=f"{family} scenario family record"),
            expected_family=family,
            identity=identity.common_input_declaration,
        )
        validated.append(row)
    return validated


def load_four_family_portfolio_scenario_result_artifact_v2(
    attempt_root: str | Path,
    *,
    expected_digest: str,
    expected_identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> LoadedFourFamilyPortfolioScenarioResultArtifactV2:
    """Strictly load a v2 result against its external typed identity and digest."""
    require_sha256(expected_digest, field="expected_digest")
    if type(expected_identity) is not FourFamilyPortfolioScenarioAttemptIdentityV2:
        raise TypeError(
            "expected_identity must be exactly FourFamilyPortfolioScenarioAttemptIdentityV2"
        )
    expected_identity._validate()
    root = Path(attempt_root)
    inventory = _attempt_file_inventory(root)
    allowed = set(_ATTEMPT_FILES)
    allowed.update(f"families/{name}/record.json" for name in _FAMILY_NAMES)
    allowed.update({"complete/record.json", "failure/record.json"})
    if not set(_ATTEMPT_FILES).issubset(inventory) or inventory - allowed:
        raise ValueError("scenario attempt tree has missing or unexpected result files")
    if "failure/record.json" in inventory:
        raise ValueError("failed scenario attempts cannot load a result")
    raw = _read_file(
        root / "result" / "manifest.json", field="scenario result manifest"
    )
    envelope = _require_object(
        _strict_json(raw, field="scenario result manifest"),
        field="scenario result manifest envelope",
        keys={"artifact_digest", "payload"},
    )
    digest = envelope["artifact_digest"]
    _require_digest(digest, field="artifact_digest")
    if digest != expected_digest:
        raise ValueError("scenario result digest differs from expected digest")
    payload = _require_object(
        envelope["payload"],
        field="scenario result payload",
        keys={
            "after_provenance_digest",
            "after_scenario_source_digest",
            "before_provenance_digest",
            "before_scenario_source_digest",
            "dataset",
            "families",
            "identity",
            "identity_digest",
            "scenario_replay",
            "schema_version",
            "started_record",
        },
    )
    if payload["schema_version"] != _RESULT_SCHEMA:
        raise ValueError("scenario result schema is unsupported")
    calculated = hashlib.sha256(
        _RESULT_DIGEST_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()
    if calculated != expected_digest:
        raise ValueError("scenario result digest does not match canonical payload")
    expected_payload = four_family_portfolio_scenario_attempt_identity_v2_payload(
        expected_identity
    )
    if canonical_json_bytes(payload["identity"]) != canonical_json_bytes(
        expected_payload
    ):
        raise ValueError(
            "scenario result identity differs from external expected identity"
        )
    identity_digest = four_family_portfolio_scenario_attempt_identity_v2_digest(
        expected_identity
    )
    if payload["identity_digest"] != identity_digest:
        raise ValueError(
            "scenario result identity digest differs from external identity"
        )
    _validate_dataset_payload(payload["dataset"], identity=expected_identity)
    replay_metadata = _validate_scenario_replay_metadata(
        payload["scenario_replay"], identity=expected_identity
    )
    if not isinstance(replay_metadata, dict):
        raise AssertionError("scenario metadata validator must return a dictionary")
    _validate_family_rows(root, payload["families"], identity=expected_identity)
    started = _require_object(
        payload["started_record"],
        field="scenario result started record entry",
        keys={"path", "sha256", "size_bytes"},
    )
    if (
        started["path"] != "started/record.json"
        or type(started["size_bytes"]) is not int
    ):
        raise ValueError("scenario result started record reference is invalid")
    started_raw = _read_file(
        root / "started" / "record.json", field="scenario started record"
    )
    if (
        len(started_raw) != started["size_bytes"]
        or hashlib.sha256(started_raw).hexdigest() != started["sha256"]
    ):
        raise ValueError("scenario result started record differs from its manifest")
    _validate_started_record(started_raw, expected_identity=expected_identity)
    provenance_digests = (
        ("before", payload["before_provenance_digest"]),
        ("after", payload["after_provenance_digest"]),
    )
    provenance_records: list[dict[str, object]] = []
    for position, expected_provenance_digest in provenance_digests:
        _require_digest(
            expected_provenance_digest, field=f"{position} provenance digest"
        )
        provenance = _validate_provenance_record(
            root,
            position,
            identity=expected_identity.common_input_declaration,
            expected_digest=expected_provenance_digest,
        )
        provenance_records.append(provenance)
    if canonical_json_bytes(provenance_records[0]) != canonical_json_bytes(
        provenance_records[1]
    ):
        raise ValueError("before/after provenance records differ")
    source_records: list[dict[str, object]] = []
    for position, expected_source_digest in (
        ("before", payload["before_scenario_source_digest"]),
        ("after", payload["after_scenario_source_digest"]),
    ):
        _require_digest(
            expected_source_digest, field=f"{position} scenario source digest"
        )
        source_raw = _read_file(
            root / "scenario_source" / position / "record.json",
            field=f"{position} scenario source record",
        )
        source_record = _require_object(
            _strict_json(source_raw, field=f"{position} scenario source record"),
            field=f"{position} scenario source record",
            keys={
                "durable_source_sha256",
                "scenario_inference_source_sha256",
                "schema_version",
            },
        )
        if content_digest(source_record) != expected_source_digest:
            raise ValueError(f"{position} scenario source digest differs from result")
        if (
            source_record["schema_version"]
            != "four_family_portfolio_scenario_source_v2"
        ):
            raise ValueError("scenario source record schema is unsupported")
        if (
            source_record["durable_source_sha256"]
            != expected_identity.expected_durable_source_sha256
            or source_record["scenario_inference_source_sha256"]
            != expected_identity.expected_scenario_inference_source_sha256
        ):
            raise ValueError("scenario source record differs from external pins")
        source_records.append(source_record)
    if canonical_json_bytes(source_records[0]) != canonical_json_bytes(
        source_records[1]
    ):
        raise ValueError("before/after scenario source records differ")
    if (
        capture_four_family_scenario_durable_source_digest_v2()
        != expected_identity.expected_durable_source_sha256
    ):
        raise ValueError("current v2 durable source differs from external pin")
    return LoadedFourFamilyPortfolioScenarioResultArtifactV2(
        root=root,
        artifact_digest=expected_digest,
        payload=payload,
    )


__all__ = [
    "FourFamilyPortfolioScenarioAttemptIdentityV2",
    "LoadedFourFamilyPortfolioScenarioResultArtifactV2",
    "capture_four_family_scenario_durable_source_digest_v2",
    "four_family_portfolio_scenario_attempt_identity_v2_digest",
    "four_family_portfolio_scenario_attempt_identity_v2_payload",
    "load_four_family_portfolio_scenario_result_artifact_v2",
    "publish_four_family_portfolio_scenario_result_artifact_v2",
]
