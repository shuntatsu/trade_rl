"""Atomic publication and inspection for canonical M2 study bootstrap."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.binance import (
    FrozenBinanceSource,
    _freeze_binance_source,
    _inspect_frozen_binance_source,
)
from trade_rl.evaluation.experiments.bootstrap.config import (
    CanonicalM2BootstrapConfig,
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.workflow import create_study, inspect_study
from trade_rl.evaluation.runs import (
    ResolvedCandidateRunSpec,
    build_candidate_run_provenance,
    resolve_candidate_run_spec,
)
from trade_rl.integrations.binance import (
    BinanceTransportMode,
    binance_interval_milliseconds,
    build_binance_market_dataset,
)
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set

_MANIFEST_SCHEMA_V1 = "canonical_m2_bootstrap_manifest_v1"
_MANIFEST_SCHEMA_V2 = "canonical_m2_bootstrap_manifest_v2"
_MANIFEST_KEYS_V1 = frozenset(
    {
        "schema_version",
        "bootstrap_config_digest",
        "vision_plan_digest",
        "raw_source_roster",
        "raw_source_roster_digest",
        "metadata_evidence",
        "dataset_id",
        "dataset_artifact_schema",
        "dataset_artifact_digest",
        "study_digest",
        "implementation_digest",
        "runtime_environment_digest",
        "bootstrap_digest",
    }
)
_MANIFEST_KEYS_V2 = _MANIFEST_KEYS_V1 | {"vision_resolution_digest"}
_ROOT_ENTRIES = frozenset(
    {"bootstrap.json", "bootstrap-manifest.json", "source", "dataset", "study"}
)
_ALLOWED_DATA_SOURCES = frozenset({"vision", "frozen:exchange-info"})


@dataclass(frozen=True, slots=True)
class CanonicalM2BootstrapResult:
    """Stable identity summary for one published canonical M2 bootstrap."""

    root: Path
    config_digest: str
    bootstrap_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    study_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for field in (
            "config_digest",
            "bootstrap_digest",
            "dataset_id",
            "dataset_artifact_digest",
            "study_digest",
        ):
            _require_digest(getattr(self, field), field=field)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            dict(payload),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return require_sha256(value, field=field)


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return cast(dict[str, object], value)


def _require_roster(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("raw_source_roster must be a non-empty array")
    result: list[dict[str, object]] = []
    for item in value:
        raw = _require_mapping(item, field="raw_source_roster item")
        if set(raw) != {"url", "sha256", "size_bytes"}:
            raise ValueError("raw source roster item keys differ from contract")
        if not isinstance(raw["url"], str) or not raw["url"]:
            raise ValueError("raw source roster URL must be non-empty")
        _require_digest(raw["sha256"], field="raw source sha256")
        size = raw["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("raw source size_bytes must be a positive integer")
        result.append(raw)
    return result


def _provenance_identity(payload: Mapping[str, object]) -> tuple[str, str]:
    return (
        _require_digest(
            payload.get("implementation_digest"), field="implementation_digest"
        ),
        _require_digest(
            payload.get("runtime_environment_digest"),
            field="runtime_environment_digest",
        ),
    )


def _prepare_output_parent(output: Path) -> None:
    parent = output.parent
    absolute_parent = parent.absolute()
    for candidate in (absolute_parent, *absolute_parent.parents):
        if candidate.is_symlink():
            raise ValueError("bootstrap output parent must not traverse a symlink")
    if parent.exists():
        if not parent.is_dir():
            raise ValueError("bootstrap output parent must be a regular directory")
        return
    parent.mkdir(parents=True)
    for candidate in (absolute_parent, *absolute_parent.parents):
        if candidate.is_symlink():
            raise ValueError("bootstrap output parent must not traverse a symlink")
    if not parent.is_dir():
        raise ValueError("bootstrap output parent must be a regular directory")


def _validate_dataset_range(
    config: CanonicalM2BootstrapConfig,
    dataset: MarketDataset,
) -> None:
    interval = np.timedelta64(
        binance_interval_milliseconds(config.base_timeframe),
        "ms",
    )
    start = np.datetime64(config.data_start.replace(tzinfo=None), "ns")
    stop = np.datetime64(config.data_stop_exclusive.replace(tzinfo=None), "ns")
    expected_first = start + interval
    if dataset.timestamps[0] != expected_first or dataset.timestamps[-1] != stop:
        raise ValueError(
            "published dataset timestamp range differs from bootstrap config"
        )


def _validate_execution_economics(
    config: CanonicalM2BootstrapConfig,
    dataset: MarketDataset,
) -> None:
    expected = config.execution_economics
    if expected is None:
        return
    expected_fields: tuple[tuple[str, object], ...] = (
        ("fee_rate", expected.fee_rate),
        ("maker_fee_rate", expected.maker_fee_rate),
        ("taker_fee_rate", expected.taker_fee_rate),
        ("spread_rate", expected.spread_rate),
        ("max_participation_rate", expected.max_participation_rate),
        ("borrow_rate", expected.borrow_rate),
    )
    for name, expected_value in expected_fields:
        observed = np.asarray(dataset.resolved_array(name))
        if not np.all(observed == expected_value):
            raise ValueError(f"published dataset execution economics differ for {name}")
    active = np.asarray(dataset.resolved_array("symbol_active"), dtype=np.bool_)
    borrow_available = np.asarray(dataset.resolved_array("borrow_available"))
    expected_borrow_available = active & expected.borrow_available
    if not np.array_equal(borrow_available, expected_borrow_available):
        raise ValueError(
            "published dataset execution economics differ for borrow_available"
        )
    identity = json.loads(dataset.identity_payload_json or "{}")
    if identity.get("execution_economics") != expected.canonical_payload():
        raise ValueError(
            "published dataset execution economics identity differs from "
            "bootstrap config"
        )


def _validate_fit_scope(
    dataset: MarketDataset,
    spec: ResolvedCandidateRunSpec,
) -> None:
    try:
        build_causal_forecast_training_set(
            dataset,
            feature_indices=spec.lean_config.feature_indices,
            fit_symbol_indices=spec.lean_config.fit_symbol_indices,
            fit_cutoff=spec.lean_config.fit_cutoff,
            horizon_hours=24,
        )
    except ValueError as error:
        raise ValueError("fit scope has no eligible training rows") from error


def _validate_study_against_config(
    config: CanonicalM2BootstrapConfig,
    *,
    dataset_root: Path,
) -> tuple[str, str, str]:
    snapshot = inspect_study(dataset_root.parent / "study")
    plan = snapshot.plan
    dataset = load_market_dataset_artifact(dataset_root)
    artifact = inspect_published_market_dataset_artifact(dataset_root)

    if snapshot.baseline is not None or snapshot.experiment_sequences:
        raise ValueError("bootstrap Study must remain before baseline execution")
    if (snapshot.root / "baseline").exists():
        raise ValueError("bootstrap Study unexpectedly contains baseline evidence")
    if plan.research_question != config.research_question:
        raise ValueError("Study research question differs from bootstrap config")
    if plan.dataset_id != dataset.dataset_id:
        raise ValueError("Study dataset id differs from dataset artifact")
    if plan.dataset_artifact_schema != artifact.schema_version:
        raise ValueError("Study dataset artifact schema differs from dataset artifact")
    if plan.dataset_artifact_digest != artifact.artifact_digest:
        raise ValueError("Study dataset artifact digest differs from dataset artifact")
    if plan.symbols != config.symbols or tuple(dataset.symbols) != config.symbols:
        raise ValueError("Study or dataset symbol roster differs from bootstrap config")
    if plan.ppo_seeds != config.ppo_seeds:
        raise ValueError("Study seed policy differs from bootstrap config")
    if tuple(plan.allowed_factors) != config.allowed_factors:
        raise ValueError("Study controlled-factor policy differs from bootstrap config")
    if plan.max_experiments != config.max_experiments:
        raise ValueError("Study experiment budget differs from bootstrap config")
    if plan.n_bootstrap != config.n_bootstrap:
        raise ValueError("Study bootstrap count differs from bootstrap config")
    if plan.bootstrap_seed != config.bootstrap_seed:
        raise ValueError("Study bootstrap seed differs from bootstrap config")

    _validate_dataset_range(config, dataset)
    _validate_execution_economics(config, dataset)
    resolved = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=config.baseline,
    )
    _validate_fit_scope(dataset, resolved)
    frozen = plan.baseline_config
    lean = resolved.lean_config
    expected_pairs: tuple[tuple[object, object, str], ...] = (
        (frozen.signal_name, config.baseline.signal_name, "signal_name"),
        (frozen.signal_index, lean.signal_index, "signal_index"),
        (frozen.feature_names, config.baseline.feature_names, "feature_names"),
        (frozen.feature_indices, lean.feature_indices, "feature_indices"),
        (
            frozen.fit_symbol_names,
            config.baseline.fit_symbol_names,
            "fit_symbol_names",
        ),
        (frozen.fit_symbol_indices, lean.fit_symbol_indices, "fit_symbol_indices"),
        (frozen.ppo_seed, config.ppo_seeds[0], "ppo_seed"),
    )
    for observed, expected, field in expected_pairs:
        if observed != expected:
            raise ValueError(
                f"Study baseline resolved {field} differs from bootstrap config"
            )
    if frozen.fit_cutoff != str(np.datetime64(config.baseline.fit_cutoff, "ns")):
        raise ValueError("Study baseline fit_cutoff differs from bootstrap config")
    if frozen.evaluation_start != str(
        np.datetime64(config.baseline.evaluation_start, "ns")
    ):
        raise ValueError(
            "Study baseline evaluation_start differs from bootstrap config"
        )
    if frozen.evaluation_stop_exclusive != str(
        np.datetime64(config.baseline.evaluation_stop_exclusive, "ns")
    ):
        raise ValueError("Study baseline evaluation_stop differs from bootstrap config")
    numeric_pairs: tuple[tuple[object, object, str], ...] = (
        (
            frozen.rule_entry_threshold,
            config.baseline.rule_entry_threshold,
            "rule_entry_threshold",
        ),
        (
            frozen.rule_exit_threshold,
            config.baseline.rule_exit_threshold,
            "rule_exit_threshold",
        ),
        (
            frozen.forecast_entry_threshold,
            config.baseline.forecast_entry_threshold,
            "forecast_entry_threshold",
        ),
        (
            frozen.forecast_exit_threshold,
            config.baseline.forecast_exit_threshold,
            "forecast_exit_threshold",
        ),
        (
            frozen.ppo_total_timesteps,
            config.baseline.ppo_total_timesteps,
            "ppo_total_timesteps",
        ),
        (frozen.gross_budget, config.baseline.gross_budget, "gross_budget"),
        (frozen.initial_capital, config.baseline.initial_capital, "initial_capital"),
    )
    for observed, expected, field in numeric_pairs:
        if observed != expected:
            raise ValueError(f"Study baseline {field} differs from bootstrap config")
    if frozen.execution_overlay != "zero_overlay_dataset_fields_authoritative":
        raise ValueError(
            "Study baseline execution overlay differs from maintained contract"
        )
    return plan.digest, plan.implementation_digest, plan.runtime_environment_digest


def _manifest_body(
    config: CanonicalM2BootstrapConfig,
    frozen: FrozenBinanceSource,
    *,
    dataset_id: str,
    dataset_artifact_schema: str,
    dataset_artifact_digest: str,
    study_digest: str,
    implementation_digest: str,
    runtime_environment_digest: str,
) -> dict[str, object]:
    if frozen.vision_resolution_digest is None:
        raise ValueError("new bootstrap manifest requires Vision resolution digest")
    return {
        "schema_version": _MANIFEST_SCHEMA_V2,
        "bootstrap_config_digest": config.digest,
        "vision_plan_digest": frozen.vision_plan_digest,
        "vision_resolution_digest": frozen.vision_resolution_digest,
        "raw_source_roster": list(frozen.raw_source_roster),
        "raw_source_roster_digest": frozen.raw_source_roster_digest,
        "metadata_evidence": frozen.metadata_evidence,
        "dataset_id": dataset_id,
        "dataset_artifact_schema": dataset_artifact_schema,
        "dataset_artifact_digest": dataset_artifact_digest,
        "study_digest": study_digest,
        "implementation_digest": implementation_digest,
        "runtime_environment_digest": runtime_environment_digest,
    }


def _inspect_manifest(root: Path) -> dict[str, object]:
    manifest = _read_json_object(
        root / "bootstrap-manifest.json",
        label="bootstrap manifest",
    )
    schema = manifest.get("schema_version")
    if schema == _MANIFEST_SCHEMA_V1:
        expected_keys = _MANIFEST_KEYS_V1
    elif schema == _MANIFEST_SCHEMA_V2:
        expected_keys = _MANIFEST_KEYS_V2
    else:
        raise ValueError("bootstrap manifest schema differs from contract")
    if set(manifest) != expected_keys:
        raise ValueError("bootstrap manifest keys differ from contract")
    observed_digest = _require_digest(
        manifest.get("bootstrap_digest"),
        field="bootstrap_digest",
    )
    body = dict(manifest)
    del body["bootstrap_digest"]
    if content_digest(body) != observed_digest:
        raise ValueError("bootstrap manifest digest mismatch")
    _require_digest(
        manifest.get("bootstrap_config_digest"),
        field="bootstrap_config_digest",
    )
    _require_digest(manifest.get("vision_plan_digest"), field="vision_plan_digest")
    if schema == _MANIFEST_SCHEMA_V2:
        _require_digest(
            manifest.get("vision_resolution_digest"),
            field="vision_resolution_digest",
        )
    roster = _require_roster(manifest.get("raw_source_roster"))
    roster_digest = _require_digest(
        manifest.get("raw_source_roster_digest"),
        field="raw_source_roster_digest",
    )
    if content_digest(roster) != roster_digest:
        raise ValueError("raw source roster digest mismatch")
    _require_mapping(manifest.get("metadata_evidence"), field="metadata_evidence")
    _require_digest(manifest.get("dataset_id"), field="dataset_id")
    if not isinstance(manifest.get("dataset_artifact_schema"), str):
        raise ValueError("dataset_artifact_schema must be a string")
    _require_digest(
        manifest.get("dataset_artifact_digest"),
        field="dataset_artifact_digest",
    )
    _require_digest(manifest.get("study_digest"), field="study_digest")
    _require_digest(
        manifest.get("implementation_digest"),
        field="implementation_digest",
    )
    _require_digest(
        manifest.get("runtime_environment_digest"),
        field="runtime_environment_digest",
    )
    return manifest


def inspect_canonical_m2_bootstrap(
    root: str | Path,
) -> CanonicalM2BootstrapResult:
    """Verify one published bootstrap from stored evidence without network access."""

    bootstrap_root = Path(root)
    if bootstrap_root.is_symlink() or not bootstrap_root.is_dir():
        raise ValueError("bootstrap root must be a regular directory")
    entries = {entry.name for entry in bootstrap_root.iterdir()}
    if entries != _ROOT_ENTRIES:
        raise ValueError("bootstrap root entries differ from contract")

    config = load_canonical_m2_bootstrap_config(bootstrap_root / "bootstrap.json")
    manifest = _inspect_manifest(bootstrap_root)
    manifest_schema = manifest["schema_version"]
    frozen = _inspect_frozen_binance_source(
        config,
        bootstrap_root / "source",
        require_resolution=manifest_schema == _MANIFEST_SCHEMA_V2,
    )
    artifact = inspect_published_market_dataset_artifact(bootstrap_root / "dataset")
    dataset = load_market_dataset_artifact(bootstrap_root / "dataset")
    study_digest, plan_implementation, plan_runtime = _validate_study_against_config(
        config,
        dataset_root=bootstrap_root / "dataset",
    )
    if manifest["bootstrap_config_digest"] != config.digest:
        raise ValueError("bootstrap config digest differs from manifest")
    if manifest["vision_plan_digest"] != frozen.vision_plan_digest:
        raise ValueError("Vision plan digest differs from bootstrap manifest")
    if manifest_schema == _MANIFEST_SCHEMA_V2:
        if manifest["vision_resolution_digest"] != frozen.vision_resolution_digest:
            raise ValueError("Vision resolution digest differs from bootstrap manifest")
    if manifest["raw_source_roster"] != list(frozen.raw_source_roster):
        raise ValueError("raw source roster differs from frozen source evidence")
    if manifest["raw_source_roster_digest"] != frozen.raw_source_roster_digest:
        raise ValueError("raw source roster digest differs from frozen source evidence")
    if manifest["metadata_evidence"] != frozen.metadata_evidence:
        raise ValueError("metadata evidence differs from bootstrap manifest")
    if manifest["dataset_id"] != dataset.dataset_id:
        raise ValueError("dataset id differs from bootstrap manifest")
    if manifest["dataset_artifact_schema"] != artifact.schema_version:
        raise ValueError("dataset artifact schema differs from bootstrap manifest")
    if manifest["dataset_artifact_digest"] != artifact.artifact_digest:
        raise ValueError("dataset artifact digest differs from bootstrap manifest")
    if manifest["study_digest"] != study_digest:
        raise ValueError("Study digest differs from bootstrap manifest")
    if manifest["implementation_digest"] != plan_implementation:
        raise ValueError("Study implementation digest differs from bootstrap manifest")
    if manifest["runtime_environment_digest"] != plan_runtime:
        raise ValueError("Study runtime digest differs from bootstrap manifest")

    return CanonicalM2BootstrapResult(
        root=bootstrap_root,
        config_digest=config.digest,
        bootstrap_digest=cast(str, manifest["bootstrap_digest"]),
        dataset_id=dataset.dataset_id,
        dataset_artifact_digest=artifact.artifact_digest,
        study_digest=study_digest,
    )


def bootstrap_canonical_m2_study(
    config_path: str | Path,
    output_root: str | Path,
) -> CanonicalM2BootstrapResult:
    """Freeze source, publish dataset and StudyPlan, then atomically publish the root."""

    config = load_canonical_m2_bootstrap_config(config_path)
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"bootstrap output already exists: {output}")
    _prepare_output_parent(output)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        start_provenance = build_candidate_run_provenance()
        start_implementation, start_runtime = _provenance_identity(start_provenance)
        _write_json(staging / "bootstrap.json", config.to_payload())

        frozen = _freeze_binance_source(config, staging / "source")
        build = build_binance_market_dataset(
            market=config.market,
            symbols=config.symbols,
            interval=config.base_timeframe,
            start_time=config.data_start,
            end_time=config.data_stop_exclusive,
            transport_mode=BinanceTransportMode.VISION,
            transport=frozen.composite_transport,
            feature_timeframes=config.feature_timeframes,
            metadata_evidence=frozen.metadata_evidence,
            execution_economics=config.execution_economics,
        )
        sources = frozenset(build.sources_used)
        if not sources or not sources.issubset(_ALLOWED_DATA_SOURCES):
            raise ValueError(
                "dataset build used source outside frozen bootstrap evidence"
            )
        if "vision" not in sources or "frozen:exchange-info" not in sources:
            raise ValueError(
                "dataset build did not use complete frozen Binance evidence"
            )

        published = publish_market_dataset_artifact(staging / "dataset", build.dataset)
        artifact = inspect_published_market_dataset_artifact(staging / "dataset")
        loaded_dataset = load_market_dataset_artifact(staging / "dataset")
        if loaded_dataset.dataset_id != build.dataset.dataset_id:
            raise ValueError("reloaded dataset id differs from built dataset")
        if artifact.artifact_digest != published.artifact_digest:
            raise ValueError(
                "reloaded dataset artifact digest differs from publication"
            )
        if tuple(loaded_dataset.symbols) != config.symbols:
            raise ValueError(
                "published dataset symbol roster differs from bootstrap config"
            )
        _validate_dataset_range(config, loaded_dataset)
        resolved = resolve_candidate_run_spec(
            loaded_dataset,
            dataset_artifact_schema=artifact.schema_version,
            dataset_artifact_digest=artifact.artifact_digest,
            config=config.baseline,
        )
        _validate_fit_scope(loaded_dataset, resolved)

        create_study(
            staging / "study",
            dataset_root=staging / "dataset",
            research_question=config.research_question,
            baseline_config=config.baseline,
            ppo_seeds=config.ppo_seeds,
            allowed_factors=config.allowed_factors,
            max_experiments=config.max_experiments,
            n_bootstrap=config.n_bootstrap,
            bootstrap_seed=config.bootstrap_seed,
        )
        study_digest, plan_implementation, plan_runtime = (
            _validate_study_against_config(
                config,
                dataset_root=staging / "dataset",
            )
        )
        if plan_implementation != start_implementation:
            raise ValueError(
                "Study implementation provenance differs from bootstrap start"
            )
        if plan_runtime != start_runtime:
            raise ValueError("Study runtime provenance differs from bootstrap start")

        body = _manifest_body(
            config,
            frozen,
            dataset_id=loaded_dataset.dataset_id,
            dataset_artifact_schema=artifact.schema_version,
            dataset_artifact_digest=artifact.artifact_digest,
            study_digest=study_digest,
            implementation_digest=start_implementation,
            runtime_environment_digest=start_runtime,
        )
        manifest = dict(body)
        manifest["bootstrap_digest"] = content_digest(body)
        _write_json(staging / "bootstrap-manifest.json", manifest)
        validated = inspect_canonical_m2_bootstrap(staging)

        end_provenance = build_candidate_run_provenance()
        end_implementation, end_runtime = _provenance_identity(end_provenance)
        if end_implementation != start_implementation or end_runtime != start_runtime:
            raise ValueError("bootstrap provenance drift detected during publication")

        result = CanonicalM2BootstrapResult(
            root=output,
            config_digest=validated.config_digest,
            bootstrap_digest=validated.bootstrap_digest,
            dataset_id=validated.dataset_id,
            dataset_artifact_digest=validated.dataset_artifact_digest,
            study_digest=validated.study_digest,
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"bootstrap output already exists: {output}")
        staging.rename(output)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CanonicalM2BootstrapResult",
    "bootstrap_canonical_m2_study",
    "inspect_canonical_m2_bootstrap",
]
