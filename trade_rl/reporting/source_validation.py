"""Fail-closed shape validation for persisted run-report source artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, list | tuple):
        return None
    return tuple(value)


def _digest_array(value: object) -> bool:
    values = _array(value)
    return values is not None and all(_sha256(item) for item in values)


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _json_object(path: Path) -> dict[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, Mapping):
        return None
    return {str(key): value for key, value in raw.items()}


def _digest_map(value: object, symbols: tuple[str, ...]) -> bool:
    if not isinstance(value, Mapping):
        return False
    values = {str(key): item for key, item in value.items()}
    return set(values) == set(symbols) and all(
        _sha256(values[symbol]) for symbol in symbols
    )


def _signal_shape_valid(root: Path) -> tuple[str, ...]:
    invalid: list[str] = []
    signal_root = root / "signal"
    if not signal_root.is_dir():
        return ()

    for path in sorted(signal_root.glob("*.json")):
        relative = str(path.relative_to(root))
        raw = _json_object(path)
        if raw is None:
            invalid.append(relative)
            continue
        schema = raw.get("schema_version")
        if path.name == "rejection.json":
            if schema == "causal_alpha_v3_signal_rejection_v2":
                fit_results = _array(raw.get("fit_results"))
                if fit_results is None or any(
                    not isinstance(item, Mapping) for item in fit_results
                ):
                    invalid.append(relative)
            continue
        if schema != "causal_alpha_v3_fit_signal_result_v2":
            continue
        if not _digest_array(raw.get("unavailable_scope_contract_digests")):
            invalid.append(relative)
            continue
        evidence = raw.get("evidence")
        if evidence is None:
            continue
        if not isinstance(evidence, Mapping):
            invalid.append(relative)
            continue
        evidence_values = {str(key): value for key, value in evidence.items()}
        if evidence_values.get(
            "schema_version"
        ) == "causal_alpha_v3_signal_gate_evidence_v2" and not _digest_array(
            evidence_values.get("metric_digests")
        ):
            invalid.append(relative)
    return tuple(invalid)


def _selection_shape_valid(root: Path) -> tuple[str, ...]:
    invalid: list[str] = []
    selection_root = root / "selection"
    for filename, schema in (
        ("evidence.json", "causal_alpha_v3_selection_evidence_v1"),
        ("rejection.json", "causal_alpha_v3_selection_rejection_v1"),
    ):
        path = selection_root / filename
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        raw = _json_object(path)
        if raw is None:
            invalid.append(relative)
            continue
        if raw.get("schema_version") == schema and not _digest_array(
            raw.get("candidate_evidence_digests")
        ):
            invalid.append(relative)

    progress_path = selection_root / "progress.json"
    if progress_path.is_file():
        relative = str(progress_path.relative_to(root))
        raw = _json_object(progress_path)
        if raw is None:
            invalid.append(relative)
        elif raw.get("schema_version") == "causal_alpha_v3_selection_progress_v1":
            counts_valid = all(
                _non_negative_int(raw.get(field))
                for field in (
                    "completed_replay_count",
                    "diagnostics_completed_count",
                    "expected_replay_count",
                    "fit_cache_hits",
                    "fit_count",
                )
            )
            candidates = _array(raw.get("candidates"))
            symbols = raw.get("symbols")
            rows_valid = candidates is not None and all(
                isinstance(item, Mapping) for item in candidates
            )
            symbols_valid = isinstance(symbols, Mapping) and all(
                isinstance(item, Mapping) for item in symbols.values()
            )
            fraction_valid = _finite_number(raw.get("completion_fraction"))
            if not (counts_valid and rows_valid and symbols_valid and fraction_valid):
                invalid.append(relative)
    return tuple(dict.fromkeys(invalid))


def _admission_shape_valid(root: Path) -> tuple[str, ...]:
    invalid: list[str] = []
    evidence_path = root / "admission" / "evidence.json"
    if evidence_path.is_file():
        relative = str(evidence_path.relative_to(root))
        raw = _json_object(evidence_path)
        if raw is None:
            invalid.append(relative)
        elif raw.get("schema_version") == "causal_alpha_v3_admission_evidence_v3":
            counts_valid = all(
                _non_negative_int(raw.get(field))
                for field in (
                    "hard_risk_violation_count",
                    "negative_gross_symbol_count",
                    "total_trade_count",
                    "unexplained_execution_rejection_count",
                )
            )
            returns_valid = all(
                _finite_number(raw.get(field))
                for field in (
                    "aggregate_gross_return",
                    "aggregate_net_return",
                    "worst_symbol_net_return",
                )
            )
            if not (
                _digest_array(raw.get("record_digests"))
                and _sha256(raw.get("base_admission_digest"))
                and counts_valid
                and returns_valid
            ):
                invalid.append(relative)

    rejection_path = root / "admission" / "rejection.json"
    if rejection_path.is_file():
        relative = str(rejection_path.relative_to(root))
        raw = _json_object(rejection_path)
        if raw is None:
            invalid.append(relative)
        elif raw.get(
            "schema_version"
        ) == "causal_alpha_v3_admission_rejection_v2" and not (
            _sha256(raw.get("admission_digest"))
            and _sha256(raw.get("selected_candidate_digest"))
        ):
            invalid.append(relative)
    return tuple(dict.fromkeys(invalid))


def _teacher_shape_valid(root: Path) -> tuple[str, ...]:
    path = root / "teacher" / "package.json"
    if not path.is_file():
        return ()
    relative = str(path.relative_to(root))
    raw = _json_object(path)
    if raw is None:
        return (relative,)
    if raw.get("schema_version") != "universal_causal_alpha_v3_teacher_package_v2":
        return ()

    symbols_raw = _array(raw.get("train_symbols"))
    if symbols_raw is None:
        return (relative,)
    symbols = tuple(str(item) for item in symbols_raw)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not isinstance(item, str) or not item for item in symbols_raw)
    ):
        return (relative,)
    maps_valid = all(
        _digest_map(raw.get(field), symbols)
        for field in (
            "admission_contract_digests",
            "batch_artifact_digests",
            "batch_digests",
            "partition_digests",
            "sample_digests",
        )
    )
    scalars_valid = all(
        _sha256(raw.get(field))
        for field in (
            "freeze_digest",
            "generator_code_digest",
            "run_manifest_digest",
            "selected_candidate_digest",
            "selection_digest",
            "teacher_admission_digest",
        )
    )
    return () if maps_valid and scalars_valid else (relative,)


def validate_run_report_source_shapes(root: Path) -> dict[str, tuple[str, ...]]:
    """Return recognized source paths whose JSON container/value shape is invalid."""

    source = Path(root)
    result = {
        "signal": _signal_shape_valid(source),
        "selection": _selection_shape_valid(source),
        "teacher_admission": _admission_shape_valid(source),
        "teacher_package": _teacher_shape_valid(source),
    }
    return {stage: paths for stage, paths in result.items() if paths}


__all__ = ["validate_run_report_source_shapes"]
