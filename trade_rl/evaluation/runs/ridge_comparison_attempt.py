"""Write-once execution journal for the public frozen-Ridge comparison."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes, to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.evaluation.runs.ridge_artifact_comparison import (
    RidgeArtifactComparison,
    RidgeArtifactComparisonConfig,
    compare_ridge_artifact_shared_cash,
)
from trade_rl.evaluation.runs.ridge_comparison_artifact import (
    load_ridge_comparison_artifact,
    publish_ridge_comparison_artifact,
)
from trade_rl.strategies.forecasts import RidgeForecastArtifactBinding

_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_ATTEMPT_SCHEMA = "ridge_comparison_attempt_v1"


@dataclass(frozen=True, slots=True)
class PublishedRidgeComparisonAttempt:
    """The completed write-once attempt directory and its strict result digest."""

    root: Path
    result_artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("root must be a Path")
        _require_sha256(self.result_artifact_digest, "result_artifact_digest")


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _declared_path(value: object, field: str) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field} must be a string or Path")
    rendered = str(value)
    if not rendered.strip():
        raise ValueError(f"{field} must not be empty")
    return Path(value).as_posix()


def _absolute_attempt_path(value: object) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("attempt_root must be a string or Path")
    if not str(value).strip():
        raise ValueError("attempt_root must not be empty")
    root = Path(value).expanduser().absolute()
    if not root.name:
        raise ValueError("attempt_root must name a directory below a parent")
    return root


def _config_declaration(config: RidgeArtifactComparisonConfig) -> object:
    profile = config.market_order_profile
    profile_declaration: object
    if profile is None:
        profile_declaration = None
    else:
        profile_declaration = {
            "digest": profile.digest,
            "declaration": profile.canonical_payload(),
        }
    return {
        "entry_threshold": config.entry_threshold,
        "execution_cost": to_json_value(config.execution_cost),
        "exit_threshold": config.exit_threshold,
        "gross_budget": config.gross_budget,
        "initial_capital": config.initial_capital,
        "market_order_profile": profile_declaration,
        "minimum_hold_bars": config.minimum_hold_bars,
        "one_way_switch_cost": config.one_way_switch_cost,
        "risk": to_json_value(config.risk),
        "start_index": config.start_index,
        "stop_index": config.stop_index,
    }


def _publish_record(destination: Path, record: object) -> None:
    """Publish one canonical record directory without replacing any path."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"attempt record already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    atomic_write_bytes(staging / "record.json", canonical_json_bytes(record))
    atomic_rename_directory(staging, destination)


def _reserve_attempt(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"attempt destination already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    # mkdir is the exclusive reservation. If another caller wins the race,
    # this fails without reading either input artifact.
    root.mkdir()


def _started_record(
    *,
    root: Path,
    dataset_root: str,
    ridge_artifact_root: str,
    expected_ridge_digest: str,
    expected_binding: RidgeForecastArtifactBinding,
    config: RidgeArtifactComparisonConfig,
    research_context_digest: str,
) -> dict[str, object]:
    return {
        "attempt_root": root.as_posix(),
        "config": _config_declaration(config),
        "expected_binding": to_json_value(expected_binding),
        "expected_ridge_digest": expected_ridge_digest,
        "paths": {
            "dataset_root": dataset_root,
            "ridge_artifact_root": ridge_artifact_root,
        },
        "research_context_digest": research_context_digest,
        "schema_version": f"{_ATTEMPT_SCHEMA}_started",
    }


def _partial_record(comparison: RidgeArtifactComparison) -> dict[str, object]:
    arms: list[dict[str, object]] = []
    for entry in comparison.entries:
        evidence = entry.replay.ledger_evidence
        if evidence is None:
            raise ValueError(
                f"comparison arm {entry.name!r} has no captured ledger evidence"
            )
        arms.append(
            {
                "ledger": to_json_value(evidence.to_mapping()),
                "name": entry.name,
            }
        )
    return {
        "arms": arms,
        "schema_version": f"{_ATTEMPT_SCHEMA}_partial_ledgers",
    }


def _failure_record(
    *,
    phase: str,
    error: BaseException,
    comparison_returned: bool,
    partial_written: bool,
    partial_error: BaseException | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "comparison_returned": comparison_returned,
        "exception_message": str(error),
        "exception_type": type(error).__name__,
        "phase": phase,
        "schema_version": f"{_ATTEMPT_SCHEMA}_failure",
    }
    if partial_written:
        record["partial_ledger_record"] = "partial/record.json"
    elif comparison_returned:
        record["ledger_evidence_note"] = (
            "comparison returned without a usable typed ledger result, or partial "
            "ledger journaling failed; no completed artifact was published"
        )
    else:
        record["ledger_evidence_note"] = (
            "comparison did not return; internal partial ledger state is not exposed "
            "by the comparison API"
        )
    if partial_error is not None:
        record["partial_journal_error"] = {
            "message": str(partial_error),
            "type": type(partial_error).__name__,
        }
    return record


def _add_failure_note(
    original: BaseException, label: str, secondary: BaseException
) -> None:
    add_note = getattr(original, "add_note", None)
    if callable(add_note):
        add_note(f"{label}: {type(secondary).__name__}: {secondary}")


def run_ridge_comparison_attempt(
    *,
    dataset_root: str | Path,
    ridge_artifact_root: str | Path,
    expected_ridge_digest: str,
    expected_binding: RidgeForecastArtifactBinding,
    config: RidgeArtifactComparisonConfig,
    attempt_root: str | Path,
    research_context_digest: str,
) -> PublishedRidgeComparisonAttempt:
    """Reserve, run, journal, publish, and strictly reload one comparison attempt.

    The context digest is a caller declaration. This wrapper neither authorizes a
    research run nor authenticates the meaning or approval of that context.
    """

    dataset_path = _declared_path(dataset_root, "dataset_root")
    ridge_path = _declared_path(ridge_artifact_root, "ridge_artifact_root")
    digest = _require_sha256(expected_ridge_digest, "expected_ridge_digest")
    context_digest = _require_sha256(research_context_digest, "research_context_digest")
    if type(expected_binding) is not RidgeForecastArtifactBinding:
        raise TypeError("expected_binding must be a RidgeForecastArtifactBinding")
    if type(config) is not RidgeArtifactComparisonConfig:
        raise TypeError("config must be a RidgeArtifactComparisonConfig")
    root = _absolute_attempt_path(attempt_root)

    _reserve_attempt(root)
    phase = "started_record"
    comparison: object | None = None
    before: Mapping[str, object] | None = None
    after: Mapping[str, object] | None = None
    partial_written = False
    partial_error: BaseException | None = None
    result_digest: str | None = None

    try:
        started = _started_record(
            root=root,
            dataset_root=dataset_path,
            ridge_artifact_root=ridge_path,
            expected_ridge_digest=digest,
            expected_binding=expected_binding,
            config=config,
            research_context_digest=context_digest,
        )
        _publish_record(root / "started", started)

        phase = "before_provenance"
        before = build_candidate_run_provenance(research_context_digest=context_digest)
        phase = "before_provenance_journal"
        _publish_record(root / "before", {"provenance": before})

        phase = "comparison"
        comparison = compare_ridge_artifact_shared_cash(
            dataset_root=dataset_root,
            ridge_artifact_root=ridge_artifact_root,
            expected_ridge_digest=digest,
            expected_binding=expected_binding,
            config=config,
        )

        phase = "after_provenance"
        after = build_candidate_run_provenance(research_context_digest=context_digest)
        phase = "after_provenance_journal"
        _publish_record(root / "after", {"provenance": after})

        phase = "verify_provenance"
        if canonical_json_bytes(before) != canonical_json_bytes(after):
            raise RuntimeError("implementation, runtime, or context provenance changed")
        if type(comparison) is not RidgeArtifactComparison:
            raise TypeError("comparison owner returned an invalid result type")

        phase = "publish_result"
        result_digest = publish_ridge_comparison_artifact(
            root / "result",
            comparison,
            provenance=before,
        )
        _require_sha256(result_digest, "result_artifact_digest")

        phase = "strict_load_result"
        loaded = load_ridge_comparison_artifact(
            root / "result",
            expected_digest=result_digest,
        )
        if loaded.artifact_digest != result_digest:
            raise ValueError("strict result loader returned a different digest")

        completed = PublishedRidgeComparisonAttempt(
            root=root,
            result_artifact_digest=result_digest,
        )
        phase = "completion_record"
        _publish_record(
            root / "complete",
            {
                "after_provenance_digest": content_digest(after),
                "before_provenance_digest": content_digest(before),
                "result_artifact_digest": result_digest,
                "schema_version": f"{_ATTEMPT_SCHEMA}_complete",
            },
        )
        return completed
    except BaseException as error:
        if type(comparison) is RidgeArtifactComparison:
            try:
                _publish_record(root / "partial", _partial_record(comparison))
                partial_written = True
            except BaseException as partial_failure:
                partial_error = partial_failure
                _add_failure_note(
                    error,
                    "partial ledger journaling also failed",
                    partial_failure,
                )
        failure = _failure_record(
            phase=phase,
            error=error,
            comparison_returned=comparison is not None,
            partial_written=partial_written,
            partial_error=partial_error,
        )
        try:
            _publish_record(root / "failure", failure)
        except BaseException as failure_error:
            _add_failure_note(error, "failure journaling also failed", failure_error)
        raise


__all__ = [
    "PublishedRidgeComparisonAttempt",
    "run_ridge_comparison_attempt",
]
