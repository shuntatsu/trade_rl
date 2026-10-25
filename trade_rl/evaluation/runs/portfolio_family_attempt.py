"""Write-once persistence wrapper for the fixed four-family replay owner."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.evaluation.runs.portfolio_family_comparison import (
    _FAMILY_NAMES,
    FourFamilyPortfolioReplayResult,
    PortfolioFamilyReplayOutcome,
    run_four_family_shared_portfolio_replay,
)
from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
    FourFamilyPortfolioAttemptIdentity,
    LoadedFourFamilyPortfolioResultArtifact,
    _family_payload,
    _require_object,
    _strict_json,
    four_family_portfolio_attempt_identity_digest,
    four_family_portfolio_attempt_identity_payload,
    load_four_family_portfolio_result_artifact,
    publish_four_family_portfolio_result_artifact,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance


@dataclass(frozen=True, slots=True)
class FourFamilyPortfolioAttemptContext:
    """Declared input roots and fixed configuration for one new attempt."""

    identity: FourFamilyPortfolioAttemptIdentity
    dataset_artifact_root: str | Path
    ridge_artifact_root: str | Path
    bundle_root: str | Path

    def __post_init__(self) -> None:
        if type(self.identity) is not FourFamilyPortfolioAttemptIdentity:
            raise TypeError("identity must be a FourFamilyPortfolioAttemptIdentity")
        for name in (
            "dataset_artifact_root",
            "ridge_artifact_root",
            "bundle_root",
        ):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a non-empty path")


@dataclass(frozen=True, slots=True)
class PublishedFourFamilyPortfolioAttempt:
    """Completed write-once attempt directory and its result digest."""

    root: Path
    result_artifact_digest: str


@dataclass(frozen=True, slots=True)
class LoadedFourFamilyPortfolioAttempt:
    attempt: PublishedFourFamilyPortfolioAttempt
    result: LoadedFourFamilyPortfolioResultArtifact


def _absolute_root(value: str | Path) -> Path:
    root = Path(value).expanduser().absolute()
    if not root.name:
        raise ValueError("attempt_root must name a directory below its parent")
    return root


def _declared_root(value: str | Path) -> str:
    return Path(value).expanduser().absolute().as_posix()


def _reserve(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"attempt destination already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()


def _publish_record(path: Path, record: object) -> bytes:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"attempt record already exists: {path}")
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{path.parent.name}.staging-",
            dir=path.parent.parent,
        )
    )
    raw = canonical_json_bytes(record)
    try:
        atomic_write_bytes(staging / path.name, raw)
        atomic_rename_directory(staging, path.parent)
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()
        raise
    return raw


def _validate_provenance(
    provenance: Mapping[str, object], identity: FourFamilyPortfolioAttemptIdentity
) -> None:
    if (
        provenance.get("implementation_digest")
        != identity.expected_implementation_digest
    ):
        raise ValueError("implementation provenance differs from expected digest")
    if (
        provenance.get("runtime_environment_digest")
        != identity.expected_runtime_environment_digest
    ):
        raise ValueError("runtime provenance differs from expected digest")
    if provenance.get("research_context_digest") != identity.research_context_digest:
        raise ValueError("research context differs from expected provenance")


def _failure_note(original: BaseException, secondary: BaseException) -> None:
    add_note = getattr(original, "add_note", None)
    if callable(add_note):
        add_note(
            f"attempt failure journal also failed: {type(secondary).__name__}: {secondary}"
        )


def _read_regular_bytes(path: Path, *, field: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    with open_regular_binary(path, field=field) as handle:
        return handle.read()


def run_four_family_portfolio_attempt(
    *,
    attempt_root: str | Path,
    context: FourFamilyPortfolioAttemptContext,
) -> PublishedFourFamilyPortfolioAttempt:
    """Reserve, execute, journal each completed arm, and publish a strict result.

    This wrapper owns a one-shot attempt directory but no financial state. The
    existing Stage4a comparison remains the sole Dataset/strategy/risk/executor/
    ledger owner. Completed family rows become durable in its callback before
    the next family starts.
    """
    if type(context) is not FourFamilyPortfolioAttemptContext:
        raise TypeError("context must be a FourFamilyPortfolioAttemptContext")
    identity = context.identity
    identity_payload = four_family_portfolio_attempt_identity_payload(identity)
    root = _absolute_root(attempt_root)
    roots = {
        "bundle_root": _declared_root(context.bundle_root),
        "dataset_artifact_root": _declared_root(context.dataset_artifact_root),
        "ridge_artifact_root": _declared_root(context.ridge_artifact_root),
    }
    _reserve(root)

    phase = "started_record"
    completed_families: list[str] = []
    completed_outcomes: list[PortfolioFamilyReplayOutcome] = []
    before: Mapping[str, object] | None = None
    after: Mapping[str, object] | None = None
    comparison: FourFamilyPortfolioReplayResult | None = None
    result_digest: str | None = None
    complete_written = False
    try:
        _publish_record(
            root / "started" / "record.json",
            {
                "declared_roots": roots,
                "identity": identity_payload,
                "schema_version": "four_family_portfolio_attempt_started_v1",
            },
        )

        phase = "before_provenance"
        before = build_candidate_run_provenance(
            research_context_digest=identity.research_context_digest
        )
        _validate_provenance(before, identity)
        _publish_record(
            root / "provenance" / "before" / "record.json",
            {
                "provenance": before,
                "schema_version": "four_family_portfolio_provenance_v1",
            },
        )

        phase = "four_family_comparison"

        def persist_completed(outcome: PortfolioFamilyReplayOutcome) -> None:
            expected_position = len(completed_families)
            if expected_position >= len(_FAMILY_NAMES):
                raise ValueError("comparison returned more than four family outcomes")
            expected_family = _FAMILY_NAMES[expected_position]
            if (
                type(outcome) is not PortfolioFamilyReplayOutcome
                or outcome.family != expected_family
            ):
                raise ValueError(
                    "comparison family callback violated fixed family order"
                )
            row = _family_payload(outcome)
            _publish_record(root / "families" / expected_family / "record.json", row)
            completed_families.append(expected_family)
            completed_outcomes.append(outcome)

        comparison = run_four_family_shared_portfolio_replay(
            dataset_artifact_root=context.dataset_artifact_root,
            expected_dataset_artifact_sha256=identity.expected_dataset_artifact_sha256,
            expected_symbols=identity.expected_symbols,
            ridge_artifact_root=context.ridge_artifact_root,
            expected_ridge_artifact_sha256=identity.ridge_artifact_sha256,
            expected_ridge_binding=identity.expected_ridge_binding,
            bundle_root=context.bundle_root,
            expected_bundle_sha256=identity.expected_bundle_sha256,
            expected_manifest_sha256=identity.expected_manifest_sha256,
            expected_completion_sha256=identity.expected_completion_sha256,
            expected_training_context_sha256=identity.expected_training_context_sha256,
            expected_checkpoint_sha256=identity.expected_checkpoint_sha256,
            bound=identity.bound,
            start_index=identity.start_index,
            stop_index=identity.stop_index,
            initial_capital=identity.initial_capital,
            target_abs_weights=identity.target_abs_weights,
            trend_config=identity.trend_config,
            ridge_entry_threshold=identity.ridge_entry_threshold,
            ridge_exit_threshold=identity.ridge_exit_threshold,
            ridge_one_way_switch_cost=identity.ridge_one_way_switch_cost,
            execution_cost=identity.execution_cost,
            risk_config=identity.risk_config,
            market_order_profile=identity.market_order_profile,
            on_family_completed=persist_completed,
        )
        if type(comparison) is not FourFamilyPortfolioReplayResult:
            raise TypeError("comparison owner returned an invalid result type")
        if tuple(completed_families) != _FAMILY_NAMES:
            raise ValueError(
                "comparison did not persist all four fixed family outcomes"
            )
        if tuple(item.family for item in comparison.families) != _FAMILY_NAMES:
            raise ValueError("comparison result family order differs from fixed roster")
        if any(
            comparison.families[index] is not outcome
            for index, outcome in enumerate(completed_outcomes)
        ):
            raise ValueError(
                "comparison returned family outcomes not observed by callback"
            )

        phase = "after_provenance"
        after = build_candidate_run_provenance(
            research_context_digest=identity.research_context_digest
        )
        _validate_provenance(after, identity)
        _publish_record(
            root / "provenance" / "after" / "record.json",
            {
                "provenance": after,
                "schema_version": "four_family_portfolio_provenance_v1",
            },
        )
        phase = "verify_provenance"
        if canonical_json_bytes(before) != canonical_json_bytes(after):
            raise RuntimeError("implementation, runtime, or context provenance changed")

        phase = "publish_result"
        result_digest = publish_four_family_portfolio_result_artifact(
            root,
            comparison,
            identity,
            before_provenance=before,
            after_provenance=after,
        )
        phase = "strict_result_reload"
        loaded = load_four_family_portfolio_result_artifact(
            root,
            expected_digest=result_digest,
            expected_identity=identity,
        )
        if loaded.artifact_digest != result_digest:
            raise ValueError("strict result loader returned a different digest")

        phase = "completion_record"
        identity_digest = four_family_portfolio_attempt_identity_digest(identity)
        _publish_record(
            root / "complete" / "record.json",
            {
                "after_provenance_digest": content_digest(after),
                "attempt_id": identity.attempt_id,
                "before_provenance_digest": content_digest(before),
                "expected_identity_digest": identity_digest,
                "result_digest": result_digest,
                "schema_version": "four_family_portfolio_attempt_complete_v1",
            },
        )
        complete_written = True
        phase = "strict_attempt_reload"
        load_four_family_portfolio_attempt(root, expected_identity=identity)
        return PublishedFourFamilyPortfolioAttempt(root, result_digest)
    except BaseException as error:
        failure = {
            "comparison_returned": comparison is not None,
            "completed_families": list(completed_families),
            "exception_message": str(error),
            "exception_type": type(error).__name__,
            "phase": phase,
            "result_digest": result_digest,
            "schema_version": "four_family_portfolio_attempt_failure_v1",
        }
        if complete_written:
            failure["completion_marker_note"] = (
                "completion was written before a later strict reload failed; loader rejects failure/completion conflicts"
            )
        try:
            _publish_record(root / "failure" / "record.json", failure)
        except BaseException as journal_error:
            _failure_note(error, journal_error)
        raise


def load_four_family_portfolio_attempt(
    attempt_root: str | Path,
    *,
    expected_identity: FourFamilyPortfolioAttemptIdentity,
) -> LoadedFourFamilyPortfolioAttempt:
    """Load only a complete, internally consistent, nonfailed attempt."""
    root = _absolute_root(attempt_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("attempt root must be a regular directory")
    failure_path = root / "failure" / "record.json"
    if failure_path.exists() or failure_path.is_symlink():
        raise ValueError("failed attempt directories cannot be loaded as complete")
    complete_raw = _read_regular_bytes(
        root / "complete" / "record.json", field="completion record"
    )
    complete = _require_object(
        _strict_json(complete_raw, field="completion record"),
        field="completion record",
        keys={
            "after_provenance_digest",
            "attempt_id",
            "before_provenance_digest",
            "expected_identity_digest",
            "result_digest",
            "schema_version",
        },
    )
    if complete["schema_version"] != "four_family_portfolio_attempt_complete_v1":
        raise ValueError("completion record schema is unsupported")
    identity_digest = four_family_portfolio_attempt_identity_digest(expected_identity)
    if (
        complete["attempt_id"] != expected_identity.attempt_id
        or complete["expected_identity_digest"] != identity_digest
    ):
        raise ValueError("completion record identity differs from expected attempt")
    result = load_four_family_portfolio_result_artifact(
        root,
        expected_digest=cast(str, complete["result_digest"]),
        expected_identity=expected_identity,
    )
    if (
        complete["before_provenance_digest"]
        != result.payload["before_provenance_digest"]
        or complete["after_provenance_digest"]
        != result.payload["after_provenance_digest"]
    ):
        raise ValueError("completion provenance digests differ from result")
    inventory: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("attempt tree must not contain symlinks")
        if path.is_file():
            inventory.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("attempt tree contains a non-regular entry")
    expected = {
        "started/record.json",
        "provenance/before/record.json",
        "provenance/after/record.json",
        "result/manifest.json",
        "complete/record.json",
        *(f"families/{name}/record.json" for name in _FAMILY_NAMES),
    }
    if inventory != expected:
        raise ValueError("completed attempt tree has missing or unexpected files")
    published = PublishedFourFamilyPortfolioAttempt(
        root=root,
        result_artifact_digest=result.artifact_digest,
    )
    return LoadedFourFamilyPortfolioAttempt(attempt=published, result=result)


__all__ = [
    "FourFamilyPortfolioAttemptContext",
    "LoadedFourFamilyPortfolioAttempt",
    "PublishedFourFamilyPortfolioAttempt",
    "load_four_family_portfolio_attempt",
    "run_four_family_portfolio_attempt",
]
