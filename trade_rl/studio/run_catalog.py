"""Validated run discovery with collision-free Studio resource identities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    WalkForwardRunManifest,
    validate_training_run_directory,
    validate_walk_forward_run_directory,
)
from trade_rl.studio.catalog_cache import CatalogCache
from trade_rl.studio.catalog_common import (
    fingerprint_identity,
    mtime,
    read_json,
    stat_fingerprint,
)
from trade_rl.studio.contracts import RunSummary
from trade_rl.studio.errors import ArtifactInvalid, ResourceNotFound
from trade_rl.studio.resource_ids import require_resource_id, resource_id
from trade_rl.studio.settings import StudioSettings

RunManifest = TrainingRunManifest | WalkForwardRunManifest


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    path: Path
    summary: RunSummary
    manifest: RunManifest


@dataclass(frozen=True, slots=True)
class _RunEntry:
    summary: RunSummary
    manifest: RunManifest | None


class RunCatalog:
    def __init__(self, settings: StudioSettings, cache: CatalogCache[object]) -> None:
        self.settings = settings
        self.cache = cache

    def _directories(self) -> tuple[Path, ...]:
        directories: set[Path] = set()
        for root in self.settings.run_roots:
            runs_root = root / "runs"
            if runs_root.is_dir():
                directories.update(
                    path for path in runs_root.iterdir() if path.is_dir()
                )
        return tuple(sorted(directories, key=lambda item: item.as_posix()))

    def _fingerprint(self, path: Path) -> tuple[object, ...]:
        raw = read_json(path / "run.json")
        files = None if raw is None else raw.get("files")
        declared: list[tuple[str, tuple[bool, int, int]]] = []
        if isinstance(files, list):
            for item in files:
                mapped = _mapping(item)
                relative = None if mapped is None else mapped.get("path")
                if isinstance(relative, str):
                    declared.append((relative, stat_fingerprint(path / relative)))
        if not declared:
            declared = [
                (item.relative_to(path).as_posix(), stat_fingerprint(item))
                for item in sorted(path.rglob("*"))
                if item.is_file() and item.name != "run.json"
            ]
        return (stat_fingerprint(path / "run.json"), tuple(declared))

    def _validate(self, path: Path) -> RunManifest:
        try:
            return validate_training_run_directory(path)
        except (OSError, ValueError, TypeError) as training_error:
            try:
                return validate_walk_forward_run_directory(path)
            except (OSError, ValueError, TypeError):
                raise training_error

    def _algorithm(self, path: Path, manifest: RunManifest) -> str:
        if isinstance(manifest, WalkForwardRunManifest):
            return "walk-forward"
        payload = read_json(path / "training-config.json")
        training = None if payload is None else _mapping(payload.get("training"))
        algorithm = None if training is None else training.get("algorithm")
        return algorithm if isinstance(algorithm, str) and algorithm else "unknown"

    def _entry(self, path: Path) -> _RunEntry:
        relative = self.settings.relative_path(path)
        fingerprint = self._fingerprint(path)

        def build() -> _RunEntry:
            try:
                manifest = self._validate(path)
                walk_forward = read_json(path / "walk-forward.json")
                selected_metrics = (
                    None
                    if walk_forward is None
                    else _mapping(walk_forward.get("selected_metrics"))
                )
                completed = (
                    manifest.completed_at
                    if isinstance(manifest, TrainingRunManifest)
                    else manifest.created_at
                )
                run_kind = (
                    manifest.run_kind
                    if isinstance(manifest, TrainingRunManifest)
                    else "walk_forward_evaluation"
                )
                return _RunEntry(
                    summary=RunSummary(
                        id=resource_id("run", relative, manifest.digest),
                        run_id=manifest.run_id,
                        manifest_digest=manifest.digest,
                        relative_path=relative,
                        run_kind=run_kind,
                        algorithm=self._algorithm(path, manifest),
                        dataset_id=manifest.dataset_id,
                        period=f"{manifest.created_at.date()} — {completed.date()}",
                        created_at=manifest.created_at.isoformat(),
                        completed_at=completed.isoformat(),
                        file_count=len(manifest.files),
                        sharpe=(
                            None
                            if selected_metrics is None
                            else _number(selected_metrics.get("sharpe"))
                        ),
                        max_drawdown=(
                            None
                            if selected_metrics is None
                            else _number(
                                selected_metrics.get(
                                    "max_drawdown",
                                    selected_metrics.get("maximum_drawdown"),
                                )
                            )
                        ),
                        total_return=(
                            None
                            if selected_metrics is None
                            else _number(selected_metrics.get("total_return"))
                        ),
                        status="VALID",
                    ),
                    manifest=manifest,
                )
            except (OSError, ValueError, TypeError) as error:
                raw = read_json(path / "run.json")
                raw_run_id = None if raw is None else raw.get("run_id")
                run_id_value = raw_run_id if isinstance(raw_run_id, str) else path.name
                identity = fingerprint_identity((relative, fingerprint, run_id_value))
                timestamp = mtime(path)
                return _RunEntry(
                    summary=RunSummary(
                        id=resource_id("run", relative, identity),
                        run_id=run_id_value,
                        relative_path=relative,
                        run_kind="unknown",
                        algorithm="unknown",
                        dataset_id="",
                        period="—",
                        created_at=timestamp,
                        completed_at=timestamp,
                        file_count=0,
                        status="INVALID",
                        validation_error=str(error),
                    ),
                    manifest=None,
                )

        return cast(_RunEntry, self.cache.get("run", path, fingerprint, build))

    def list(self) -> tuple[RunSummary, ...]:
        records = tuple(self._entry(path).summary for path in self._directories())
        return tuple(sorted(records, key=lambda item: item.completed_at, reverse=True))

    def resolve(self, value: str) -> ResolvedRun:
        try:
            require_resource_id(value, kind="run")
        except ValueError as error:
            raise ResourceNotFound(f"unknown Studio run resource: {value}") from error
        for path in self._directories():
            entry = self._entry(path)
            if entry.summary.id != value:
                continue
            if entry.summary.status != "VALID" or entry.manifest is None:
                raise ArtifactInvalid(
                    entry.summary.validation_error or "run artifact is invalid"
                )
            return ResolvedRun(
                path=path, summary=entry.summary, manifest=entry.manifest
            )
        raise ResourceNotFound(f"unknown Studio run resource: {value}")

    def resolve_for_evidence(self, value: str) -> Path:
        try:
            require_resource_id(value, kind="run")
        except ValueError as error:
            raise ResourceNotFound(f"unknown Studio run resource: {value}") from error
        for path in self._directories():
            if self._entry(path).summary.id == value:
                return path
        raise ResourceNotFound(f"unknown Studio run resource: {value}")


__all__ = ["ResolvedRun", "RunCatalog", "RunManifest"]
