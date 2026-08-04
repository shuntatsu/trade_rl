"""Fail-closed Studio projection of live behavior-cloning progress."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from trade_rl.studio.contracts import JobSummary, StudioModel
from trade_rl.studio.errors import ArtifactInvalid
from trade_rl.studio.settings import StudioSettings

_SEED = re.compile(r"^seed-(\d+)$")
_PROGRESS_NAME = "behavior-cloning-progress.json"


class BehaviorCloningProgressResponse(StudioModel):
    schema_version: Literal["behavior_cloning_progress_v1"] = "behavior_cloning_progress_v1"
    available: bool
    phase: Literal["not_started", "preparing", "training", "evaluating", "passed", "failed"]
    epoch: int | None = Field(default=None, ge=0)
    total_epochs: int | None = Field(default=None, ge=1)
    best_epoch: int | None = Field(default=None, ge=0)
    percent: float | None = Field(default=None, ge=0.0, le=100.0)
    seed: int | None = Field(default=None, ge=0)
    fold: str | None = None
    configuration: str | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    estimated_remaining_seconds: float | None = Field(default=None, ge=0.0)
    validation_loss: float | None = Field(default=None, ge=0.0)
    gate_loss: float | None = Field(default=None, ge=0.0)
    target_loss: float | None = Field(default=None, ge=0.0)
    composed_loss: float | None = Field(default=None, ge=0.0)
    gate_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    gate_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    activity_ratio: float | None = Field(default=None, ge=0.0)
    all_hold_collapse: bool | None = None
    all_trade_collapse: bool | None = None
    early_stopping: bool | None = None
    updated_at: str | None = None
    source: str | None = None


class StudioBehaviorCloningProgressReader:
    def __init__(self, settings: StudioSettings) -> None:
        self.settings = settings

    def _artifact_root(self, job: JobSummary) -> Path:
        root = (self.settings.project_root / job.artifact_root).resolve()
        try:
            root.relative_to(self.settings.project_root.resolve())
        except ValueError as error:
            raise ArtifactInvalid("job artifact root escapes the Studio project") from error
        return root

    @staticmethod
    def _identity(path: Path) -> tuple[int | None, str | None, str | None]:
        seed: int | None = None
        fold: str | None = None
        configuration: str | None = None
        for part in path.parts:
            match = _SEED.fullmatch(part)
            if match is not None:
                seed = int(match.group(1))
            elif part.startswith("fold-"):
                fold = part
            elif part.startswith("configuration-"):
                configuration = part
        return seed, fold, configuration

    def _source(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.settings.project_root.resolve()).as_posix()
        except ValueError as error:
            raise ArtifactInvalid("behavior-cloning source is outside the project") from error

    @staticmethod
    def _latest(paths: list[Path]) -> Path | None:
        return max(paths, key=lambda path: path.stat().st_mtime_ns, default=None)

    def _run_roots(self, job: JobSummary) -> tuple[Path, ...]:
        root = self._artifact_root(job)
        resolved: list[Path] = []
        for namespace in (".staging", "runs", "failed"):
            candidate = (root / namespace / job.run_id).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise ArtifactInvalid("behavior-cloning run path escapes artifact root") from error
            if candidate.is_dir():
                resolved.append(candidate)
        return tuple(resolved)

    def inspect(self, job: JobSummary) -> BehaviorCloningProgressResponse:
        run_roots = self._run_roots(job)
        progress_paths = [
            path
            for root in run_roots
            for path in root.rglob(_PROGRESS_NAME)
            if path.is_file() and not path.is_symlink()
        ]
        progress_path = self._latest(progress_paths)
        if progress_path is not None:
            try:
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArtifactInvalid("behavior-cloning progress is invalid") from error
            if not isinstance(payload, dict) or payload.get("schema_version") != "behavior_cloning_progress_v1":
                raise ArtifactInvalid("behavior-cloning progress schema is invalid")
            seed, fold, configuration = self._identity(progress_path)
            epoch = payload.get("epoch")
            total_epochs = payload.get("total_epochs")
            percent = (
                min(100.0, 100.0 * epoch / total_epochs)
                if isinstance(epoch, int) and isinstance(total_epochs, int) and total_epochs > 0
                else None
            )
            return BehaviorCloningProgressResponse.model_validate(
                {
                    **payload,
                    "available": True,
                    "percent": percent,
                    "seed": payload.get("seed", seed),
                    "fold": fold,
                    "configuration": configuration,
                    "source": self._source(progress_path),
                }
            )

        teacher_paths = [
            path
            for root in run_roots
            for path in root.rglob("manifest.json")
            if path.parent.name == "teacher" and path.is_file() and not path.is_symlink()
        ]
        teacher_path = self._latest(teacher_paths)
        if teacher_path is not None:
            seed, fold, configuration = self._identity(teacher_path)
            return BehaviorCloningProgressResponse(
                available=True,
                phase="preparing",
                seed=seed,
                fold=fold,
                configuration=configuration,
                source=self._source(teacher_path),
            )
        return BehaviorCloningProgressResponse(available=False, phase="not_started")


__all__ = ["BehaviorCloningProgressResponse", "StudioBehaviorCloningProgressReader"]
