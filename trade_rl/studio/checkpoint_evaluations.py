"""Fail-closed Studio projection of deterministic checkpoint evaluations."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from trade_rl.studio.contracts import JobSummary, StudioModel
from trade_rl.studio.errors import ArtifactInvalid
from trade_rl.studio.settings import StudioSettings

_SELECTION_NAME = "checkpoint-selection.json"
_SELECTION_SCHEMA = "checkpoint_selection_v2_seed_aware"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_FOLD_DIRECTORY = re.compile(r"^fold-\d+$")


class CheckpointEvaluationItemResponse(StudioModel):
    fold: str
    configuration: str
    seed: int = Field(ge=0)
    policy_digest: str
    evaluation_digest: str
    score: float
    total_return: float = Field(gt=-1.0)
    finalist: bool
    checkpoint_range: tuple[int, int]
    source: str


class CheckpointEvaluationsResponse(StudioModel):
    available: bool
    items: tuple[CheckpointEvaluationItemResponse, ...]
    production_status: Literal["NO-GO"] = "NO-GO"


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactInvalid(f"checkpoint evaluation {field} is invalid")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactInvalid(f"checkpoint evaluation {field} is invalid")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ArtifactInvalid(f"checkpoint evaluation {field} is non-finite")
    return resolved


def _total_return(score: float) -> float:
    try:
        value = math.expm1(score)
    except OverflowError as error:
        raise ArtifactInvalid("checkpoint evaluation return overflowed") from error
    if not math.isfinite(value) or value <= -1.0:
        raise ArtifactInvalid("checkpoint evaluation return is invalid")
    return value


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ArtifactInvalid(f"checkpoint evaluation {field} is invalid")
    return value


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactInvalid(f"checkpoint evaluation {field} is invalid")
    return cast(dict[str, Any], value)


class StudioCheckpointEvaluationReader:
    """Read checkpoint-selection evidence only under one known job's run root."""

    def __init__(self, settings: StudioSettings) -> None:
        self.settings = settings

    def _artifact_root(self, job: JobSummary) -> Path:
        root = (self.settings.project_root / job.artifact_root).resolve()
        try:
            root.relative_to(self.settings.project_root.resolve())
        except ValueError as error:
            raise ArtifactInvalid(
                "job artifact root escapes the Studio project"
            ) from error
        return root

    def _paths(self, job: JobSummary) -> tuple[Path, ...]:
        root = self._artifact_root(job)
        selected: dict[str, Path] = {}
        for namespace in (".staging", "runs", "failed"):
            run_root = (root / namespace / job.run_id).resolve()
            try:
                run_root.relative_to(root)
            except ValueError as error:
                raise ArtifactInvalid(
                    "checkpoint evaluation path escapes artifact root"
                ) from error
            if not run_root.is_dir():
                continue
            for candidate in sorted(run_root.rglob(_SELECTION_NAME)):
                resolved = candidate.resolve()
                try:
                    relative = resolved.relative_to(run_root)
                except ValueError as error:
                    raise ArtifactInvalid(
                        "checkpoint evaluation file escapes run root"
                    ) from error
                if resolved.is_file() and not resolved.is_symlink():
                    selected.setdefault(relative.as_posix(), resolved)
        return tuple(selected[key] for key in sorted(selected))

    def _source(self, path: Path) -> str:
        try:
            return path.relative_to(self.settings.project_root.resolve()).as_posix()
        except ValueError as error:
            raise ArtifactInvalid(
                "checkpoint evaluation source is outside the project"
            ) from error

    @staticmethod
    def _checkpoint_range(payload: dict[str, Any]) -> tuple[int, int]:
        value = payload.get("checkpoint_range")
        if not isinstance(value, list) or len(value) != 2:
            raise ArtifactInvalid("checkpoint evaluation range is invalid")
        start = _integer(value[0], field="range start")
        stop = _integer(value[1], field="range stop")
        if start >= stop:
            raise ArtifactInvalid("checkpoint evaluation range is empty")
        return start, stop

    @staticmethod
    def _fold(path: Path) -> str:
        for part in reversed(path.parts):
            if _FOLD_DIRECTORY.fullmatch(part) is not None:
                return part
        raise ArtifactInvalid("checkpoint evaluation fold identity is unavailable")

    def _items(self, path: Path) -> tuple[CheckpointEvaluationItemResponse, ...]:
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactInvalid(
                f"checkpoint evaluation cannot be read: {path.name}"
            ) from error
        payload = _mapping(raw, field="document")
        if payload.get("schema_version") != _SELECTION_SCHEMA:
            raise ArtifactInvalid("checkpoint evaluation schema is unsupported")
        checkpoint_range = self._checkpoint_range(payload)
        candidates = payload.get("candidates")
        finalists = payload.get("seed_finalists")
        if not isinstance(candidates, list) or not isinstance(finalists, list):
            raise ArtifactInvalid("checkpoint evaluation candidates are invalid")
        finalist_scores: dict[tuple[int, str, str], float] = {}
        for raw_finalist in finalists:
            finalist = _mapping(raw_finalist, field="finalist")
            key = (
                _integer(finalist.get("seed"), field="finalist seed"),
                _digest(
                    finalist.get("policy_digest"),
                    field="finalist policy digest",
                ),
                _digest(
                    finalist.get("checkpoint_evaluation_digest"),
                    field="finalist evaluation digest",
                ),
            )
            if key in finalist_scores:
                raise ArtifactInvalid("checkpoint finalist identity is duplicated")
            finalist_scores[key] = _number(
                finalist.get("checkpoint_score"),
                field="finalist score",
            )
        fold = self._fold(path)
        configuration = path.parent.name
        if not configuration:
            raise ArtifactInvalid("checkpoint configuration is unavailable")
        source = self._source(path)
        items: list[CheckpointEvaluationItemResponse] = []
        candidate_keys: set[tuple[int, str, str]] = set()
        for raw_candidate in candidates:
            candidate = _mapping(raw_candidate, field="candidate")
            seed = _integer(candidate.get("seed"), field="seed")
            policy_digest = _digest(
                candidate.get("policy_digest"), field="policy digest"
            )
            evaluation_digest = _digest(
                candidate.get("evaluation_digest"), field="evaluation digest"
            )
            score = _number(candidate.get("score"), field="score")
            key = (seed, policy_digest, evaluation_digest)
            if key in candidate_keys:
                raise ArtifactInvalid("checkpoint candidate identity is duplicated")
            candidate_keys.add(key)
            finalist_score = finalist_scores.get(key)
            if finalist_score is not None and not math.isclose(
                finalist_score,
                score,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ArtifactInvalid(
                    "checkpoint finalist score does not match candidate"
                )
            items.append(
                CheckpointEvaluationItemResponse(
                    fold=fold,
                    configuration=configuration,
                    seed=seed,
                    policy_digest=policy_digest,
                    evaluation_digest=evaluation_digest,
                    score=score,
                    total_return=_total_return(score),
                    finalist=key in finalist_scores,
                    checkpoint_range=checkpoint_range,
                    source=source,
                )
            )
        if not set(finalist_scores).issubset(candidate_keys):
            raise ArtifactInvalid("checkpoint finalist is absent from candidates")
        return tuple(items)

    def inspect(self, job: JobSummary) -> CheckpointEvaluationsResponse:
        items = tuple(item for path in self._paths(job) for item in self._items(path))
        ordered = tuple(
            sorted(
                items,
                key=lambda item: (
                    item.seed,
                    item.fold,
                    item.configuration,
                    not item.finalist,
                    item.policy_digest,
                ),
            )
        )
        return CheckpointEvaluationsResponse(
            available=bool(ordered),
            items=ordered,
        )


__all__ = [
    "CheckpointEvaluationItemResponse",
    "CheckpointEvaluationsResponse",
    "StudioCheckpointEvaluationReader",
]
