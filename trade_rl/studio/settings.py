"""Filesystem and network settings for the local Trade RL Studio runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolved_roots(
    project_root: Path, raw: str | None, defaults: tuple[str, ...]
) -> tuple[Path, ...]:
    values = (
        defaults
        if raw is None
        else tuple(item for item in raw.split(os.pathsep) if item)
    )
    roots: list[Path] = []
    for value in values:
        path = Path(value)
        resolved = (
            (project_root / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )
        if resolved != project_root and project_root not in resolved.parents:
            raise ValueError("studio roots must remain inside project_root")
        roots.append(resolved)
    return tuple(dict.fromkeys(roots))


@dataclass(frozen=True, slots=True)
class StudioSettings:
    """Resolved local roots accepted by the Studio API."""

    project_root: Path
    dataset_roots: tuple[Path, ...]
    run_roots: tuple[Path, ...]
    config_roots: tuple[Path, ...]
    job_root: Path
    serving_root: Path | None = None
    paper_snapshot_path: Path | None = None

    def __post_init__(self) -> None:
        project_root = self.project_root.resolve()
        object.__setattr__(self, "project_root", project_root)
        for field_name in ("dataset_roots", "run_roots", "config_roots"):
            values = tuple(path.resolve() for path in getattr(self, field_name))
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            for value in values:
                if value != project_root and project_root not in value.parents:
                    raise ValueError(f"{field_name} must remain inside project_root")
            object.__setattr__(self, field_name, values)
        job_root = self.job_root.resolve()
        if job_root != project_root and project_root not in job_root.parents:
            raise ValueError("job_root must remain inside project_root")
        object.__setattr__(self, "job_root", job_root)
        serving_root = (self.serving_root or (project_root / "var/serving")).resolve()
        if serving_root != project_root and project_root not in serving_root.parents:
            raise ValueError("serving_root must remain inside project_root")
        object.__setattr__(self, "serving_root", serving_root)
        paper_snapshot = (
            self.paper_snapshot_path
            or (project_root / "var/studio/paper-inference.json")
        ).resolve()
        if paper_snapshot != project_root and project_root not in paper_snapshot.parents:
            raise ValueError("paper_snapshot_path must remain inside project_root")
        object.__setattr__(self, "paper_snapshot_path", paper_snapshot)

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> StudioSettings:
        resolved_project = (project_root or Path.cwd()).resolve()
        return cls(
            project_root=resolved_project,
            dataset_roots=_resolved_roots(
                resolved_project,
                os.environ.get("TRADE_RL_STUDIO_DATASET_ROOTS"),
                ("artifacts/datasets", "var/quickstart/dataset"),
            ),
            run_roots=_resolved_roots(
                resolved_project,
                os.environ.get("TRADE_RL_STUDIO_RUN_ROOTS"),
                ("artifacts/research", "var/quickstart/artifacts"),
            ),
            config_roots=_resolved_roots(
                resolved_project,
                os.environ.get("TRADE_RL_STUDIO_CONFIG_ROOTS"),
                ("configs", "examples"),
            ),
            job_root=_resolved_roots(
                resolved_project,
                os.environ.get("TRADE_RL_STUDIO_JOB_ROOT"),
                ("var/studio/jobs",),
            )[0],
            serving_root=_resolved_roots(
                resolved_project,
                os.environ.get("TRADE_RL_STUDIO_SERVING_ROOT"),
                ("var/serving",),
            )[0],
            paper_snapshot_path=_resolved_roots(
                resolved_project,
                os.environ.get("TRADE_RL_STUDIO_PAPER_SNAPSHOT"),
                ("var/studio/paper-inference.json",),
            )[0],
        )

    def relative_path(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved != self.project_root and self.project_root not in resolved.parents:
            raise ValueError("path escapes project_root")
        return resolved.relative_to(self.project_root).as_posix()

    def _resolve_under(self, value: str, roots: tuple[Path, ...]) -> Path:
        raw = Path(value)
        if raw.is_absolute():
            raise ValueError("studio paths must be relative")
        resolved = (self.project_root / raw).resolve()
        if not any(root == resolved or root in resolved.parents for root in roots):
            raise ValueError("path is outside configured roots")
        return resolved

    def resolve_dataset_path(self, value: str) -> Path:
        return self._resolve_under(value, self.dataset_roots)

    def resolve_config_path(self, value: str) -> Path:
        return self._resolve_under(value, self.config_roots)

    def resolve_run_root(self, value: str | None = None) -> Path:
        if value is None:
            return self.run_roots[0]
        return self._resolve_under(value, self.run_roots)
