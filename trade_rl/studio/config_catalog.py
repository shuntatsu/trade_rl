"""Canonical training-config discovery and identity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.studio.catalog_cache import CatalogCache
from trade_rl.studio.catalog_common import fingerprint_identity, stat_fingerprint
from trade_rl.studio.contracts import ConfigSummary
from trade_rl.studio.errors import ArtifactInvalid, ResourceNotFound
from trade_rl.studio.resource_ids import require_resource_id, resource_id
from trade_rl.studio.settings import StudioSettings


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    path: Path
    summary: ConfigSummary
    config: object


def _load_training_config(path: Path) -> object:
    from trade_rl.workflows.training_run import TrainingRunConfig

    return TrainingRunConfig.from_json(path)


def _ensure_inside_project(config: object, project_root: Path) -> None:
    paths = [getattr(config, "alpha_artifact"), getattr(config, "factor_artifact")]
    paths.extend(path for _, path in getattr(config, "resume_checkpoints"))
    for path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved != project_root and project_root not in resolved.parents:
            raise ValueError(
                "training config references an artifact outside project_root"
            )


class ConfigCatalog:
    def __init__(self, settings: StudioSettings, cache: CatalogCache[object]) -> None:
        self.settings = settings
        self.cache = cache

    def _paths(self) -> tuple[Path, ...]:
        paths: set[Path] = set()
        for root in self.settings.config_roots:
            if root.is_file() and root.suffix == ".json":
                paths.add(root)
            elif root.is_dir():
                paths.update(root.rglob("*.json"))
        return tuple(sorted(paths, key=lambda item: item.as_posix()))

    def _fingerprint(self, path: Path) -> tuple[bool, int, int]:
        return stat_fingerprint(path)

    def _load(self, path: Path) -> tuple[ConfigSummary, object | None]:
        relative = self.settings.relative_path(path)
        fingerprint = self._fingerprint(path)

        def build() -> tuple[ConfigSummary, object | None]:
            try:
                config = _load_training_config(path)
                _ensure_inside_project(config, self.settings.project_root)
                digest = content_digest(getattr(config, "candidate_digest_payload")())
                return (
                    ConfigSummary(
                        id=resource_id("config", relative, digest),
                        config_digest=digest,
                        name=path.stem,
                        relative_path=relative,
                        algorithm=getattr(getattr(config, "training"), "algorithm"),
                        status="VALID",
                    ),
                    config,
                )
            except (ImportError, OSError, ValueError, TypeError) as error:
                identity = fingerprint_identity((relative, fingerprint))
                return (
                    ConfigSummary(
                        id=resource_id("config", relative, identity),
                        config_digest=None,
                        name=path.stem,
                        relative_path=relative,
                        algorithm="unknown",
                        status="INVALID",
                        validation_error=str(error),
                    ),
                    None,
                )

        return cast(
            tuple[ConfigSummary, object | None],
            self.cache.get("config", path, fingerprint, build),
        )

    def list(self) -> tuple[ConfigSummary, ...]:
        return tuple(self._load(path)[0] for path in self._paths())

    def resolve(self, value: str) -> ResolvedConfig:
        try:
            require_resource_id(value, kind="config")
        except ValueError as error:
            raise ResourceNotFound(
                f"unknown Studio config resource: {value}"
            ) from error
        for path in self._paths():
            summary, config = self._load(path)
            if summary.id != value:
                continue
            if (
                summary.status != "VALID"
                or config is None
                or summary.config_digest is None
            ):
                raise ArtifactInvalid(
                    summary.validation_error or "training config is invalid"
                )
            return ResolvedConfig(path=path, summary=summary, config=config)
        raise ResourceNotFound(f"unknown Studio config resource: {value}")


__all__ = ["ConfigCatalog", "ResolvedConfig"]
