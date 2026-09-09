"""Deterministic implementation and runtime provenance for candidate runs."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
from hashlib import sha256
from pathlib import Path
from typing import Final

from trade_rl._validation import require_sha256
from trade_rl.artifacts.hashing import content_digest

PROVENANCE_SCHEMA: Final = "candidate_run_provenance_v1"
_IMPLEMENTATION_SCHEMA: Final = "candidate_run_implementation_v1"
_RUNTIME_SCHEMA: Final = "candidate_run_runtime_v1"
_RUNTIME_DISTRIBUTIONS: Final = (
    "trade-rl",
    "numpy",
    "gymnasium",
    "lightgbm",
    "stable-baselines3",
    "torch",
)


def _implementation_manifest(package_root: Path) -> dict[str, object]:
    """Return a path-independent exact-byte manifest for Python package sources."""

    root = Path(package_root)
    files: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink() or not path.is_file():
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "schema_version": _IMPLEMENTATION_SCHEMA,
        "files": files,
    }


def _implementation_digest(manifest: object) -> str:
    """Return the canonical digest for one implementation manifest."""

    return content_digest(manifest)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_environment_manifest() -> dict[str, object]:
    """Return the fixed runtime/dependency roster used for candidate evidence."""

    return {
        "schema_version": _RUNTIME_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
        },
        "os": {
            "family": platform.system(),
            "release": platform.release(),
        },
        "machine": platform.machine(),
        "packages": {
            name: _distribution_version(name) for name in _RUNTIME_DISTRIBUTIONS
        },
    }


def build_candidate_run_provenance(
    *,
    research_context_digest: str | None = None,
) -> dict[str, object]:
    """Build canonical provenance, optionally bound to a preregistered context."""

    if research_context_digest is not None:
        require_sha256(
            research_context_digest,
            field="research_context_digest",
        )
    package_root = Path(__file__).resolve().parents[2]
    implementation = _implementation_manifest(package_root)
    runtime_environment = runtime_environment_manifest()
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "implementation": implementation,
        "implementation_digest": _implementation_digest(implementation),
        "runtime_environment": runtime_environment,
        "runtime_environment_digest": content_digest(runtime_environment),
        "research_context_digest": research_context_digest,
    }


__all__ = [
    "PROVENANCE_SCHEMA",
    "build_candidate_run_provenance",
    "runtime_environment_manifest",
]
