"""Prospective-only JSON evidence publication on the existing StudyStore primitive.

This API is for future, separately authorized research protocols. It is not a
migration, repair, rerun, or reinterpretation of any consumed/frozen stage.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import read_verified_bytes
from trade_rl.evaluation.experiments.errors import PublishedArtifactCleanupError
from trade_rl.evaluation.experiments.store import StudyStore

MAX_JSON_EVIDENCE_BYTES = 1024 * 1024
_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,79}\.json", re.ASCII)


@dataclass(frozen=True, slots=True)
class PublishedJsonEvidence:
    """Verified local bytes; not a signature, independent anchor, or run grant."""

    path: Path
    sha256: str
    size_bytes: int


def publish_json_evidence_once(
    *,
    root: Path | str,
    name: str,
    payload: Mapping[str, object],
) -> PublishedJsonEvidence:
    """Atomically publish a small canonical JSON leaf without replacing evidence.

    The shared StudyStore handles staging, exclusive hard-link publication,
    no-follow directory capabilities on supported POSIX hosts and collision
    refusal. Portable parent-swap limitations remain. This wrapper binds
    the exact expected canonical bytes to a *second* descriptor-safe readback.

    If readback fails *after* StudyStore returns, publication may already have
    committed: raise a do-not-retry error, never delete or overwrite evidence.
    Concurrent/unauthorized mutation after successful readback is outside this
    local-software guarantee. Any future stage needs independent provenance,
    data/clock checks and its own owner research authorization.
    """
    if type(name) is not str or _NAME.fullmatch(name) is None:
        raise ValueError("evidence name must be one lowercase JSON leaf filename")
    if not isinstance(payload, Mapping):
        raise TypeError("evidence payload must be a canonical JSON mapping")
    # Serialize and reject invalid/oversized evidence before creating directories
    # or reserving any output filename.
    encoded = canonical_json_bytes(payload)
    if len(encoded) > MAX_JSON_EVIDENCE_BYTES:
        raise ValueError("evidence JSON size exceeds the prospective 1 MiB ceiling")

    expected_sha = hashlib.sha256(encoded).hexdigest()
    store = StudyStore(root)
    published = store.publish_json_once(name, payload)
    try:
        read_verified_bytes(
            published,
            expected_digest=expected_sha,
            expected_size_bytes=len(encoded),
            field="prospective JSON evidence",
        )
    except Exception as error:
        raise PublishedArtifactCleanupError(
            "prospective evidence may be published but verified readback failed; "
            "do not retry publication"
        ) from error

    return PublishedJsonEvidence(
        path=published,
        sha256=expected_sha,
        size_bytes=len(encoded),
    )


__all__ = [
    "MAX_JSON_EVIDENCE_BYTES",
    "PublishedJsonEvidence",
    "publish_json_evidence_once",
]
