"""Exclusive canonical files for existing frozen direct-simple predictions.

Parents are caller-trusted namespaces. Publication requires hard-link support;
it does not authenticate historical fitting or guarantee power-loss durability.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream


@dataclass(frozen=True, slots=True)
class PublishedSimpleReturnStreamArtifact:
    """Location and externally retainable identities of a published stream."""

    path: Path
    digest: str
    dataset_id: str


def load_simple_return_stream_artifact(
    path: str | Path,
    *,
    expected_digest: str,
    expected_dataset_id: str,
) -> FrozenSimpleReturnStream:
    """Verify one regular-file snapshot using both externally supplied pins.

    The existing reader checks frozen projections without refitting. Adjacent
    files are not authorities; no manifest or default pin is consulted.
    Stationary special files and final symlinks are rejected before opening.
    """
    require_sha256(expected_digest, field="expected_digest")
    require_sha256(expected_dataset_id, field="expected_dataset_id")
    source = Path(path)
    if not stat.S_ISREG(source.lstat().st_mode):
        raise ValueError("simple-return stream must be a regular file")
    with open_regular_binary(source, field="simple-return stream") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("simple-return stream file digest mismatch")
    payload = json.loads(raw)
    if canonical_json_bytes(payload) != raw:
        raise ValueError("simple-return stream file must contain canonical JSON")
    restored = FrozenSimpleReturnStream.from_payload(
        payload, expected_digest=expected_digest
    )
    if restored.dataset_id != expected_dataset_id:
        raise ValueError("simple-return stream Dataset identity mismatch")
    return restored


def publish_simple_return_stream_artifact(
    path: str | Path, stream: FrozenSimpleReturnStream
) -> PublishedSimpleReturnStreamArtifact:
    """Commit a complete staged stream with one exclusive atomic hard link.

    A successful link is the commit point. Ordinary temporary-file cleanup
    errors after it do not invalidate publication and may leave an owned file.
    Unsupported filesystems fail closed; no replacement fallback is attempted.
    Asynchronous interruption and malicious parent replacement are not covered.
    """
    if not isinstance(stream, FrozenSimpleReturnStream):
        raise ValueError("publication requires a frozen simple-return stream")
    payload = stream.payload()
    raw = canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    restored = FrozenSimpleReturnStream.from_payload(payload, expected_digest=digest)
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"simple-return stream destination exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.staging-", dir=output.parent
    )
    staging = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        load_simple_return_stream_artifact(
            staging, expected_digest=digest, expected_dataset_id=restored.dataset_id
        )
        record = PublishedSimpleReturnStreamArtifact(
            output, digest, restored.dataset_id
        )
        os.link(staging, output)
        return record
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            staging.unlink()
        except OSError:
            pass


__all__ = [
    "PublishedSimpleReturnStreamArtifact",
    "load_simple_return_stream_artifact",
    "publish_simple_return_stream_artifact",
]
