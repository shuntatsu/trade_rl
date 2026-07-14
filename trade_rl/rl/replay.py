"""Content-addressed replay-buffer artifacts for off-policy resume."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

REPLAY_FILE = "replay-buffer.pkl"
REPLAY_MANIFEST = "manifest.json"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayBufferManifest:
    artifact_digest: str
    replay_digest: str
    size_bytes: int
    algorithm: str
    environment_digest: str
    training_config_digest: str
    timesteps: int
    schema_version: str = "replay_buffer_artifact_v1"

    def __post_init__(self) -> None:
        require_sha256(self.artifact_digest, field="artifact_digest")
        require_sha256(self.replay_digest, field="replay_digest")
        require_sha256(self.environment_digest, field="environment_digest")
        require_sha256(self.training_config_digest, field="training_config_digest")
        if self.algorithm not in {"sac", "td3", "tqc"}:
            raise ValueError("replay buffer algorithm must be off-policy")
        if self.size_bytes < 0 or self.timesteps < 0:
            raise ValueError("replay buffer sizes and timesteps must be non-negative")
        if self.artifact_digest != content_digest(self.digest_payload()):
            raise ValueError("replay buffer artifact digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "environment_digest": self.environment_digest,
            "replay_digest": self.replay_digest,
            "replay_file": REPLAY_FILE,
            "schema_version": self.schema_version,
            "size_bytes": self.size_bytes,
            "timesteps": self.timesteps,
            "training_config_digest": self.training_config_digest,
        }


def write_replay_buffer_artifact(
    root: str | Path,
    *,
    source: str | Path,
    algorithm: str,
    environment_digest: str,
    training_config_digest: str,
    timesteps: int,
) -> ReplayBufferManifest:
    output = Path(root)
    if output.exists():
        raise FileExistsError(f"replay artifact destination exists: {output}")
    source_path = Path(source)
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError("replay buffer source must be a regular file")
    output.mkdir(parents=True)
    replay_path = output / REPLAY_FILE
    shutil.copyfile(source_path, replay_path)
    payload = {
        "algorithm": algorithm,
        "environment_digest": environment_digest,
        "replay_digest": _digest(replay_path),
        "replay_file": REPLAY_FILE,
        "schema_version": "replay_buffer_artifact_v1",
        "size_bytes": replay_path.stat().st_size,
        "timesteps": timesteps,
        "training_config_digest": training_config_digest,
    }
    manifest = ReplayBufferManifest(
        artifact_digest=content_digest(payload),
        replay_digest=str(payload["replay_digest"]),
        size_bytes=int(payload["size_bytes"]),
        algorithm=algorithm,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
        timesteps=timesteps,
    )
    temporary = output / f".{REPLAY_MANIFEST}.tmp"
    temporary.write_bytes(canonical_json_bytes(asdict(manifest)))
    os.replace(temporary, output / REPLAY_MANIFEST)
    return manifest


def load_replay_buffer_artifact(
    root: str | Path,
) -> tuple[ReplayBufferManifest, Path]:
    path = Path(root)
    actual = {item.name for item in path.iterdir()}
    if actual != {REPLAY_FILE, REPLAY_MANIFEST}:
        raise ValueError("replay artifact contains undeclared or missing files")
    if any(item.is_symlink() for item in path.iterdir()):
        raise ValueError("replay artifact must not contain symlinks")
    raw = json.loads((path / REPLAY_MANIFEST).read_text(encoding="utf-8"))
    manifest = ReplayBufferManifest(**raw)
    replay = path / REPLAY_FILE
    if replay.stat().st_size != manifest.size_bytes:
        raise ValueError("replay buffer size mismatch")
    if _digest(replay) != manifest.replay_digest:
        raise ValueError("replay buffer digest mismatch")
    return manifest, replay


__all__ = [
    "ReplayBufferManifest",
    "load_replay_buffer_artifact",
    "write_replay_buffer_artifact",
]
