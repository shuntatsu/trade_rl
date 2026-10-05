"""Write-once external trace sidecars verified without PPO deserialization."""

from __future__ import annotations

import json
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.atomic_write import atomic_rename_directory
from trade_rl.artifacts.verified_file import file_digest, open_regular_binary
from trade_rl.evaluation.rl_allocation.transition_trace import (
    AllocationTransitionRecorder,
)
from trade_rl.evaluation.rl_allocation.transition_validation import (
    validate_transition_events,
)
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest


def _directory(root: Path, files: set[str]) -> None:
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != files
    ):
        raise ValueError("transition/bundle requires an exact regular directory")


def _canonical(path: Path) -> tuple[bytes, dict[str, Any]]:
    with open_regular_binary(path, field="transition manifest") as stream:
        raw = stream.read()
    try:
        value = json.loads(raw)
        if type(value) is not dict or canonical_json_bytes(value) != raw:
            raise ValueError("transition manifest must contain canonical JSON")
        return raw, value
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("invalid transition manifest bytes") from error


def _bundle(root: Path, pin: str) -> dict[str, Any]:
    require_sha256(pin, field="expected_bundle_digest")
    _directory(root, {"manifest.json", "policy.zip"})
    _, manifest = _canonical(root / "manifest.json")
    if content_digest(manifest) != pin:
        raise ValueError("transition bundle differs from external pin")
    validated = validate_allocation_manifest(manifest, require_policy=True)
    if file_digest(root / "policy.zip") != validated["policy_sha256"]:
        raise ValueError("transition policy bytes differ from saved bundle")
    return validated


def _manifest(
    bundle: dict[str, Any], bundle_pin: str, events: list[dict[str, Any]], raw: bytes
) -> dict[str, Any]:
    dimensions = validate_transition_events(events, bundle)
    training = bundle["training"]
    return {
        "schema": "allocation_ppo_transition_trace_v1",
        "status": "COMPLETE",
        "bundle_digest": bundle_pin,
        "policy_sha256": bundle["policy_sha256"],
        "recipe_digest": bundle["recipe_digest"],
        "training_digest": content_digest(training),
        "source_digest": content_digest(training["source"]),
        "protocol_digest": training["protocol_digest"],
        "input_digest": content_digest(training["observation_consumption"]),
        "optimization_digest": content_digest(training["optimization"]),
        "observation_digest": content_digest(bundle["recipe"]["observation"]),
        "tensor_encoding": "little_endian_float32_int64_hex_v1",
        "admission": "chronological_full_rollout_before_get_v1",
        "completion": "after_validated_final_policy_v1",
        "transitions_sha256": sha256(raw).hexdigest(),
        **dimensions,
    }


def publish_allocation_transition_trace(
    recorder: AllocationTransitionRecorder,
    root: Path,
    *,
    bundle_root: Path,
    expected_bundle_digest: str,
) -> str:
    """Publish only a completed observer matched to an already saved bundle."""
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise FileExistsError("transition trace destination already exists")
    root = root.resolve()
    if root.exists():
        raise FileExistsError("transition trace destination already exists")
    bundle_root = Path(bundle_root)
    if root.is_relative_to(bundle_root.resolve()):
        raise ValueError(
            "transition sidecar must remain separate from its saved bundle"
        )
    bundle = _bundle(bundle_root, expected_bundle_digest)
    finalized = dict(bundle)
    del finalized["policy_sha256"]
    if recorder.training_manifest != finalized:
        raise ValueError("transition trace belongs to a different successful fit")
    raw = b"".join(event + b"\n" for event in recorder.events)
    manifest = _manifest(
        bundle,
        expected_bundle_digest,
        [json.loads(event) for event in recorder.events],
        raw,
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    try:
        (staging / "transitions.jsonl").write_bytes(raw)
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        atomic_rename_directory(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return content_digest(manifest)


def read_allocation_transition_trace(
    root: Path, *, expected_digest: str, bundle_root: Path, expected_bundle_digest: str
) -> dict[str, Any]:
    """Closed reader; hashes/relations do not prove who produced these bytes."""
    require_sha256(expected_digest, field="expected_digest")
    root = Path(root)
    _directory(root, {"manifest.json", "transitions.jsonl"})
    _, manifest = _canonical(root / "manifest.json")
    if content_digest(manifest) != expected_digest:
        raise ValueError("transition manifest differs from external pin")
    with open_regular_binary(
        root / "transitions.jsonl", field="transition rows"
    ) as stream:
        raw = stream.read()
    try:
        lines = raw.splitlines(keepends=True)
        events = [json.loads(line) for line in lines]
        if b"".join(canonical_json_bytes(event) + b"\n" for event in events) != raw:
            raise ValueError("transition rows must preserve canonical JSONL bytes")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("invalid transition row bytes") from error
    bundle = _bundle(Path(bundle_root), expected_bundle_digest)
    expected = _manifest(bundle, expected_bundle_digest, events, raw)
    if canonical_json_bytes(expected) != canonical_json_bytes(manifest):
        raise ValueError("transition manifest differs from its closed bundle/row links")
    return manifest


__all__ = ["publish_allocation_transition_trace", "read_allocation_transition_trace"]
