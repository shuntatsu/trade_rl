"""Write-once external scheduled evidence; closed reading never loads a PPO model."""

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
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.evaluation.rl_allocation.scheduled_transition_trace import (
    ScheduledAllocationTransitionRecorder,
)
from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
    validate_scheduled_transition_events,
)
from trade_rl.evaluation.rl_allocation.transition_trace_io import (
    _bundle,
    _canonical,
    _directory,
)


def _manifest(
    bundle: dict[str, Any],
    pin: str,
    events: list[dict[str, Any]],
    raw: bytes,
    mode: str,
) -> dict[str, Any]:
    dimensions = validate_scheduled_transition_events(
        events, bundle, learner_diagnostics=mode
    )
    training = bundle["training"]
    return {
        "schema": "allocation_ppo_scheduled_transition_trace_v1",
        "status": "COMPLETE",
        "bundle_digest": pin,
        "policy_sha256": bundle["policy_sha256"],
        "recipe_digest": bundle["recipe_digest"],
        "training_digest": content_digest(training),
        "schedule_digest": training["schedule_digest"],
        "source_records_digest": content_digest(training["sources"]),
        "protocol_digest": training["protocol_digest"],
        "optimization_digest": content_digest(training["optimization"]),
        "observation_digest": content_digest(bundle["recipe"]["observation"]),
        "tensor_encoding": "little_endian_float32_int64_hex_v1",
        "admission": "chronological_full_rollout_before_get_v1",
        "completion": "after_validated_scheduled_policy_v1",
        "transitions_sha256": sha256(raw).hexdigest(),
        **dimensions,
    }


def publish_scheduled_allocation_transition_trace(
    recorder: ScheduledAllocationTransitionRecorder,
    root: Path,
    *,
    bundle_root: Path,
    expected_bundle_digest: str,
) -> str:
    """Only successful detached fit; raw model SHA is not fitting authentication."""
    if type(recorder) is not ScheduledAllocationTransitionRecorder:
        raise ValueError("scheduled sidecar requires its exact completed producer")
    root, bundle_root = Path(root), Path(bundle_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError("scheduled sidecar destination already exists")
    root = root.resolve()
    if root.is_relative_to(bundle_root.resolve()):
        raise ValueError("scheduled sidecar must remain separate from its saved bundle")
    recorder.validate_current_parameters()
    bundle = _bundle(bundle_root, expected_bundle_digest)
    finalized = dict(bundle)
    del finalized["policy_sha256"]
    if canonical_json_bytes(recorder.training_manifest) != canonical_json_bytes(
        finalized
    ):
        raise ValueError("scheduled sidecar belongs to a different successful fit")
    raw = b"".join(event + b"\n" for event in recorder.events)
    manifest = _manifest(
        bundle,
        expected_bundle_digest,
        [json.loads(row) for row in recorder.events],
        raw,
        recorder.learner_diagnostics,
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


def read_scheduled_allocation_transition_trace(
    root: Path,
    *,
    expected_digest: str,
    bundle_root: Path,
    expected_bundle_digest: str,
    require_learner_diagnostics: bool = False,
) -> dict[str, Any]:
    """Verify canonical pins plus every chronological relation without a backend."""
    if type(require_learner_diagnostics) is not bool:
        raise ValueError("scheduled diagnostics requirement must be a native boolean")
    require_sha256(expected_digest, field="expected_digest")
    root = Path(root)
    _directory(root, {"manifest.json", "transitions.jsonl"})
    _, manifest = _canonical(root / "manifest.json")
    if content_digest(manifest) != expected_digest:
        raise ValueError("scheduled sidecar differs from external pin")
    mode = manifest.get("learner_diagnostics")
    if (
        mode not in ("none", "native_ppo_update_v1")
        or require_learner_diagnostics
        and mode != "native_ppo_update_v1"
    ):
        raise ValueError(
            "scheduled required learner diagnostics are absent or undeclared"
        )
    with open_regular_binary(
        root / "transitions.jsonl", field="scheduled transition rows"
    ) as stream:
        raw = stream.read()
    try:
        events = [json.loads(line) for line in raw.splitlines(keepends=True)]
        if b"".join(canonical_json_bytes(event) + b"\n" for event in events) != raw:
            raise ValueError("scheduled rows must preserve canonical JSONL")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("invalid scheduled row bytes") from error
    bundle = _bundle(Path(bundle_root), expected_bundle_digest)
    expected = _manifest(bundle, expected_bundle_digest, events, raw, mode)
    if canonical_json_bytes(expected) != canonical_json_bytes(manifest):
        raise ValueError("scheduled manifest differs from closed bundle/row links")
    return manifest


__all__ = [
    "publish_scheduled_allocation_transition_trace",
    "read_scheduled_allocation_transition_trace",
]
