"""Immutable filesystem artifacts for Causal Alpha V4 target context."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.v4_context import V4ContextBlock, V4TargetContext
from trade_rl.domain.common import require_sha256

V4_CONTEXT_ARTIFACT_SCHEMA: Final = "causal_alpha_v4_target_context_artifact_v1"
V4_CONTEXT_MANIFEST_NAME: Final = "manifest.json"
V4_CONTEXT_ARRAYS_NAME: Final = "arrays.npz"

_ARRAY_NAMES: Final = frozenset(
    {
        "decision_indices",
        "local_values",
        "local_available",
        "local_staleness_hours",
        "global_values",
        "global_available",
        "global_staleness_hours",
        "beta",
        "beta_available",
    }
)
_MANIFEST_FIELDS: Final = frozenset(
    {
        "artifact_digest",
        "arrays_sha256",
        "beta_source_digest",
        "context_digest",
        "first_decision_index",
        "global_context_digest",
        "global_feature_names",
        "global_source_digest",
        "last_decision_index",
        "local_context_digest",
        "local_feature_names",
        "local_source_digest",
        "profile_name",
        "row_count",
        "schema_version",
        "symbol",
    }
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_payload(
    context: V4TargetContext,
    *,
    arrays_sha256: str,
) -> dict[str, object]:
    require_sha256(arrays_sha256, field="V4 context arrays_sha256")
    indices = context.local.decision_indices
    if indices.size == 0:
        raise ValueError("V4 context artifact cannot contain zero rows")
    return {
        "arrays_sha256": arrays_sha256,
        "beta_source_digest": context.beta_source_digest,
        "context_digest": context.digest,
        "first_decision_index": int(indices[0]),
        "global_context_digest": context.global_market.digest,
        "global_feature_names": list(context.global_market.feature_names),
        "global_source_digest": context.global_market.source_digest,
        "last_decision_index": int(indices[-1]),
        "local_context_digest": context.local.digest,
        "local_feature_names": list(context.local.feature_names),
        "local_source_digest": context.local.source_digest,
        "profile_name": context.profile_name,
        "row_count": int(indices.size),
        "schema_version": V4_CONTEXT_ARTIFACT_SCHEMA,
        "symbol": context.symbol,
    }


def _artifact_manifest(
    context: V4TargetContext,
    *,
    arrays_sha256: str,
) -> dict[str, object]:
    payload = _manifest_payload(context, arrays_sha256=arrays_sha256)
    return {**payload, "artifact_digest": content_digest(payload)}


def _write_arrays(path: Path, context: V4TargetContext) -> str:
    np.savez(
        path,
        decision_indices=context.local.decision_indices,
        local_values=context.local.values,
        local_available=context.local.available,
        local_staleness_hours=context.local.staleness_hours,
        global_values=context.global_market.values,
        global_available=context.global_market.available,
        global_staleness_hours=context.global_market.staleness_hours,
        beta=context.beta,
        beta_available=context.beta_available,
    )
    return _sha256_bytes(path.read_bytes())


def _strict_manifest(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("V4 context artifact manifest must be an object")
    payload = dict(raw)
    if set(payload) != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - set(payload))
        unknown = sorted(set(payload) - _MANIFEST_FIELDS)
        raise ValueError(
            "V4 context artifact manifest fields mismatch; "
            f"missing={missing}, unknown={unknown}"
        )
    if payload.get("schema_version") != V4_CONTEXT_ARTIFACT_SCHEMA:
        raise ValueError("V4 context artifact schema mismatch")
    artifact_digest = payload.get("artifact_digest")
    if not isinstance(artifact_digest, str):
        raise ValueError("V4 context artifact digest is missing")
    require_sha256(artifact_digest, field="V4 context artifact digest")
    identity = {
        key: value for key, value in payload.items() if key != "artifact_digest"
    }
    if content_digest(identity) != artifact_digest:
        raise ValueError("V4 context artifact digest mismatch")
    return payload


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an ordered array")
    result = tuple(value)
    if (
        not result
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return result


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _load_arrays(path: Path, *, expected_sha256: str) -> dict[str, np.ndarray]:
    require_sha256(expected_sha256, field="V4 context arrays_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"V4 context arrays are missing: {path}")
    if _sha256_bytes(path.read_bytes()) != expected_sha256:
        raise ValueError("V4 context arrays digest mismatch")
    try:
        with np.load(path, allow_pickle=False) as payload:
            members = frozenset(payload.files)
            if members != _ARRAY_NAMES:
                missing = sorted(_ARRAY_NAMES - members)
                unknown = sorted(members - _ARRAY_NAMES)
                raise ValueError(
                    "V4 context array members mismatch; "
                    f"missing={missing}, unknown={unknown}"
                )
            return {
                name: np.asarray(payload[name]).copy(order="C")
                for name in payload.files
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("V4 context"):
            raise
        raise ValueError("V4 context arrays cannot be decoded") from error


def _reconstruct_context(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> V4TargetContext:
    row_count = _non_negative_int(manifest["row_count"], field="V4 context row_count")
    if row_count <= 0:
        raise ValueError("V4 context row_count must be positive")
    first = _non_negative_int(
        manifest["first_decision_index"], field="V4 context first_decision_index"
    )
    last = _non_negative_int(
        manifest["last_decision_index"], field="V4 context last_decision_index"
    )
    decisions = np.asarray(arrays["decision_indices"], dtype=np.int64).reshape(-1)
    if (
        decisions.shape != (row_count,)
        or int(decisions[0]) != first
        or int(decisions[-1]) != last
    ):
        raise ValueError("V4 context array row identity mismatch")

    local_source = str(manifest["local_source_digest"])
    global_source = str(manifest["global_source_digest"])
    beta_source = str(manifest["beta_source_digest"])
    for field, value in (
        ("local_source_digest", local_source),
        ("global_source_digest", global_source),
        ("beta_source_digest", beta_source),
    ):
        require_sha256(value, field=f"V4 context {field}")

    local = V4ContextBlock(
        feature_names=_string_list(
            manifest["local_feature_names"], field="local_feature_names"
        ),
        decision_indices=decisions,
        values=arrays["local_values"],
        available=arrays["local_available"],
        staleness_hours=arrays["local_staleness_hours"],
        source_digest=local_source,
    )
    global_market = V4ContextBlock(
        feature_names=_string_list(
            manifest["global_feature_names"], field="global_feature_names"
        ),
        decision_indices=decisions,
        values=arrays["global_values"],
        available=arrays["global_available"],
        staleness_hours=arrays["global_staleness_hours"],
        source_digest=global_source,
    )

    expected_local = str(manifest["local_context_digest"])
    expected_global = str(manifest["global_context_digest"])
    require_sha256(expected_local, field="V4 context local_context_digest")
    require_sha256(expected_global, field="V4 context global_context_digest")
    if local.digest != expected_local or global_market.digest != expected_global:
        raise ValueError("V4 context block identity mismatch")

    context = V4TargetContext(
        symbol=str(manifest["symbol"]),
        local=local,
        global_market=global_market,
        beta=arrays["beta"],
        beta_available=arrays["beta_available"],
        beta_source_digest=beta_source,
        profile_name=str(manifest["profile_name"]),
    )
    expected_context = str(manifest["context_digest"])
    require_sha256(expected_context, field="V4 context context_digest")
    if context.digest != expected_context:
        raise ValueError("V4 target context identity mismatch")
    return context


def load_v4_target_context_artifact(root: str | Path) -> V4TargetContext:
    """Load and strictly validate one immutable V4 target-context artifact."""

    artifact_root = Path(root)
    manifest_path = artifact_root / V4_CONTEXT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V4 context manifest is missing: {manifest_path}")
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("V4 context artifact manifest is invalid JSON") from error
    manifest = _strict_manifest(raw_manifest)
    arrays_sha256 = manifest["arrays_sha256"]
    if not isinstance(arrays_sha256, str):
        raise ValueError("V4 context arrays_sha256 must be a string")
    arrays = _load_arrays(
        artifact_root / V4_CONTEXT_ARRAYS_NAME,
        expected_sha256=arrays_sha256,
    )
    return _reconstruct_context(manifest, arrays)


def write_v4_target_context_artifact(
    root: str | Path,
    context: V4TargetContext,
) -> Path:
    """Atomically publish one V4 context, allowing only identical reuse."""

    if not isinstance(context, V4TargetContext):
        raise TypeError("context must be V4TargetContext")
    output = Path(root)
    if output.exists():
        existing = load_v4_target_context_artifact(output)
        if existing.digest == context.digest:
            return output
        raise FileExistsError(
            f"V4 context destination already exists with different content: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        arrays_path = staging / V4_CONTEXT_ARRAYS_NAME
        arrays_sha256 = _write_arrays(arrays_path, context)
        manifest = _artifact_manifest(context, arrays_sha256=arrays_sha256)
        (staging / V4_CONTEXT_MANIFEST_NAME).write_bytes(
            canonical_json_bytes(manifest) + b"\n"
        )
        staged = load_v4_target_context_artifact(staging)
        if staged.digest != context.digest:
            raise ValueError("staged V4 context changed identity")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


__all__ = [
    "V4_CONTEXT_ARRAYS_NAME",
    "V4_CONTEXT_ARTIFACT_SCHEMA",
    "V4_CONTEXT_MANIFEST_NAME",
    "load_v4_target_context_artifact",
    "write_v4_target_context_artifact",
]
