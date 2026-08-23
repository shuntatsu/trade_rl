"""Immutable auxiliary-context manifest for the research-only Causal Alpha V4 lane."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V4_CONTEXT_MANIFEST_SCHEMA: Final = "causal_alpha_v4_context_manifest_v1"
_ALLOWED_PROFILES: Final = frozenset(
    {"cross_market_core_v1", "cross_market_derivatives_v1"}
)
_ALLOWED_PIT_FLOW_PROFILES: Final = frozenset({"pit_flow_v1"})


def _portable_relative_path(value: str | Path, *, field: str) -> Path:
    text = str(value).replace("\\", "/").strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a portable relative path")
    return Path(*path.parts)


def _context_digests(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("V4 context_digests must be an ordered array")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("V4 context_digests entries must be symbol/digest pairs")
        symbol, digest = item
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("V4 context symbols must be non-empty strings")
        if not isinstance(digest, str):
            raise ValueError("V4 context digest must be a string")
        require_sha256(digest, field=f"V4 context digest {symbol}")
        result.append((symbol.strip(), digest))
    resolved = tuple(result)
    if not resolved:
        raise ValueError("V4 context_digests must not be empty")
    symbols = tuple(symbol for symbol, _ in resolved)
    if len(set(symbols)) != len(symbols):
        raise ValueError("V4 context symbols must be unique")
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV4ContextManifest:
    """Bind one V4 auxiliary-context generation to one immutable base runtime."""

    base_runtime_manifest_digest: str
    profile_name: str
    context_artifact_relpath: Path
    context_digests: tuple[tuple[str, str], ...]
    local_schema_digest: str
    global_schema_digest: str
    pit_flow_profile: str | None
    source_capability_digest: str
    schema_version: str = CAUSAL_ALPHA_V4_CONTEXT_MANIFEST_SCHEMA
    manifest_digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "base_runtime_manifest_digest",
            "local_schema_digest",
            "global_schema_digest",
            "source_capability_digest",
        ):
            require_sha256(getattr(self, name), field=f"V4 context manifest {name}")
        if self.profile_name not in _ALLOWED_PROFILES:
            raise ValueError("V4 context profile_name is unsupported")
        relpath = _portable_relative_path(
            self.context_artifact_relpath,
            field="V4 context artifact relpath",
        )
        contexts = _context_digests(self.context_digests)
        pit_flow = self.pit_flow_profile
        if pit_flow is not None and pit_flow not in _ALLOWED_PIT_FLOW_PROFILES:
            raise ValueError("V4 context pit_flow_profile is unsupported")
        if self.schema_version != CAUSAL_ALPHA_V4_CONTEXT_MANIFEST_SCHEMA:
            raise ValueError("V4 context manifest schema mismatch")
        object.__setattr__(self, "context_artifact_relpath", relpath)
        object.__setattr__(self, "context_digests", contexts)
        expected = content_digest(self.digest_payload())
        if self.manifest_digest and self.manifest_digest != expected:
            raise ValueError("V4 context manifest digest mismatch")
        object.__setattr__(self, "manifest_digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "base_runtime_manifest_digest": self.base_runtime_manifest_digest,
            "context_artifact_relpath": self.context_artifact_relpath.as_posix(),
            "context_digests": [list(item) for item in self.context_digests],
            "global_schema_digest": self.global_schema_digest,
            "local_schema_digest": self.local_schema_digest,
            "pit_flow_profile": self.pit_flow_profile,
            "profile_name": self.profile_name,
            "schema_version": self.schema_version,
            "source_capability_digest": self.source_capability_digest,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_json_dict(cls, value: object) -> "CausalAlphaV4ContextManifest":
        if not isinstance(value, dict):
            raise ValueError("V4 context manifest must be an object")
        expected = {item.name for item in fields(cls)}
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError(
                "V4 context manifest has unknown or missing fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        raw_pit = value["pit_flow_profile"]
        if raw_pit is not None and not isinstance(raw_pit, str):
            raise ValueError("V4 context pit_flow_profile must be null or a string")
        try:
            return cls(
                base_runtime_manifest_digest=str(value["base_runtime_manifest_digest"]),
                profile_name=str(value["profile_name"]),
                context_artifact_relpath=Path(str(value["context_artifact_relpath"])),
                context_digests=_context_digests(value["context_digests"]),
                local_schema_digest=str(value["local_schema_digest"]),
                global_schema_digest=str(value["global_schema_digest"]),
                pit_flow_profile=raw_pit,
                source_capability_digest=str(value["source_capability_digest"]),
                schema_version=str(value["schema_version"]),
                manifest_digest=str(value["manifest_digest"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError("V4 context manifest field types are invalid") from error


def validate_causal_alpha_v4_context_manifest_against_base(
    manifest: CausalAlphaV4ContextManifest,
    base_runtime: Any,
) -> None:
    """Require exact base-runtime identity and train/validation/test symbol order."""

    if not isinstance(manifest, CausalAlphaV4ContextManifest):
        raise TypeError("manifest must be CausalAlphaV4ContextManifest")
    base_digest = getattr(base_runtime, "manifest_digest", None)
    if base_digest != manifest.base_runtime_manifest_digest:
        raise ValueError("V4 context manifest base runtime digest mismatch")
    train = tuple(getattr(base_runtime, "train_symbols", ()))
    validation = tuple(getattr(base_runtime, "validation_symbols", ()))
    test = tuple(getattr(base_runtime, "test_symbols", ()))
    expected_symbols = (*train, *validation, *test)
    observed_symbols = tuple(symbol for symbol, _ in manifest.context_digests)
    if observed_symbols != expected_symbols:
        raise ValueError("V4 context manifest symbol order does not match base runtime")


def load_causal_alpha_v4_context_manifest(
    path: str | Path,
) -> CausalAlphaV4ContextManifest:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("V4 context manifest JSON is invalid") from error
    return CausalAlphaV4ContextManifest.from_json_dict(payload)


def write_causal_alpha_v4_context_manifest(
    path: str | Path,
    manifest: CausalAlphaV4ContextManifest,
) -> Path:
    """Write one immutable manifest, allowing byte-identical idempotent reuse."""

    if not isinstance(manifest, CausalAlphaV4ContextManifest):
        raise TypeError("manifest must be CausalAlphaV4ContextManifest")
    output = Path(path)
    payload = canonical_json_bytes(manifest.to_json_dict()) + b"\n"
    if output.exists():
        if output.is_file() and output.read_bytes() == payload:
            return output
        raise FileExistsError(
            f"V4 context manifest already exists with different content: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return output


__all__ = [
    "CAUSAL_ALPHA_V4_CONTEXT_MANIFEST_SCHEMA",
    "CausalAlphaV4ContextManifest",
    "load_causal_alpha_v4_context_manifest",
    "validate_causal_alpha_v4_context_manifest_against_base",
    "write_causal_alpha_v4_context_manifest",
]
