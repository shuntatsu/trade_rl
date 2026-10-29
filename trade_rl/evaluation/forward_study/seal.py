"""Preconditions that must hold before FS1 may be scored: the seal and the refusals.

Nothing here computes a P&L. :func:`authorize_scoring` collects EVERY failed
precondition and raises :class:`Fs1Refusal` listing all of them (it never partially
authorises); only a fully satisfied call returns a :class:`ScoringPermit`, which the
scoring step requires. The permit is a software guard against accidental use, not a
tamper-proof credential: the real gate is the human authorization the reference
points to, which this module cannot verify (no network, no GitHub identity check).

Preconditions (reason codes):

* ``scoring_window_not_closed`` - the wall clock is before the scoring end;
* ``authorization_missing`` / ``authorization_malformed`` - no GitHub Issue comment URL;
* ``registry_digest_differs_from_frozen`` / ``params_digest_mismatch`` - the parameters
  are not the frozen ones, or differ from the digest in the seal record;
* ``code_identity_unavailable`` / ``code_sha_mismatch`` / ``code_tree_dirty`` /
  ``sealed_code_sha_missing`` - the checked-out code is not the sealed commit;
* ``manifest_missing`` / ``manifest_invalid`` / ``manifest_sha256_mismatch`` /
  ``manifest_window_mismatch`` - the dataset seal manifest is absent, malformed, not the
  one recorded in the seal record, or for another window/roster;
* ``output_exists`` - FS1 is scored once; the result file is write-once.

Dataset checks that need the loaded arrays live in :func:`dataset_problems`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forward_study.registry import (
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
    MANIFEST_SCHEMA,
    Fs1Registry,
)

AUTHORIZATION_PATTERN: Final = re.compile(
    r"^https://github\.com/shuntatsu/trade_rl/issues/[1-9][0-9]*#issuecomment-[1-9][0-9]*$"
)
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA: Final = re.compile(r"^[0-9a-f]{40}$")
_NS_PER_HOUR: Final = 3_600 * 1_000_000_000
_PERMIT_TOKEN: Final = object()


class Fs1Refusal(RuntimeError):
    """FS1 will not be scored; ``reasons`` lists every unmet precondition."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        super().__init__("FS1 scoring refused: " + ", ".join(reasons))
        self.reasons = reasons


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _ns(text: str) -> int:
    return int(np.datetime64(text, "ns").astype(np.int64))


@dataclass(frozen=True)
class DatasetSealManifest:
    """The record made BEFORE scoring that binds the authorised dataset."""

    schema_version: str
    dataset_id: str
    symbols: tuple[str, ...]
    first_timestamp: str
    last_timestamp: str
    n_bars: int
    build_policy_digest: str
    source_manifest: dict[str, str]
    authorization_reference: str
    sealed_utc: str


def parse_manifest(raw: bytes) -> DatasetSealManifest:
    """Strictly parse manifest bytes; ``ValueError`` on any deviation."""

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest is not UTF-8 JSON: {exc}") from exc
    expected = {
        "schema_version",
        "dataset_id",
        "symbols",
        "first_timestamp",
        "last_timestamp",
        "n_bars",
        "build_policy_digest",
        "source_manifest",
        "authorization_reference",
        "sealed_utc",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("manifest keys differ from the sealed schema")
    if data["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("manifest schema_version is not the FS1 seal schema")
    for name in ("dataset_id", "build_policy_digest"):
        if not isinstance(data[name], str) or not _SHA256.match(data[name]):
            raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    symbols = data["symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(value, str) and value for value in symbols)
    ):
        raise ValueError("symbols must be a non-empty list of names")
    n_bars = data["n_bars"]
    if type(n_bars) is not int or n_bars < 2:
        raise ValueError("n_bars must be an integer of at least two")
    for name in ("first_timestamp", "last_timestamp", "sealed_utc"):
        try:
            datetime.fromisoformat(data[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    sources = data["source_manifest"]
    if (
        not isinstance(sources, dict)
        or not sources
        or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and _SHA256.match(value)
            for key, value in sources.items()
        )
    ):
        raise ValueError("source_manifest must map source names to SHA-256 digests")
    reference = data["authorization_reference"]
    if not isinstance(reference, str) or not AUTHORIZATION_PATTERN.match(reference):
        raise ValueError("authorization_reference must be a GitHub Issue comment URL")
    return DatasetSealManifest(
        schema_version=data["schema_version"],
        dataset_id=data["dataset_id"],
        symbols=tuple(symbols),
        first_timestamp=data["first_timestamp"],
        last_timestamp=data["last_timestamp"],
        n_bars=n_bars,
        build_policy_digest=data["build_policy_digest"],
        source_manifest=dict(sources),
        authorization_reference=reference,
        sealed_utc=data["sealed_utc"],
    )


def manifest_bytes_from_dataset(
    dataset: MarketDataset,
    *,
    source_manifest: dict[str, str],
    build_policy_digest: str,
    authorization_reference: str,
    sealed_utc: str,
) -> bytes:
    """Canonical manifest bytes for a built dataset (the seal step, run before scoring).

    Reads only dataset metadata and records the caller's source digests; it computes no
    return and fetches nothing. The result is parsed back to validate it.
    """

    stamps = dataset.timestamps.astype("datetime64[s]")
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "symbols": list(dataset.symbols),
        "first_timestamp": str(stamps[0]),
        "last_timestamp": str(stamps[-1]),
        "n_bars": int(dataset.n_bars),
        "build_policy_digest": build_policy_digest,
        "source_manifest": dict(sorted(source_manifest.items())),
        "authorization_reference": authorization_reference,
        "sealed_utc": sealed_utc,
    }
    raw = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    parse_manifest(raw)
    return raw


@dataclass(frozen=True)
class CodeIdentity:
    """The checked-out code: HEAD commit and whether tracked files are unmodified."""

    git_head: str | None
    tracked_tree_clean: bool


def current_code_identity(repo_root: Path | None = None) -> CodeIdentity:
    """Read ``git`` for the checkout that contains this package (no network)."""

    root = repo_root or Path(__file__).resolve().parents[3]

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    if head is None or status is None:
        return CodeIdentity(None, False)
    return CodeIdentity(head.strip() or None, status.strip() == "")


@dataclass(frozen=True)
class ScoringPermit:
    """Proof that every precondition held; required by the scoring step."""

    registry: Fs1Registry
    manifest: DatasetSealManifest
    manifest_sha256: str
    authorization: str
    params_digest: str
    code_sha: str
    authorized_at_utc: str
    _token: object = field(default=None, repr=False, compare=False)


def require_permit(permit: object) -> ScoringPermit:
    if (
        not isinstance(permit, ScoringPermit)
        or getattr(permit, "_token", None) is not _PERMIT_TOKEN
    ):
        raise TypeError("scoring requires a ScoringPermit from authorize_scoring")
    return permit


def authorize_scoring(
    *,
    now: datetime,
    authorization: str | None,
    expected_params_digest: str | None,
    sealed_code_sha: str | None,
    code_identity: CodeIdentity,
    manifest_bytes: bytes | None,
    expected_manifest_sha256: str | None,
    registry: Fs1Registry = FS1_REGISTRY,
    frozen_params_digest: str = FROZEN_PARAMS_DIGEST,
) -> ScoringPermit:
    """Return a permit only if every precondition holds, else raise with all reasons."""

    reasons: list[str] = []
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")
    if now < _utc(registry.scoring_end):
        reasons.append("scoring_window_not_closed")
    if not authorization:
        reasons.append("authorization_missing")
    elif not AUTHORIZATION_PATTERN.match(authorization):
        reasons.append("authorization_malformed")

    computed = registry.digest()
    if computed != frozen_params_digest:
        reasons.append("registry_digest_differs_from_frozen")
    if expected_params_digest != computed:
        reasons.append("params_digest_mismatch")

    if not sealed_code_sha or not _GIT_SHA.match(sealed_code_sha):
        reasons.append("sealed_code_sha_missing")
    if code_identity.git_head is None:
        reasons.append("code_identity_unavailable")
    else:
        if sealed_code_sha and code_identity.git_head != sealed_code_sha:
            reasons.append("code_sha_mismatch")
        if not code_identity.tracked_tree_clean:
            reasons.append("code_tree_dirty")

    manifest: DatasetSealManifest | None = None
    manifest_sha = ""
    if manifest_bytes is None:
        reasons.append("manifest_missing")
    else:
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if expected_manifest_sha256 != manifest_sha:
            reasons.append("manifest_sha256_mismatch")
        try:
            manifest = parse_manifest(manifest_bytes)
        except ValueError:
            reasons.append("manifest_invalid")
    if manifest is not None and (
        manifest.symbols != registry.symbols
        or _ns(manifest.last_timestamp) != _ns(registry.scoring_end)
        or _ns(manifest.first_timestamp) > _ns(registry.dataset_first_row_latest)
    ):
        reasons.append("manifest_window_mismatch")

    if reasons:
        raise Fs1Refusal(tuple(dict.fromkeys(reasons)))
    assert manifest is not None and authorization and sealed_code_sha
    return ScoringPermit(
        registry=registry,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        authorization=authorization,
        params_digest=computed,
        code_sha=sealed_code_sha,
        authorized_at_utc=now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        _token=_PERMIT_TOKEN,
    )


def dataset_problems(permit: ScoringPermit, dataset: MarketDataset) -> tuple[str, ...]:
    """Every way the loaded dataset fails the seal (empty tuple when it matches)."""

    registry, manifest = permit.registry, permit.manifest
    problems: list[str] = []
    if not dataset.identity_verified:
        problems.append("dataset_identity_unverified")
    if dataset.dataset_id != manifest.dataset_id:
        problems.append("dataset_id_mismatch")
    if tuple(dataset.symbols) != registry.symbols:
        problems.append("dataset_symbols_mismatch")
    stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    if stamps.size != manifest.n_bars:
        problems.append("dataset_n_bars_mismatch")
    if stamps.size < 2 or not np.all(np.diff(stamps) == _NS_PER_HOUR):
        problems.append("dataset_not_regular_hourly")
        return tuple(problems)
    if int(stamps[0]) != _ns(manifest.first_timestamp) or int(stamps[0]) > _ns(
        registry.dataset_first_row_latest
    ):
        problems.append("dataset_first_row_mismatch")
    if int(stamps[-1]) != _ns(registry.scoring_end) or int(stamps[-1]) != _ns(
        manifest.last_timestamp
    ):
        problems.append("dataset_last_row_mismatch")
    tradable = np.asarray(dataset.tradable)
    window = (stamps >= _ns(registry.development_data_end)) & (
        stamps < _ns(registry.scoring_end)
    )
    if not window.any():
        problems.append("dataset_window_empty")
    else:
        fraction = tradable[window].mean(axis=0)
        for symbol, value in zip(dataset.symbols, fraction, strict=True):
            if float(value) < registry.min_tradable_fraction:
                problems.append(f"dataset_tradable_fraction_low_{symbol}")
    return tuple(problems)


def verify_dataset(permit: ScoringPermit, dataset: MarketDataset) -> None:
    """Raise :class:`Fs1Refusal` unless the dataset matches the sealed manifest."""

    problems = dataset_problems(require_permit(permit), dataset)
    if problems:
        raise Fs1Refusal(problems)


__all__ = [
    "AUTHORIZATION_PATTERN",
    "CodeIdentity",
    "DatasetSealManifest",
    "Fs1Refusal",
    "ScoringPermit",
    "authorize_scoring",
    "current_code_identity",
    "dataset_problems",
    "manifest_bytes_from_dataset",
    "parse_manifest",
    "require_permit",
    "verify_dataset",
]
