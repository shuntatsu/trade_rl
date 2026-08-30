"""Train-only fit/statistics provenance for Universal Trade RL U0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

UNIVERSAL_TRADE_RL_FIT_PROVENANCE_SCHEMA: Final = (
    "universal_trade_rl_fit_provenance_v1"
)


class UniversalTradeRLFitPurpose(str, Enum):
    """Operations whose learned/statistical state must be Train-only."""

    FEATURE_NORMALIZATION = "feature_normalization"
    FORECAST_FIT = "forecast_fit"
    CALIBRATION = "calibration"
    POPULATION_THRESHOLD_FIT = "population_threshold_fit"
    REWARD_COEFFICIENT_FIT = "reward_coefficient_fit"
    RL_TRAINING = "rl_training"


def _canonical_symbols(symbols: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(symbols, tuple) or not symbols:
        raise ValueError("fit provenance source symbols must be a non-empty tuple")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError("fit provenance source symbols are invalid")
    if len(set(symbols)) != len(symbols):
        raise ValueError("fit provenance source symbols must be unique")
    if symbols != tuple(sorted(symbols)):
        raise ValueError("fit provenance source symbols must be sorted")
    return symbols


def _canonical_source_digests(
    values: tuple[tuple[str, str], ...],
    *,
    source_symbols: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, tuple) or len(values) != len(source_symbols):
        raise ValueError("fit provenance source identity contract is invalid")
    result: list[tuple[str, str]] = []
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("fit provenance source identity contract is invalid")
        symbol, dataset_digest = item
        if not isinstance(symbol, str) or not isinstance(dataset_digest, str):
            raise ValueError("fit provenance source identity contract is invalid")
        require_sha256(dataset_digest, field=f"fit provenance source {symbol} digest")
        result.append((symbol, dataset_digest))
    resolved = tuple(result)
    if tuple(symbol for symbol, _digest in resolved) != source_symbols:
        raise ValueError("fit provenance source identity does not match source symbols")
    return resolved


def _knowledge_cutoff(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("fit provenance knowledge cutoff must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class UniversalTradeRLFitProvenance:
    """Immutable evidence that one fit/statistics operation used Train data only."""

    purpose: UniversalTradeRLFitPurpose
    universe_manifest_digest: str
    source_symbols: tuple[str, ...]
    source_dataset_digests: tuple[tuple[str, str], ...]
    knowledge_cutoff: int
    schema_version: str = UNIVERSAL_TRADE_RL_FIT_PROVENANCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, UniversalTradeRLFitPurpose):
            raise TypeError("fit provenance purpose is invalid")
        if self.schema_version != UNIVERSAL_TRADE_RL_FIT_PROVENANCE_SCHEMA:
            raise ValueError("unsupported Universal Trade RL fit provenance schema")
        require_sha256(
            self.universe_manifest_digest,
            field="fit provenance universe manifest digest",
        )
        symbols = _canonical_symbols(self.source_symbols)
        source_digests = _canonical_source_digests(
            self.source_dataset_digests,
            source_symbols=symbols,
        )
        cutoff = _knowledge_cutoff(self.knowledge_cutoff)
        object.__setattr__(self, "source_symbols", symbols)
        object.__setattr__(self, "source_dataset_digests", source_digests)
        object.__setattr__(self, "knowledge_cutoff", cutoff)

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("Universal Trade RL fit provenance digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "purpose": self.purpose.value,
            "universe_manifest_digest": self.universe_manifest_digest,
            "source_symbols": self.source_symbols,
            "source_dataset_digests": self.source_dataset_digests,
            "knowledge_cutoff": self.knowledge_cutoff,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_universal_trade_rl_fit_provenance(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    access: UniversalTradeRLUniverseAccess,
    purpose: UniversalTradeRLFitPurpose,
    source_symbols: tuple[str, ...],
    knowledge_cutoff: int,
) -> UniversalTradeRLFitProvenance:
    """Build provenance only after the access firewall accepts the fit scope."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("fit provenance manifest is invalid")
    if not isinstance(access, UniversalTradeRLUniverseAccess):
        raise TypeError("fit provenance access contract is invalid")
    if not isinstance(purpose, UniversalTradeRLFitPurpose):
        raise TypeError("fit provenance purpose is invalid")
    if access.universe_manifest_digest != manifest.digest:
        raise ValueError("fit provenance universe access generation mismatch")
    cutoff = _knowledge_cutoff(knowledge_cutoff)

    scope = access.require_fit_scope(source_symbols)
    source_digests: list[tuple[str, str]] = []
    for symbol in scope:
        try:
            entry = manifest.entry_for(symbol)
        except KeyError as error:
            raise ValueError(
                f"fit provenance source symbol is absent from manifest: {symbol}"
            ) from error
        if entry.role is not UniversalTradeRLSymbolRole.TRAIN:
            raise PermissionError("Universal Trade RL fit provenance source is Train-only")
        source_digests.append((symbol, entry.dataset_digest))

    return UniversalTradeRLFitProvenance(
        purpose=purpose,
        universe_manifest_digest=manifest.digest,
        source_symbols=scope,
        source_dataset_digests=tuple(source_digests),
        knowledge_cutoff=cutoff,
    )


def require_universal_trade_rl_train_only_provenance(
    provenance: UniversalTradeRLFitProvenance,
    *,
    manifest: UniversalTradeRLUniverseManifest,
) -> UniversalTradeRLFitProvenance:
    """Validate provenance against current manifest source identities, fail closed."""

    if not isinstance(provenance, UniversalTradeRLFitProvenance):
        raise TypeError("fit provenance contract is invalid")
    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("fit provenance validation manifest is invalid")
    if provenance.universe_manifest_digest != manifest.digest:
        raise ValueError("fit provenance universe manifest identity mismatch")

    expected: list[tuple[str, str]] = []
    for symbol in provenance.source_symbols:
        try:
            entry = manifest.entry_for(symbol)
        except KeyError as error:
            raise ValueError(
                f"fit provenance source symbol is absent from manifest: {symbol}"
            ) from error
        if entry.role is not UniversalTradeRLSymbolRole.TRAIN:
            raise ValueError("fit provenance source role is not Train-only")
        expected.append((symbol, entry.dataset_digest))
    if tuple(expected) != provenance.source_dataset_digests:
        raise ValueError("fit provenance source dataset identity mismatch")
    return provenance


__all__ = [
    "UNIVERSAL_TRADE_RL_FIT_PROVENANCE_SCHEMA",
    "UniversalTradeRLFitProvenance",
    "UniversalTradeRLFitPurpose",
    "build_universal_trade_rl_fit_provenance",
    "require_universal_trade_rl_train_only_provenance",
]
