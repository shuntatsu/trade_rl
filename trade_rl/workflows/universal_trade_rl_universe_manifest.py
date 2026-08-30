"""Artifact-bound complete universe manifest for Universal Trade RL U0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolExclusion,
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
    universal_trade_rl_source_catalog_digest,
)

UNIVERSAL_TRADE_RL_UNIVERSE_MANIFEST_SCHEMA: Final = (
    "universal_trade_rl_universe_manifest_v1"
)
_ENTRY_KEYS: Final = (
    "symbol",
    "role",
    "exclusion_reason",
    "dataset_digest",
    "first_timestamp_ns",
    "last_timestamp_ns",
    "row_count",
)
_MANIFEST_KEYS: Final = (
    "schema_version",
    "config_digest",
    "source_catalog_digest",
    "entries",
    "artifact_digest",
)


def _mapping(value: object, *, keys: tuple[str, ...], field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} contract must be an object")
    result = {str(key): item for key, item in value.items()}
    if set(result) != set(keys) or len(result) != len(keys):
        raise ValueError(f"{field} contract must use exact keys")
    return result


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} contract must be an integer and not boolean")
    return value


@dataclass(frozen=True, slots=True)
class UniversalTradeRLUniverseEntry:
    """One complete role/exclusion assignment bound to source-data identity."""

    symbol: str
    role: UniversalTradeRLSymbolRole | None
    exclusion_reason: str | None
    dataset_digest: str
    first_timestamp_ns: int
    last_timestamp_ns: int
    row_count: int

    def __post_init__(self) -> None:
        source = UniversalTradeRLSymbolSource(
            symbol=self.symbol,
            dataset_digest=self.dataset_digest,
            first_timestamp_ns=self.first_timestamp_ns,
            last_timestamp_ns=self.last_timestamp_ns,
            row_count=self.row_count,
        )
        if self.role is not None and not isinstance(
            self.role, UniversalTradeRLSymbolRole
        ):
            raise TypeError(
                "universe entry role must be a Universal Trade RL symbol role"
            )
        has_role = self.role is not None
        has_exclusion = self.exclusion_reason is not None
        if has_role == has_exclusion:
            raise ValueError("universe entry must have exactly one role or exclusion")
        if has_exclusion:
            if not isinstance(self.exclusion_reason, str):
                raise TypeError("universe entry exclusion reason must be a string")
            reason = require_non_empty(
                self.exclusion_reason,
                field="universe entry exclusion reason",
            )
            object.__setattr__(self, "exclusion_reason", reason)
        object.__setattr__(self, "symbol", source.symbol)
        object.__setattr__(self, "first_timestamp_ns", source.first_timestamp_ns)
        object.__setattr__(self, "last_timestamp_ns", source.last_timestamp_ns)
        object.__setattr__(self, "row_count", source.row_count)

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "role": None if self.role is None else self.role.value,
            "exclusion_reason": self.exclusion_reason,
            "dataset_digest": self.dataset_digest,
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
            "row_count": self.row_count,
        }

    @classmethod
    def from_payload(cls, payload: object) -> UniversalTradeRLUniverseEntry:
        values = _mapping(payload, keys=_ENTRY_KEYS, field="universe entry")
        role_value = values["role"]
        if role_value is None:
            role = None
        elif isinstance(role_value, str):
            try:
                role = UniversalTradeRLSymbolRole(role_value)
            except ValueError as error:
                raise ValueError("universe entry role contract is invalid") from error
        else:
            raise ValueError("universe entry role contract is invalid")
        reason = values["exclusion_reason"]
        if reason is not None and not isinstance(reason, str):
            raise ValueError("universe entry exclusion reason contract is invalid")
        symbol = values["symbol"]
        dataset_digest = values["dataset_digest"]
        if not isinstance(symbol, str) or not isinstance(dataset_digest, str):
            raise ValueError("universe entry source identity contract is invalid")
        return cls(
            symbol=symbol,
            role=role,
            exclusion_reason=reason,
            dataset_digest=dataset_digest,
            first_timestamp_ns=_integer(
                values["first_timestamp_ns"], field="first_timestamp_ns"
            ),
            last_timestamp_ns=_integer(
                values["last_timestamp_ns"], field="last_timestamp_ns"
            ),
            row_count=_integer(values["row_count"], field="row_count"),
        )

    def to_source(self) -> UniversalTradeRLSymbolSource:
        return UniversalTradeRLSymbolSource(
            symbol=self.symbol,
            dataset_digest=self.dataset_digest,
            first_timestamp_ns=self.first_timestamp_ns,
            last_timestamp_ns=self.last_timestamp_ns,
            row_count=self.row_count,
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLUniverseManifest:
    """Complete immutable role/exclusion partition bound to exact source identities."""

    config_digest: str
    source_catalog_digest: str
    entries: tuple[UniversalTradeRLUniverseEntry, ...]
    schema_version: str = UNIVERSAL_TRADE_RL_UNIVERSE_MANIFEST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_TRADE_RL_UNIVERSE_MANIFEST_SCHEMA:
            raise ValueError("unsupported Universal Trade RL universe manifest schema")
        require_sha256(self.config_digest, field="universe manifest config digest")
        require_sha256(
            self.source_catalog_digest,
            field="universe manifest source catalog digest",
        )
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("universe manifest entries must be a non-empty tuple")
        if any(
            not isinstance(item, UniversalTradeRLUniverseEntry) for item in self.entries
        ):
            raise TypeError("universe manifest entries contain invalid contracts")
        symbols = tuple(item.symbol for item in self.entries)
        if len(set(symbols)) != len(symbols):
            raise ValueError("universe manifest entries must contain unique symbols")
        if symbols != tuple(sorted(symbols)):
            raise ValueError("universe manifest entries must be sorted")

        reconstructed_config = _config_from_entries(self.entries)
        if reconstructed_config.digest != self.config_digest:
            raise ValueError("universe manifest config digest contract mismatch")
        reconstructed_sources = tuple(item.to_source() for item in self.entries)
        expected_source_digest = universal_trade_rl_source_catalog_digest(
            reconstructed_sources
        )
        if expected_source_digest != self.source_catalog_digest:
            raise ValueError(
                "universe manifest source catalog digest contract mismatch"
            )

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("Universal Trade RL universe manifest digest mismatch")
        object.__setattr__(self, "digest", expected)

    def entry_for(self, symbol: str) -> UniversalTradeRLUniverseEntry:
        if not isinstance(symbol, str):
            raise TypeError("universe manifest symbol lookup must be a string")
        for entry in self.entries:
            if entry.symbol == symbol:
                return entry
        raise KeyError(symbol)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "config_digest": self.config_digest,
            "source_catalog_digest": self.source_catalog_digest,
            "entries": tuple(item.to_payload() for item in self.entries),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> UniversalTradeRLUniverseManifest:
        values = _mapping(payload, keys=_MANIFEST_KEYS, field="universe manifest")
        if values["schema_version"] != UNIVERSAL_TRADE_RL_UNIVERSE_MANIFEST_SCHEMA:
            raise ValueError("unsupported Universal Trade RL universe manifest schema")
        entries_raw = values["entries"]
        if not isinstance(entries_raw, Sequence) or isinstance(
            entries_raw, (str, bytes)
        ):
            raise ValueError("universe manifest entries contract must be an array")
        config_digest = values["config_digest"]
        source_catalog_digest = values["source_catalog_digest"]
        artifact_digest = values["artifact_digest"]
        if not all(
            isinstance(value, str)
            for value in (config_digest, source_catalog_digest, artifact_digest)
        ):
            raise ValueError("universe manifest digest contract is invalid")
        return cls(
            config_digest=config_digest,
            source_catalog_digest=source_catalog_digest,
            entries=tuple(
                UniversalTradeRLUniverseEntry.from_payload(item) for item in entries_raw
            ),
            schema_version=UNIVERSAL_TRADE_RL_UNIVERSE_MANIFEST_SCHEMA,
            digest=artifact_digest,
        )


def _config_from_entries(
    entries: tuple[UniversalTradeRLUniverseEntry, ...],
) -> UniversalTradeRLUniverseConfig:
    groups: dict[UniversalTradeRLSymbolRole, list[str]] = {
        role: [] for role in UniversalTradeRLSymbolRole
    }
    exclusions: list[UniversalTradeRLSymbolExclusion] = []
    for entry in entries:
        if entry.role is None:
            if entry.exclusion_reason is None:
                raise ValueError("universe manifest entry contract is incomplete")
            exclusions.append(
                UniversalTradeRLSymbolExclusion(
                    symbol=entry.symbol,
                    reason=entry.exclusion_reason,
                )
            )
        else:
            groups[entry.role].append(entry.symbol)
    try:
        return UniversalTradeRLUniverseConfig(
            train_symbols=tuple(groups[UniversalTradeRLSymbolRole.TRAIN]),
            development_symbols=tuple(groups[UniversalTradeRLSymbolRole.DEVELOPMENT]),
            admission_symbols=tuple(groups[UniversalTradeRLSymbolRole.ADMISSION]),
            exclusions=tuple(exclusions),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "universe manifest role/exclusion contract is invalid"
        ) from error


def build_universal_trade_rl_universe_manifest(
    *,
    config: UniversalTradeRLUniverseConfig,
    sources: tuple[UniversalTradeRLSymbolSource, ...],
) -> UniversalTradeRLUniverseManifest:
    """Bind the complete available source catalog to one immutable U0 partition."""

    if not isinstance(config, UniversalTradeRLUniverseConfig):
        raise TypeError("universe manifest config is invalid")
    source_catalog_digest = universal_trade_rl_source_catalog_digest(sources)
    source_symbols = {item.symbol for item in sources}
    configured_symbols = set(config.train_symbols)
    configured_symbols.update(config.development_symbols)
    configured_symbols.update(config.admission_symbols)
    configured_symbols.update(item.symbol for item in config.exclusions)

    missing = tuple(sorted(configured_symbols - source_symbols))
    if missing:
        raise ValueError(f"missing configured symbol: {missing[0]}")
    unassigned = tuple(sorted(source_symbols - configured_symbols))
    if unassigned:
        raise ValueError(f"unassigned available symbol: {unassigned[0]}")

    exclusion_reasons = {item.symbol: item.reason for item in config.exclusions}
    entries: list[UniversalTradeRLUniverseEntry] = []
    for source in sources:
        role = config.role_for(source.symbol)
        reason = exclusion_reasons.get(source.symbol)
        entries.append(
            UniversalTradeRLUniverseEntry(
                symbol=source.symbol,
                role=role,
                exclusion_reason=reason,
                dataset_digest=source.dataset_digest,
                first_timestamp_ns=source.first_timestamp_ns,
                last_timestamp_ns=source.last_timestamp_ns,
                row_count=source.row_count,
            )
        )
    return UniversalTradeRLUniverseManifest(
        config_digest=config.digest,
        source_catalog_digest=source_catalog_digest,
        entries=tuple(entries),
    )


__all__ = [
    "UNIVERSAL_TRADE_RL_UNIVERSE_MANIFEST_SCHEMA",
    "UniversalTradeRLUniverseEntry",
    "UniversalTradeRLUniverseManifest",
    "build_universal_trade_rl_universe_manifest",
]
