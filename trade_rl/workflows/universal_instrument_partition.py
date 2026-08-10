"""Catalog-bound symbol partitions for mandatory zero-shot evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.stored_instrument_catalog import StoredInstrumentCatalog
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.symbol_disjoint_manifest import (
    SymbolDisjointManifest,
    SymbolSplit,
    build_symbol_disjoint_manifest,
)

UNIVERSAL_INSTRUMENT_PARTITION_SCHEMA: Final = "universal_instrument_partition_v1"
UniversalInstrumentSplit = Literal["train", "validation", "test"]
_SPLITS: Final[tuple[UniversalInstrumentSplit, ...]] = (
    "train",
    "validation",
    "test",
)


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _string_tuple(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a string list or tuple")
    resolved = tuple(value)
    if (not resolved and not allow_empty) or any(
        not isinstance(item, str) or not item for item in resolved
    ):
        raise ValueError(f"{field} contains invalid symbols")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must contain unique symbols")
    return resolved


def _immutable_write(path: Path, payload: bytes, *, field: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(
                f"{field} already exists with different content: {path}"
            )
        return path
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def universal_split_counts(symbol_count: int) -> tuple[int, int, int]:
    """Return train/validation/test counts that preserve zero-shot evidence."""

    if isinstance(symbol_count, bool) or not isinstance(symbol_count, int):
        raise ValueError("universal symbol count must be an integer")
    if symbol_count < 15:
        raise ValueError("universal zero-shot research requires at least 15 symbols")
    validation_count = max(3, symbol_count // 5)
    test_count = max(3, symbol_count // 5)
    train_count = symbol_count - validation_count - test_count
    if train_count < 9:
        raise ValueError(
            "universal zero-shot research requires at least 9 train symbols"
        )
    return train_count, validation_count, test_count


@dataclass(frozen=True, slots=True)
class UniversalInstrumentPartition:
    """One immutable symbol-disjoint split bound to a stored catalog."""

    catalog_digest: str
    symbol_disjoint_manifest: SymbolDisjointManifest
    schema_version: str = UNIVERSAL_INSTRUMENT_PARTITION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_INSTRUMENT_PARTITION_SCHEMA:
            raise ValueError("unsupported universal instrument partition schema")
        require_sha256(
            self.catalog_digest,
            field="universal instrument partition catalog_digest",
        )
        if not isinstance(self.symbol_disjoint_manifest, SymbolDisjointManifest):
            raise TypeError("symbol_disjoint_manifest must be SymbolDisjointManifest")
        train_count, validation_count, test_count = universal_split_counts(
            len(self.symbol_disjoint_manifest.source_universe)
        )
        expected_counts = {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        }
        if self.symbol_disjoint_manifest.split_counts != expected_counts:
            raise ValueError("universal instrument partition split counts mismatch")
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("universal instrument partition digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def symbol_disjoint_manifest_digest(self) -> str:
        return self.symbol_disjoint_manifest.digest

    @property
    def seed(self) -> int:
        return self.symbol_disjoint_manifest.seed

    @property
    def train_symbols(self) -> tuple[str, ...]:
        return self.symbol_disjoint_manifest.train_symbols

    @property
    def validation_symbols(self) -> tuple[str, ...]:
        return self.symbol_disjoint_manifest.validation_symbols

    @property
    def test_symbols(self) -> tuple[str, ...]:
        return self.symbol_disjoint_manifest.test_symbols

    @property
    def split_counts(self) -> dict[str, int]:
        return self.symbol_disjoint_manifest.split_counts

    def symbols_for(
        self,
        split: UniversalInstrumentSplit | str,
    ) -> tuple[str, ...]:
        if split not in _SPLITS:
            raise ValueError("universal instrument split is invalid")
        return self.symbol_disjoint_manifest.symbols_for(cast(SymbolSplit, split))

    def require_symbol(
        self,
        symbol: str,
        *,
        split: UniversalInstrumentSplit | str,
    ) -> str:
        resolved = _non_empty_string(symbol, field="universal instrument symbol")
        if resolved not in self.symbols_for(split):
            raise ValueError(f"symbol {resolved} is not declared for {split}")
        return resolved

    def require_symbols(
        self,
        symbols: tuple[str, ...] | list[str],
        *,
        split: UniversalInstrumentSplit | str,
    ) -> tuple[str, ...]:
        resolved = _string_tuple(
            symbols,
            field="universal instrument symbols",
        )
        return tuple(self.require_symbol(symbol, split=split) for symbol in resolved)

    def digest_payload(self) -> dict[str, object]:
        return {
            "catalog_digest": self.catalog_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "split_counts": self.split_counts,
            "symbol_disjoint_manifest_digest": (self.symbol_disjoint_manifest_digest),
            "test_symbols": self.test_symbols,
            "train_symbols": self.train_symbols,
            "validation_symbols": self.validation_symbols,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {
            "catalog_digest": self.catalog_digest,
            "digest": self.digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "split_counts": self.split_counts,
            "symbol_disjoint_manifest_digest": (self.symbol_disjoint_manifest_digest),
            "test_symbols": list(self.test_symbols),
            "train_symbols": list(self.train_symbols),
            "validation_symbols": list(self.validation_symbols),
        }


def build_universal_instrument_partition(
    catalog: StoredInstrumentCatalog,
    *,
    seed: int,
) -> UniversalInstrumentPartition:
    """Derive one deterministic catalog-bound zero-shot partition."""

    if not isinstance(catalog, StoredInstrumentCatalog):
        raise TypeError("catalog must be StoredInstrumentCatalog")
    resolved_seed = _non_negative_integer(seed, field="universal partition seed")
    _, validation_count, test_count = universal_split_counts(
        len(catalog.eligible_symbols)
    )
    manifest = build_symbol_disjoint_manifest(
        catalog.eligible_symbols,
        seed=resolved_seed,
        validation_count=validation_count,
        test_count=test_count,
        minimum_symbols_per_split=3,
    )
    return UniversalInstrumentPartition(
        catalog_digest=catalog.digest,
        symbol_disjoint_manifest=manifest,
    )


def write_universal_instrument_partition(
    path: str | Path,
    partition: UniversalInstrumentPartition,
) -> Path:
    """Write one canonical partition or require exact immutable reuse."""

    if not isinstance(partition, UniversalInstrumentPartition):
        raise TypeError("partition must be UniversalInstrumentPartition")
    return _immutable_write(
        Path(path),
        canonical_json_bytes(partition.to_json_dict()),
        field="universal instrument partition",
    )


def _json_object(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("universal instrument partition must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("universal instrument partition must be a JSON object")
    return dict(payload)


def _split_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(_SPLITS):
        raise ValueError("universal instrument split_counts closure mismatch")
    result: dict[str, int] = {}
    for split in _SPLITS:
        count = value[split]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("universal instrument split counts must be non-negative")
        result[split] = count
    return result


def load_universal_instrument_partition(
    path: str | Path,
    *,
    catalog: StoredInstrumentCatalog,
    symbol_disjoint_manifest: SymbolDisjointManifest,
) -> UniversalInstrumentPartition:
    """Load a strict partition against its supplied immutable dependencies."""

    if not isinstance(catalog, StoredInstrumentCatalog):
        raise TypeError("catalog must be StoredInstrumentCatalog")
    if not isinstance(symbol_disjoint_manifest, SymbolDisjointManifest):
        raise TypeError("symbol_disjoint_manifest must be SymbolDisjointManifest")
    payload = _json_object(path)
    required = {
        "catalog_digest",
        "digest",
        "schema_version",
        "seed",
        "split_counts",
        "symbol_disjoint_manifest_digest",
        "test_symbols",
        "train_symbols",
        "validation_symbols",
    }
    if set(payload) != required:
        raise ValueError("universal instrument partition field closure mismatch")
    serialized_catalog_digest = _non_empty_string(
        payload["catalog_digest"],
        field="universal instrument catalog_digest",
    )
    if serialized_catalog_digest != catalog.digest:
        raise ValueError("universal instrument partition catalog digest mismatch")
    serialized_manifest_digest = _non_empty_string(
        payload["symbol_disjoint_manifest_digest"],
        field="universal instrument symbol-disjoint digest",
    )
    if serialized_manifest_digest != symbol_disjoint_manifest.digest:
        raise ValueError(
            "universal instrument partition symbol-disjoint digest mismatch"
        )
    if set(symbol_disjoint_manifest.source_universe) != set(catalog.eligible_symbols):
        raise ValueError(
            "universal instrument partition catalog symbol closure mismatch"
        )

    partition = UniversalInstrumentPartition(
        catalog_digest=serialized_catalog_digest,
        symbol_disjoint_manifest=symbol_disjoint_manifest,
        schema_version=_non_empty_string(
            payload["schema_version"],
            field="universal instrument schema_version",
        ),
        digest=_non_empty_string(
            payload["digest"],
            field="universal instrument digest",
        ),
    )
    observed = {
        "seed": _non_negative_integer(
            payload["seed"],
            field="universal instrument seed",
        ),
        "split_counts": _split_counts(payload["split_counts"]),
        "test_symbols": _string_tuple(
            payload["test_symbols"],
            field="universal test symbols",
        ),
        "train_symbols": _string_tuple(
            payload["train_symbols"],
            field="universal train symbols",
        ),
        "validation_symbols": _string_tuple(
            payload["validation_symbols"],
            field="universal validation symbols",
        ),
    }
    expected = {
        "seed": partition.seed,
        "split_counts": partition.split_counts,
        "test_symbols": partition.test_symbols,
        "train_symbols": partition.train_symbols,
        "validation_symbols": partition.validation_symbols,
    }
    if observed != expected:
        raise ValueError("universal instrument partition serialized closure mismatch")
    return partition


__all__ = [
    "UNIVERSAL_INSTRUMENT_PARTITION_SCHEMA",
    "UniversalInstrumentPartition",
    "UniversalInstrumentSplit",
    "build_universal_instrument_partition",
    "load_universal_instrument_partition",
    "universal_split_counts",
    "write_universal_instrument_partition",
]
