"""Deterministic symbol-disjoint train/validation/test evaluation manifests."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_unique_non_empty

SYMBOL_DISJOINT_MANIFEST_SCHEMA: Final = "symbol_disjoint_manifest_v1"
SymbolSplit = Literal["train", "validation", "test"]
_SPLITS: Final[tuple[SymbolSplit, ...]] = ("train", "validation", "test")


def _validate_non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validate_positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _canonical_symbols(symbols: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    return tuple(sorted(require_unique_non_empty(tuple(symbols), field=field)))


def _rank(seed: int, symbol: str) -> tuple[str, str]:
    return (
        content_digest(
            {
                "schema_version": "symbol_disjoint_rank_v1",
                "seed": seed,
                "symbol": symbol,
            }
        ),
        symbol,
    )


@dataclass(frozen=True, slots=True)
class SymbolDisjointManifest:
    """A content-addressed partition whose symbol sets cannot overlap."""

    source_universe: tuple[str, ...]
    universe_digest: str
    seed: int
    train_symbols: tuple[str, ...]
    validation_symbols: tuple[str, ...]
    test_symbols: tuple[str, ...]
    minimum_symbols_per_split: int = 3
    schema_version: str = SYMBOL_DISJOINT_MANIFEST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_DISJOINT_MANIFEST_SCHEMA:
            raise ValueError("unsupported symbol-disjoint manifest schema")
        seed = _validate_non_negative_int(self.seed, field="symbol_disjoint.seed")
        minimum = _validate_positive_int(
            self.minimum_symbols_per_split,
            field="symbol_disjoint.minimum_symbols_per_split",
        )
        source_universe = _canonical_symbols(
            self.source_universe, field="symbol_disjoint.source_universe"
        )
        train = _canonical_symbols(
            self.train_symbols, field="symbol_disjoint.train_symbols"
        )
        validation = _canonical_symbols(
            self.validation_symbols, field="symbol_disjoint.validation_symbols"
        )
        test = _canonical_symbols(
            self.test_symbols, field="symbol_disjoint.test_symbols"
        )
        split_values = {"train": train, "validation": validation, "test": test}
        for split, values in split_values.items():
            if len(values) < minimum:
                raise ValueError(
                    f"symbol-disjoint {split} split is below the minimum size"
                )
        if not set(train).isdisjoint(validation):
            raise ValueError("symbol-disjoint train and validation splits must be disjoint")
        if not set(train).isdisjoint(test):
            raise ValueError("symbol-disjoint train and test splits must be disjoint")
        if not set(validation).isdisjoint(test):
            raise ValueError(
                "symbol-disjoint validation and test splits must be disjoint"
            )
        if set(train) | set(validation) | set(test) != set(source_universe):
            raise ValueError("symbol-disjoint universe closure mismatch")

        expected_universe_digest = content_digest(
            {
                "schema_version": "canonical_symbol_universe_v1",
                "symbols": source_universe,
            }
        )
        if self.universe_digest != expected_universe_digest:
            raise ValueError("symbol-disjoint universe digest mismatch")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "minimum_symbols_per_split", minimum)
        object.__setattr__(self, "source_universe", source_universe)
        object.__setattr__(self, "train_symbols", train)
        object.__setattr__(self, "validation_symbols", validation)
        object.__setattr__(self, "test_symbols", test)

        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("symbol-disjoint manifest digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def all_symbols(self) -> tuple[str, ...]:
        return self.source_universe

    @property
    def split_counts(self) -> dict[str, int]:
        return {split: len(self.symbols_for(split)) for split in _SPLITS}

    def symbols_for(self, split: SymbolSplit) -> tuple[str, ...]:
        if split == "train":
            return self.train_symbols
        if split == "validation":
            return self.validation_symbols
        if split == "test":
            return self.test_symbols
        raise ValueError("symbol-disjoint split is invalid")

    def combinations_for(
        self, split: SymbolSplit, *, size: int
    ) -> tuple[tuple[str, ...], ...]:
        resolved_size = _validate_positive_int(
            size, field="symbol_disjoint.combination_size"
        )
        symbols = self.symbols_for(split)
        if resolved_size > len(symbols):
            raise ValueError("symbol-disjoint combination size exceeds split size")
        return tuple(itertools.combinations(symbols, resolved_size))

    def digest_payload(self) -> dict[str, object]:
        return {
            "minimum_symbols_per_split": self.minimum_symbols_per_split,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source_universe": self.source_universe,
            "split_counts": self.split_counts,
            "test_symbols": self.test_symbols,
            "train_symbols": self.train_symbols,
            "universe_digest": self.universe_digest,
            "validation_symbols": self.validation_symbols,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def build_symbol_disjoint_manifest(
    symbols: tuple[str, ...],
    *,
    seed: int,
    validation_count: int,
    test_count: int,
    minimum_symbols_per_split: int = 3,
) -> SymbolDisjointManifest:
    source_universe = _canonical_symbols(
        symbols, field="symbol_disjoint.source_universe"
    )
    resolved_seed = _validate_non_negative_int(seed, field="symbol_disjoint.seed")
    validation_size = _validate_non_negative_int(
        validation_count, field="symbol_disjoint.validation_count"
    )
    test_size = _validate_non_negative_int(
        test_count, field="symbol_disjoint.test_count"
    )
    minimum = _validate_positive_int(
        minimum_symbols_per_split,
        field="symbol_disjoint.minimum_symbols_per_split",
    )
    train_size = len(source_universe) - validation_size - test_size
    if min(train_size, validation_size, test_size) < minimum:
        raise ValueError("each symbol-disjoint split must satisfy the minimum size")

    ranked = tuple(
        sorted(source_universe, key=lambda symbol: _rank(resolved_seed, symbol))
    )
    train = tuple(sorted(ranked[:train_size]))
    validation = tuple(sorted(ranked[train_size : train_size + validation_size]))
    test = tuple(sorted(ranked[train_size + validation_size :]))
    universe_digest = content_digest(
        {
            "schema_version": "canonical_symbol_universe_v1",
            "symbols": source_universe,
        }
    )
    return SymbolDisjointManifest(
        source_universe=source_universe,
        universe_digest=universe_digest,
        seed=resolved_seed,
        train_symbols=train,
        validation_symbols=validation,
        test_symbols=test,
        minimum_symbols_per_split=minimum,
    )


def write_symbol_disjoint_manifest(
    path: str | Path, manifest: SymbolDisjointManifest
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(manifest.to_json_dict()))
    return output


def load_symbol_disjoint_manifest(path: str | Path) -> SymbolDisjointManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("symbol-disjoint manifest must be a JSON object")
    required = {
        "digest",
        "minimum_symbols_per_split",
        "schema_version",
        "seed",
        "source_universe",
        "split_counts",
        "test_symbols",
        "train_symbols",
        "universe_digest",
        "validation_symbols",
    }
    if set(payload) != required:
        raise ValueError("symbol-disjoint manifest field closure mismatch")
    list_fields = (
        "source_universe",
        "train_symbols",
        "validation_symbols",
        "test_symbols",
    )
    if any(not isinstance(payload[field], list) for field in list_fields):
        raise ValueError("symbol-disjoint symbol fields must be lists")
    manifest = SymbolDisjointManifest(
        source_universe=tuple(payload["source_universe"]),
        universe_digest=payload["universe_digest"],
        seed=payload["seed"],
        train_symbols=tuple(payload["train_symbols"]),
        validation_symbols=tuple(payload["validation_symbols"]),
        test_symbols=tuple(payload["test_symbols"]),
        minimum_symbols_per_split=payload["minimum_symbols_per_split"],
        schema_version=payload["schema_version"],
        digest=payload["digest"],
    )
    raw_counts = payload["split_counts"]
    if not isinstance(raw_counts, dict) or raw_counts != manifest.split_counts:
        raise ValueError("symbol-disjoint split counts mismatch")
    return manifest


__all__ = [
    "SYMBOL_DISJOINT_MANIFEST_SCHEMA",
    "SymbolDisjointManifest",
    "SymbolSplit",
    "build_symbol_disjoint_manifest",
    "load_symbol_disjoint_manifest",
    "write_symbol_disjoint_manifest",
]
