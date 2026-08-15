"""Execution and run identities for the hardened causal alpha V3 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

_EXECUTION_SCHEMA: Final = "causal_alpha_v3_execution_identity_v1"
_RUN_SCHEMA: Final = "causal_alpha_v3_run_manifest_v2"


def _strict_payload(
    raw: Mapping[str, Any],
    *,
    fields: frozenset[str],
    schema: str,
    label: str,
) -> dict[str, Any]:
    values = dict(raw)
    if set(values) != fields:
        missing = sorted(fields - set(values))
        unknown = sorted(set(values) - fields)
        raise ValueError(
            f"{label} fields mismatch; missing={missing}, unknown={unknown}"
        )
    if values.get("schema_version") != schema:
        raise ValueError(f"{label} schema is unsupported")
    return values


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ExecutionIdentity:
    """Immutable identity for code/runtime semantics used by V3 replay."""

    train_symbols: tuple[str, ...]
    training_contract_digest: str
    instrument_context_schema_digest: str
    source_tree_digest: str
    symbol_runtime_digests: tuple[tuple[str, str], ...]
    schema_version: str = _EXECUTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if (
            not symbols
            or len(set(symbols)) != len(symbols)
            or any(not symbol for symbol in symbols)
        ):
            raise ValueError("V3 execution identity train_symbols must be unique")
        for name in (
            "training_contract_digest",
            "instrument_context_schema_digest",
            "source_tree_digest",
        ):
            require_sha256(getattr(self, name), field=f"V3 execution {name}")
        runtime = tuple(
            (str(symbol), str(digest)) for symbol, digest in self.symbol_runtime_digests
        )
        if tuple(symbol for symbol, _ in runtime) != symbols:
            raise ValueError("V3 execution runtime scope/order drifted")
        for symbol, digest in runtime:
            require_sha256(digest, field=f"V3 runtime digest {symbol}")
        if self.schema_version != _EXECUTION_SCHEMA:
            raise ValueError("unsupported V3 execution identity schema")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "symbol_runtime_digests", runtime)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 execution identity digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "instrument_context_schema_digest": self.instrument_context_schema_digest,
            "schema_version": self.schema_version,
            "source_tree_digest": self.source_tree_digest,
            "symbol_runtime_digests": self.symbol_runtime_digests,
            "train_symbols": self.train_symbols,
            "training_contract_digest": self.training_contract_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> CausalAlphaV3ExecutionIdentity:
        fields = frozenset(
            {
                "artifact_digest",
                "instrument_context_schema_digest",
                "schema_version",
                "source_tree_digest",
                "symbol_runtime_digests",
                "train_symbols",
                "training_contract_digest",
            }
        )
        values = _strict_payload(
            raw,
            fields=fields,
            schema=_EXECUTION_SCHEMA,
            label="V3 execution identity",
        )
        return cls(
            train_symbols=tuple(str(item) for item in values["train_symbols"]),
            training_contract_digest=str(values["training_contract_digest"]),
            instrument_context_schema_digest=str(
                values["instrument_context_schema_digest"]
            ),
            source_tree_digest=str(values["source_tree_digest"]),
            symbol_runtime_digests=tuple(
                (str(symbol), str(digest))
                for symbol, digest in values["symbol_runtime_digests"]
            ),
            schema_version=str(values["schema_version"]),
            digest=str(values["artifact_digest"]),
        )


@dataclass(frozen=True, slots=True)
class CausalAlphaV3RunManifestV2:
    """V3 run identity that closes resume over runtime/execution semantics."""

    train_symbols: tuple[str, ...]
    config_digest: str
    catalog_digest: str
    partition_digest: str
    split_manifest_digest: str
    feature_schema_digest: str
    statistics_digest: str
    generator_code_digest: str
    nested_partition_digest: str
    execution_identity_digest: str
    training_contract_digest: str
    instrument_context_schema_digest: str
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = _RUN_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if (
            not symbols
            or len(set(symbols)) != len(symbols)
            or any(not symbol for symbol in symbols)
        ):
            raise ValueError("V3 run manifest train_symbols must be unique")
        for name in (
            "config_digest",
            "catalog_digest",
            "partition_digest",
            "split_manifest_digest",
            "feature_schema_digest",
            "statistics_digest",
            "generator_code_digest",
            "nested_partition_digest",
            "execution_identity_digest",
            "training_contract_digest",
            "instrument_context_schema_digest",
        ):
            require_sha256(getattr(self, name), field=f"V3 run manifest {name}")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V3 run manifest must remain research-only")
        if self.schema_version != _RUN_SCHEMA:
            raise ValueError("unsupported V3 run manifest schema")
        object.__setattr__(self, "train_symbols", symbols)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 run manifest digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "catalog_digest": self.catalog_digest,
            "config_digest": self.config_digest,
            "execution_identity_digest": self.execution_identity_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "generator_code_digest": self.generator_code_digest,
            "instrument_context_schema_digest": self.instrument_context_schema_digest,
            "nested_partition_digest": self.nested_partition_digest,
            "partition_digest": self.partition_digest,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "schema_version": self.schema_version,
            "split_manifest_digest": self.split_manifest_digest,
            "statistics_digest": self.statistics_digest,
            "train_symbols": self.train_symbols,
            "training_contract_digest": self.training_contract_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> CausalAlphaV3RunManifestV2:
        fields = frozenset(
            {
                "artifact_digest",
                "catalog_digest",
                "config_digest",
                "execution_identity_digest",
                "feature_schema_digest",
                "generator_code_digest",
                "instrument_context_schema_digest",
                "nested_partition_digest",
                "partition_digest",
                "promotion_eligible",
                "research_only",
                "schema_version",
                "split_manifest_digest",
                "statistics_digest",
                "train_symbols",
                "training_contract_digest",
            }
        )
        values = _strict_payload(
            raw,
            fields=fields,
            schema=_RUN_SCHEMA,
            label="V3 run manifest",
        )
        if (
            values["research_only"] is not True
            or values["promotion_eligible"] is not False
        ):
            raise ValueError("V3 run manifest safety flags are invalid")
        return cls(
            train_symbols=tuple(str(item) for item in values["train_symbols"]),
            config_digest=str(values["config_digest"]),
            catalog_digest=str(values["catalog_digest"]),
            partition_digest=str(values["partition_digest"]),
            split_manifest_digest=str(values["split_manifest_digest"]),
            feature_schema_digest=str(values["feature_schema_digest"]),
            statistics_digest=str(values["statistics_digest"]),
            generator_code_digest=str(values["generator_code_digest"]),
            nested_partition_digest=str(values["nested_partition_digest"]),
            execution_identity_digest=str(values["execution_identity_digest"]),
            training_contract_digest=str(values["training_contract_digest"]),
            instrument_context_schema_digest=str(
                values["instrument_context_schema_digest"]
            ),
            research_only=True,
            promotion_eligible=False,
            schema_version=str(values["schema_version"]),
            digest=str(values["artifact_digest"]),
        )


__all__ = ["CausalAlphaV3ExecutionIdentity", "CausalAlphaV3RunManifestV2"]
