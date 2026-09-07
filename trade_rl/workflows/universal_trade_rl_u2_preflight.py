"""Train-only, FIT-bounded data preflight for Universal Trade RL U2."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeVar

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitProvenance,
    UniversalTradeRLFitPurpose,
    require_universal_trade_rl_train_only_provenance,
)
from trade_rl.workflows.universal_trade_rl_u1_contract import UniversalTradeRLU1Contract
from trade_rl.workflows.universal_trade_rl_u2_contract import UniversalTradeRLU2Contract
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    U2_DECISION_STEP_NS,
    UniversalTradeRLU2TimePartition,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseEntry,
    UniversalTradeRLUniverseManifest,
)

U2_TRAINING_SOURCE_SCHEMA: Final = "universal_trade_rl_u2_training_source_v1"
U2_TRAINING_SOURCE_CLOSURE_SCHEMA: Final = (
    "universal_trade_rl_u2_training_source_closure_v1"
)
U2_BOUNDED_DATASET_REQUEST_SCHEMA: Final = (
    "universal_trade_rl_u2_bounded_dataset_request_v1"
)

_SOURCE_KEYS: Final = (
    "schema_version",
    "symbol",
    "dataset_digest",
    "source_first_timestamp_ns",
    "source_last_timestamp_ns",
    "source_row_count",
    "fit_first_timestamp_ns",
    "fit_last_timestamp_ns",
    "fit_stop_timestamp_ns_exclusive",
    "fit_bar_count",
)
_CLOSURE_KEYS: Final = (
    "schema_version",
    "u2_contract_digest",
    "universe_manifest_digest",
    "u1_contract_digest",
    "normalizer_digest",
    "normalizer_provenance_digest",
    "time_partition_digest",
    "fit_first_timestamp_ns",
    "fit_last_timestamp_ns",
    "fit_stop_timestamp_ns_exclusive",
    "fit_bar_count",
    "sources",
    "artifact_digest",
)

_LoadedT = TypeVar("_LoadedT")


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer and not boolean")
    return value


def _exact_mapping(
    value: object,
    *,
    keys: tuple[str, ...],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object with exact keys")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field} keys must be strings")
        result[key] = item
    if set(result) != set(keys) or len(result) != len(keys):
        raise ValueError(f"{field} must use exact keys")
    return result


def _sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


def _require_dense_grid(
    *,
    first_timestamp_ns: int,
    last_timestamp_ns: int,
    row_count: int,
    field: str,
) -> None:
    if first_timestamp_ns < 0 or last_timestamp_ns < first_timestamp_ns:
        raise ValueError(f"{field} timestamps are invalid")
    if row_count <= 0:
        raise ValueError(f"{field} row count must be positive")
    expected_last = first_timestamp_ns + (row_count - 1) * U2_DECISION_STEP_NS
    if last_timestamp_ns != expected_last:
        raise ValueError(f"{field} must be a dense 15m grid")


@dataclass(frozen=True, slots=True)
class U2TrainingSource:
    """One U0 Train source closed to the preregistered FIT interval."""

    symbol: str
    dataset_digest: str
    source_first_timestamp_ns: int
    source_last_timestamp_ns: int
    source_row_count: int
    fit_first_timestamp_ns: int
    fit_last_timestamp_ns: int
    fit_stop_timestamp_ns_exclusive: int
    fit_bar_count: int
    schema_version: str = U2_TRAINING_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != U2_TRAINING_SOURCE_SCHEMA:
            raise ValueError("unsupported U2 training source schema")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("U2 training source symbol must be non-empty")
        require_sha256(
            self.dataset_digest, field=f"U2 source {self.symbol} dataset digest"
        )

        source_first = _integer(
            self.source_first_timestamp_ns,
            field="U2 source first timestamp",
        )
        source_last = _integer(
            self.source_last_timestamp_ns,
            field="U2 source last timestamp",
        )
        source_rows = _integer(self.source_row_count, field="U2 source row count")
        fit_first = _integer(
            self.fit_first_timestamp_ns,
            field="U2 FIT first timestamp",
        )
        fit_last = _integer(
            self.fit_last_timestamp_ns,
            field="U2 FIT last timestamp",
        )
        fit_stop = _integer(
            self.fit_stop_timestamp_ns_exclusive,
            field="U2 FIT exclusive stop timestamp",
        )
        fit_bars = _integer(self.fit_bar_count, field="U2 FIT bar count")
        _require_dense_grid(
            first_timestamp_ns=source_first,
            last_timestamp_ns=source_last,
            row_count=source_rows,
            field="U2 source",
        )
        _require_dense_grid(
            first_timestamp_ns=fit_first,
            last_timestamp_ns=fit_last,
            row_count=fit_bars,
            field="U2 FIT interval",
        )
        if fit_stop != fit_last + U2_DECISION_STEP_NS:
            raise ValueError(
                "U2 FIT exclusive stop must be one 15m step after FIT last"
            )
        fit_offset_ns = fit_first - source_first
        if fit_offset_ns < 0 or fit_offset_ns % U2_DECISION_STEP_NS != 0:
            raise ValueError("U2 FIT interval must align to the source 15m grid")
        if source_first > fit_first or source_last < fit_last:
            raise ValueError("U2 source does not fully cover the FIT interval")

    @property
    def fit_start_index(self) -> int:
        return (
            self.fit_first_timestamp_ns - self.source_first_timestamp_ns
        ) // U2_DECISION_STEP_NS

    @property
    def fit_stop_index(self) -> int:
        return self.fit_start_index + self.fit_bar_count

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "dataset_digest": self.dataset_digest,
            "source_first_timestamp_ns": self.source_first_timestamp_ns,
            "source_last_timestamp_ns": self.source_last_timestamp_ns,
            "source_row_count": self.source_row_count,
            "fit_first_timestamp_ns": self.fit_first_timestamp_ns,
            "fit_last_timestamp_ns": self.fit_last_timestamp_ns,
            "fit_stop_timestamp_ns_exclusive": self.fit_stop_timestamp_ns_exclusive,
            "fit_bar_count": self.fit_bar_count,
        }

    @classmethod
    def from_payload(cls, payload: object) -> U2TrainingSource:
        values = _exact_mapping(payload, keys=_SOURCE_KEYS, field="U2 training source")
        schema = values["schema_version"]
        symbol = values["symbol"]
        dataset_digest = values["dataset_digest"]
        if not isinstance(schema, str):
            raise ValueError("U2 training source schema_version must be a string")
        if not isinstance(symbol, str) or not isinstance(dataset_digest, str):
            raise ValueError("U2 training source identity fields must be strings")
        return cls(
            symbol=symbol,
            dataset_digest=dataset_digest,
            source_first_timestamp_ns=_integer(
                values["source_first_timestamp_ns"],
                field="U2 source first timestamp",
            ),
            source_last_timestamp_ns=_integer(
                values["source_last_timestamp_ns"],
                field="U2 source last timestamp",
            ),
            source_row_count=_integer(
                values["source_row_count"],
                field="U2 source row count",
            ),
            fit_first_timestamp_ns=_integer(
                values["fit_first_timestamp_ns"],
                field="U2 FIT first timestamp",
            ),
            fit_last_timestamp_ns=_integer(
                values["fit_last_timestamp_ns"],
                field="U2 FIT last timestamp",
            ),
            fit_stop_timestamp_ns_exclusive=_integer(
                values["fit_stop_timestamp_ns_exclusive"],
                field="U2 FIT exclusive stop timestamp",
            ),
            fit_bar_count=_integer(values["fit_bar_count"], field="U2 FIT bar count"),
            schema_version=schema,
        )


@dataclass(frozen=True, slots=True)
class U2TrainingSourceClosure:
    """Content-addressed proof that only verified U0 Train sources can enter U2 FIT."""

    u2_contract_digest: str
    universe_manifest_digest: str
    u1_contract_digest: str
    normalizer_digest: str
    normalizer_provenance_digest: str
    time_partition_digest: str
    fit_first_timestamp_ns: int
    fit_last_timestamp_ns: int
    fit_stop_timestamp_ns_exclusive: int
    fit_bar_count: int
    sources: tuple[U2TrainingSource, ...]
    schema_version: str = U2_TRAINING_SOURCE_CLOSURE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_TRAINING_SOURCE_CLOSURE_SCHEMA:
            raise ValueError("unsupported U2 training source closure schema")
        for field_name, value in (
            ("u2_contract_digest", self.u2_contract_digest),
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("normalizer_digest", self.normalizer_digest),
            ("normalizer_provenance_digest", self.normalizer_provenance_digest),
            ("time_partition_digest", self.time_partition_digest),
        ):
            require_sha256(value, field=f"U2 source closure {field_name}")

        fit_first = _integer(
            self.fit_first_timestamp_ns,
            field="U2 closure FIT first timestamp",
        )
        fit_last = _integer(
            self.fit_last_timestamp_ns,
            field="U2 closure FIT last timestamp",
        )
        fit_stop = _integer(
            self.fit_stop_timestamp_ns_exclusive,
            field="U2 closure FIT exclusive stop timestamp",
        )
        fit_bars = _integer(
            self.fit_bar_count,
            field="U2 closure FIT bar count",
        )
        _require_dense_grid(
            first_timestamp_ns=fit_first,
            last_timestamp_ns=fit_last,
            row_count=fit_bars,
            field="U2 closure FIT interval",
        )
        if fit_stop != fit_last + U2_DECISION_STEP_NS:
            raise ValueError("U2 closure FIT exclusive stop drifted")

        sources = tuple(self.sources)
        if not sources or any(
            not isinstance(source, U2TrainingSource) for source in sources
        ):
            raise TypeError("U2 training source closure requires valid source records")
        symbols = tuple(source.symbol for source in sources)
        if symbols != tuple(sorted(symbols)) or len(set(symbols)) != len(symbols):
            raise ValueError(
                "U2 training source closure symbols must be unique and canonical"
            )
        for source in sources:
            if (
                source.fit_first_timestamp_ns != fit_first
                or source.fit_last_timestamp_ns != fit_last
                or source.fit_stop_timestamp_ns_exclusive != fit_stop
                or source.fit_bar_count != fit_bars
            ):
                raise ValueError("U2 training source FIT bounds differ from closure")
        object.__setattr__(self, "sources", sources)

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(
                self.digest, field="U2 training source closure artifact digest"
            )
            if self.digest != expected:
                raise ValueError("U2 training source closure digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "u2_contract_digest": self.u2_contract_digest,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "normalizer_digest": self.normalizer_digest,
            "normalizer_provenance_digest": self.normalizer_provenance_digest,
            "time_partition_digest": self.time_partition_digest,
            "fit_first_timestamp_ns": self.fit_first_timestamp_ns,
            "fit_last_timestamp_ns": self.fit_last_timestamp_ns,
            "fit_stop_timestamp_ns_exclusive": self.fit_stop_timestamp_ns_exclusive,
            "fit_bar_count": self.fit_bar_count,
            "sources": tuple(source.to_payload() for source in self.sources),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> U2TrainingSourceClosure:
        values = _exact_mapping(payload, keys=_CLOSURE_KEYS, field="U2 source closure")
        string_fields = (
            "schema_version",
            "u2_contract_digest",
            "universe_manifest_digest",
            "u1_contract_digest",
            "normalizer_digest",
            "normalizer_provenance_digest",
            "time_partition_digest",
            "artifact_digest",
        )
        strings: dict[str, str] = {}
        for field_name in string_fields:
            value = values[field_name]
            if not isinstance(value, str):
                raise ValueError(f"U2 source closure {field_name} must be a string")
            strings[field_name] = value
        sources = tuple(
            U2TrainingSource.from_payload(item)
            for item in _sequence(values["sources"], field="U2 source closure sources")
        )
        return cls(
            u2_contract_digest=strings["u2_contract_digest"],
            universe_manifest_digest=strings["universe_manifest_digest"],
            u1_contract_digest=strings["u1_contract_digest"],
            normalizer_digest=strings["normalizer_digest"],
            normalizer_provenance_digest=strings["normalizer_provenance_digest"],
            time_partition_digest=strings["time_partition_digest"],
            fit_first_timestamp_ns=_integer(
                values["fit_first_timestamp_ns"],
                field="U2 closure FIT first timestamp",
            ),
            fit_last_timestamp_ns=_integer(
                values["fit_last_timestamp_ns"],
                field="U2 closure FIT last timestamp",
            ),
            fit_stop_timestamp_ns_exclusive=_integer(
                values["fit_stop_timestamp_ns_exclusive"],
                field="U2 closure FIT exclusive stop timestamp",
            ),
            fit_bar_count=_integer(
                values["fit_bar_count"],
                field="U2 closure FIT bar count",
            ),
            sources=sources,
            schema_version=strings["schema_version"],
            digest=strings["artifact_digest"],
        )


@dataclass(frozen=True, slots=True)
class U2BoundedDatasetRequest:
    """The only maintained U2 numeric-read request: one source bounded to FIT."""

    symbol: str
    dataset_digest: str
    source_first_timestamp_ns: int
    source_last_timestamp_ns: int
    source_row_count: int
    fit_first_timestamp_ns: int
    fit_last_timestamp_ns: int
    fit_stop_timestamp_ns_exclusive: int
    fit_bar_count: int
    schema_version: str = U2_BOUNDED_DATASET_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != U2_BOUNDED_DATASET_REQUEST_SCHEMA:
            raise ValueError("unsupported U2 bounded dataset request schema")
        U2TrainingSource(
            symbol=self.symbol,
            dataset_digest=self.dataset_digest,
            source_first_timestamp_ns=self.source_first_timestamp_ns,
            source_last_timestamp_ns=self.source_last_timestamp_ns,
            source_row_count=self.source_row_count,
            fit_first_timestamp_ns=self.fit_first_timestamp_ns,
            fit_last_timestamp_ns=self.fit_last_timestamp_ns,
            fit_stop_timestamp_ns_exclusive=self.fit_stop_timestamp_ns_exclusive,
            fit_bar_count=self.fit_bar_count,
        )

    @classmethod
    def from_source(cls, source: U2TrainingSource) -> U2BoundedDatasetRequest:
        if not isinstance(source, U2TrainingSource):
            raise TypeError("U2 bounded dataset request requires a verified source")
        return cls(
            symbol=source.symbol,
            dataset_digest=source.dataset_digest,
            source_first_timestamp_ns=source.source_first_timestamp_ns,
            source_last_timestamp_ns=source.source_last_timestamp_ns,
            source_row_count=source.source_row_count,
            fit_first_timestamp_ns=source.fit_first_timestamp_ns,
            fit_last_timestamp_ns=source.fit_last_timestamp_ns,
            fit_stop_timestamp_ns_exclusive=source.fit_stop_timestamp_ns_exclusive,
            fit_bar_count=source.fit_bar_count,
        )


def _train_entries(
    manifest: UniversalTradeRLUniverseManifest,
) -> tuple[UniversalTradeRLUniverseEntry, ...]:
    return tuple(
        entry
        for entry in manifest.entries
        if entry.role is UniversalTradeRLSymbolRole.TRAIN
    )


def build_universal_trade_rl_u2_training_source_closure(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    u1_contract: UniversalTradeRLU1Contract,
    u2_contract: UniversalTradeRLU2Contract,
    time_partition: UniversalTradeRLU2TimePartition,
    normalizer: UniversalTradeSequenceNormalizer,
    normalizer_provenance: UniversalTradeRLFitProvenance,
) -> U2TrainingSourceClosure:
    """Verify all U0/U1/FIT identities before any maintained U2 numeric read."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 preflight requires a U0 universe manifest")
    if not isinstance(u1_contract, UniversalTradeRLU1Contract):
        raise TypeError("U2 preflight requires a U1 contract")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 preflight requires a U2 contract")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 preflight requires a U2 time partition")
    if not isinstance(normalizer, UniversalTradeSequenceNormalizer):
        raise TypeError("U2 preflight requires a U1 sequence normalizer")
    if not isinstance(normalizer_provenance, UniversalTradeRLFitProvenance):
        raise TypeError("U2 preflight requires normalizer provenance")

    if manifest.digest != u2_contract.universe_manifest_digest:
        raise ValueError("U2 preflight U0 manifest identity mismatch")
    if u1_contract.digest != u2_contract.u1_contract_digest:
        raise ValueError("U2 preflight U1 contract digest identity mismatch")
    if u1_contract.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 preflight U1/U0 universe identity mismatch")
    if time_partition.digest != u2_contract.time_partition_digest:
        raise ValueError("U2 preflight time partition digest identity mismatch")
    if time_partition.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 preflight time partition/U0 universe identity mismatch")
    if u2_contract.fit_end_ns != time_partition.fit_end_ns:
        raise ValueError("U2 preflight U2 contract FIT end mismatch")

    if u1_contract.normalizer_digest != u2_contract.u1_normalizer_digest:
        raise ValueError("U2 preflight U1/U2 normalizer digest mismatch")
    if normalizer.digest != u1_contract.normalizer_digest:
        raise ValueError("U2 preflight normalizer artifact digest mismatch")
    if (
        normalizer_provenance.purpose
        is not UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION
    ):
        raise ValueError(
            "U2 normalizer provenance purpose must be FEATURE_NORMALIZATION"
        )
    require_universal_trade_rl_train_only_provenance(
        normalizer_provenance,
        manifest=manifest,
    )
    if normalizer_provenance.digest != u1_contract.normalizer_provenance_digest:
        raise ValueError("U2 preflight U1 normalizer provenance digest mismatch")
    if normalizer.provenance_digest != normalizer_provenance.digest:
        raise ValueError("U2 preflight normalizer/provenance artifact mismatch")
    if normalizer.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 preflight normalizer/U0 manifest identity mismatch")
    if normalizer.contract_digest != u1_contract.policy_contract_digest:
        raise ValueError("U2 preflight normalizer/U1 policy contract mismatch")
    if normalizer.clip_value != u1_contract.normalizer_clip_value:
        raise ValueError("U2 preflight normalizer/U1 clip semantics mismatch")

    fit = time_partition.window("fit")
    fit_end = fit.last_timestamp_ns
    if (
        u1_contract.normalizer_knowledge_cutoff_ns != fit_end
        or normalizer.knowledge_cutoff_ns != fit_end
        or normalizer_provenance.knowledge_cutoff != fit_end
    ):
        raise ValueError("U2 normalizer cutoff must equal preregistered FIT end")

    train_entries = _train_entries(manifest)
    train_symbols = tuple(entry.symbol for entry in train_entries)
    if not train_symbols:
        raise ValueError("U2 preflight requires at least one Train symbol")
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    access.require_fit_scope(train_symbols)
    if normalizer_provenance.source_symbols != train_symbols:
        raise ValueError("U2 normalizer provenance must cover the complete Train scope")
    if normalizer.train_symbols != train_symbols:
        raise ValueError("U2 normalizer Train symbol identity mismatch")

    expected_dataset_digests = tuple(
        (entry.symbol, entry.dataset_digest) for entry in train_entries
    )
    if normalizer_provenance.source_dataset_digests != expected_dataset_digests:
        raise ValueError("U2 normalizer provenance dataset source identity mismatch")
    if normalizer.source_dataset_digests != expected_dataset_digests:
        raise ValueError(
            "U2 normalizer dataset source identity drifted from U0 manifest"
        )

    fit_stop = fit.last_timestamp_ns + U2_DECISION_STEP_NS
    sources = tuple(
        U2TrainingSource(
            symbol=entry.symbol,
            dataset_digest=entry.dataset_digest,
            source_first_timestamp_ns=entry.first_timestamp_ns,
            source_last_timestamp_ns=entry.last_timestamp_ns,
            source_row_count=entry.row_count,
            fit_first_timestamp_ns=fit.first_timestamp_ns,
            fit_last_timestamp_ns=fit.last_timestamp_ns,
            fit_stop_timestamp_ns_exclusive=fit_stop,
            fit_bar_count=fit.bar_count,
        )
        for entry in train_entries
    )
    return U2TrainingSourceClosure(
        u2_contract_digest=u2_contract.digest,
        universe_manifest_digest=manifest.digest,
        u1_contract_digest=u1_contract.digest,
        normalizer_digest=normalizer.digest,
        normalizer_provenance_digest=normalizer_provenance.digest,
        time_partition_digest=time_partition.digest,
        fit_first_timestamp_ns=fit.first_timestamp_ns,
        fit_last_timestamp_ns=fit.last_timestamp_ns,
        fit_stop_timestamp_ns_exclusive=fit_stop,
        fit_bar_count=fit.bar_count,
        sources=sources,
    )


def load_universal_trade_rl_u2_fit_sources(
    *,
    closure: U2TrainingSourceClosure,
    requested_symbols: Sequence[str],
    loader: Callable[[U2BoundedDatasetRequest], _LoadedT],
) -> tuple[_LoadedT, ...]:
    """Validate the complete request set, then perform only FIT-bounded numeric reads."""

    if not isinstance(closure, U2TrainingSourceClosure):
        raise TypeError(
            "U2 numeric loading requires a verified training source closure"
        )
    if isinstance(requested_symbols, (str, bytes)) or not isinstance(
        requested_symbols, Sequence
    ):
        raise TypeError("U2 requested symbols must be a sequence")
    requested = tuple(requested_symbols)
    if not requested:
        raise ValueError("U2 requested Train symbol scope cannot be empty")
    if any(not isinstance(symbol, str) or not symbol for symbol in requested):
        raise ValueError("U2 requested Train symbols must be non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("U2 requested Train symbol scope cannot contain duplicates")
    if not callable(loader):
        raise TypeError("U2 FIT numeric loader must be callable")

    by_symbol = {source.symbol: source for source in closure.sources}
    missing = tuple(symbol for symbol in requested if symbol not in by_symbol)
    if missing:
        raise ValueError(
            "requested symbol is outside the verified U2 Train training scope: "
            + ", ".join(missing)
        )

    requests = tuple(
        U2BoundedDatasetRequest.from_source(by_symbol[symbol]) for symbol in requested
    )
    return tuple(loader(request) for request in requests)


__all__ = [
    "U2_BOUNDED_DATASET_REQUEST_SCHEMA",
    "U2_TRAINING_SOURCE_CLOSURE_SCHEMA",
    "U2_TRAINING_SOURCE_SCHEMA",
    "U2BoundedDatasetRequest",
    "U2TrainingSource",
    "U2TrainingSourceClosure",
    "build_universal_trade_rl_u2_training_source_closure",
    "load_universal_trade_rl_u2_fit_sources",
]
