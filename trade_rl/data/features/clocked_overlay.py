"""Attach explicit higher-clock price/volume features to an existing Dataset.

This is a feature-only composition on a maintained execution clock. It does
not construct coarse-clock execution rows or infer funding, publication, or
a provider's bar finality from prices.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketCalendarKind,
    timeframe_hours,
)
from trade_rl.data.features.cross_asset import CROSS_ASSET_FEATURE_KINDS
from trade_rl.data.features.multitimeframe import align_clocked_native_feature
from trade_rl.data.identity import content_and_arrays_digest, parse_identity_json
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_clock import NativeBarClock
from trade_rl.data.source import NativeRawMarketSeries, RawMarketSeries

_OVERLAY_SCHEMA = "clocked_native_feature_overlay_v1"
_UNSUPPORTED_KINDS = CROSS_ASSET_FEATURE_KINDS | {
    FeatureKind.FUNDING_BPS,
    FeatureKind.FUNDING_CHANGE,
    FeatureKind.FUNDING_ZSCORE,
}


def _clock_payload(clock: NativeBarClock) -> dict[str, object]:
    if type(clock) is not NativeBarClock:
        raise ValueError("expected an explicit NativeBarClock")
    return {
        "kind": clock.kind.value,
        "fixed_seconds": clock.fixed_seconds,
        "fixed_anchor_ns": (
            None
            if clock.fixed_anchor is None
            else int(clock.fixed_anchor.astype(np.int64))
        ),
    }


def clocked_native_source_digest(
    raw: RawMarketSeries,
    *,
    contract: InstrumentContract,
    clock: NativeBarClock,
) -> str:
    """Bind all normalized source arrays, lifecycle and explicit native clock.

    The source must consist of complete, tradable rows on its declared clock.
    This digest is not proof of venue authenticity or publication history.
    """
    if (
        type(raw) not in (RawMarketSeries, NativeRawMarketSeries)
        or type(contract) is not InstrumentContract
    ):
        raise ValueError("expected a RawMarketSeries and InstrumentContract")
    clock_payload = _clock_payload(clock)
    clock.interval_bounds(raw.timestamps)
    if not np.all(raw.tradable):
        raise ValueError("higher-clock source must contain only complete tradable bars")
    arrays: list[tuple[str, np.ndarray]] = []
    for item in fields(RawMarketSeries):
        value = getattr(raw, item.name)
        if not isinstance(value, np.ndarray):
            raise ValueError("native source array is not resolved")
        arrays.append((item.name, value))
    return content_and_arrays_digest(
        {
            "schema": "clocked_native_raw_source_v1",
            "clock": clock_payload,
            "instrument": contract.canonical_payload(),
        },
        arrays,
    )


@dataclass(frozen=True, slots=True)
class ClockedNativeSource:
    """One symbol's complete normalized bars and independently expected pin."""

    contract: InstrumentContract
    raw: RawMarketSeries
    expected_source_digest: str

    def __post_init__(self) -> None:
        if type(self.contract) is not InstrumentContract or type(self.raw) not in (
            RawMarketSeries,
            NativeRawMarketSeries,
        ):
            raise ValueError("expected a native source and instrument contract")
        if type(self.expected_source_digest) is not str:
            raise ValueError("expected source digest must be a native string")
        require_sha256(self.expected_source_digest, field="expected_source_digest")


@dataclass(frozen=True, slots=True)
class ClockedNativeFeatureGroup:
    """One native clock, complete symbol roster, and ordered feature specs."""

    clock: NativeBarClock
    sources: tuple[ClockedNativeSource, ...]
    specs: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        _clock_payload(self.clock)
        if (
            type(self.sources) is not tuple
            or not self.sources
            or any(type(source) is not ClockedNativeSource for source in self.sources)
        ):
            raise ValueError("native source roster must be a nonempty tuple")
        if (
            type(self.specs) is not tuple
            or not self.specs
            or any(
                type(spec) is not FeatureSpec
                or type(spec.kind) is not FeatureKind
                or spec.kind in _UNSUPPORTED_KINDS
                or spec.timeframe is not None
                for spec in self.specs
            )
        ):
            raise ValueError(
                "overlay requires explicit single-instrument price/volume feature specs"
            )


def append_clocked_native_features(
    dataset: MarketDataset,
    *,
    clock: NativeBarClock,
    sources: tuple[ClockedNativeSource, ...],
    specs: tuple[FeatureSpec, ...],
) -> MarketDataset:
    """Convenience entry point for one explicitly pinned higher-clock group."""
    return append_clocked_native_feature_groups(
        dataset, groups=(ClockedNativeFeatureGroup(clock, sources, specs),)
    )


def append_clocked_native_feature_groups(
    dataset: MarketDataset,
    *,
    groups: tuple[ClockedNativeFeatureGroup, ...],
) -> MarketDataset:
    """Compose distinct higher-clock feature groups on the original execution grid.

    Admission requires an identity-verified, canonical market_build_v3
    continuous Dataset on a maintained base interval, without an external
    global overlay. Every native interval is larger than and aligned to the
    base grid, and every group binds the full ordered base instrument roster.
    Submit all groups together; chaining overlays is not admitted.

    The canonical native event/publication alignment is reused. Features
    at unavailable/inactive/nontradable base rows are masked. Existing
    features, execution economics and calendar remain unchanged. The output
    uses existing v6/v7 content identity and artifact codec with versioned
    transformation and dependency metadata, not a new MarketBuildConfig.
    This verifies normalized input content, not provider authenticity or
    publication history, and grants no research or execution authority.
    """
    if type(dataset) is not MarketDataset or not dataset.identity_verified:
        raise ValueError("clocked feature overlay requires a verified Dataset identity")
    if (
        dataset.calendar_kind is not MarketCalendarKind.CONTINUOUS
        or not dataset.regular_cadence
    ):
        raise ValueError("clocked feature overlay requires a continuous base clock")
    if any(name.startswith("xa_") for name in dataset.global_feature_names):
        raise ValueError(
            "external global feature provenance is not admitted by this overlay"
        )
    assert dataset.identity_payload_json is not None
    source_identity = parse_identity_json(dataset.identity_payload_json)
    config = source_identity.get("config")
    instruments = source_identity.get("instruments")
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != "market_build_v3"
        or content_digest(config) != dataset.feature_config_digest
        or not isinstance(instruments, list)
    ):
        raise ValueError("base identity requires a canonical market_build_v3 contract")
    base_timeframe = config.get("base_timeframe")
    if (
        not isinstance(base_timeframe, str)
        or timeframe_hours(base_timeframe) != dataset.bar_hours
    ):
        raise ValueError("base identity and execution clock differ")
    original_specs = config.get("features")
    if (
        not isinstance(original_specs, list)
        or any(not isinstance(spec, dict) for spec in original_specs)
        or tuple(spec.get("name") for spec in original_specs) != dataset.feature_names
    ):
        raise ValueError("base feature dependency roster differs from Dataset")

    if (
        type(groups) is not tuple
        or not groups
        or any(type(group) is not ClockedNativeFeatureGroup for group in groups)
    ):
        raise ValueError("expected nonempty native clock groups")
    clock_payloads = tuple(_clock_payload(group.clock) for group in groups)
    clock_keys = tuple(content_digest(payload) for payload in clock_payloads)
    if len(set(clock_keys)) != len(clock_keys):
        raise ValueError("native clock groups must be distinct")
    specs = tuple(spec for group in groups for spec in group.specs)
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names) or set(names) & set(dataset.feature_names):
        raise ValueError("overlay feature names must be distinct and new")
    if source_identity.get("dataset_contract") != to_json_value(
        dataset.identity_contract_payload()
    ):
        raise ValueError("base instrument or contract metadata differ from identity")

    first = int(dataset.timestamps[0].astype("datetime64[ns]").astype(np.int64))
    last = int(dataset.timestamps[-1].astype("datetime64[ns]").astype(np.int64))
    step = int(dataset.timestamps[1].astype("datetime64[ns]").astype(np.int64)) - first
    # Verify every group and source before calculating any features.
    for group in groups:
        sources = group.sources
        if tuple(source.contract.symbol for source in sources) != dataset.symbols:
            raise ValueError("native source roster must match the base Dataset")
        if (
            to_json_value(
                tuple(source.contract.canonical_payload() for source in sources)
            )
            != instruments
        ):
            raise ValueError("native instrument contracts differ from base identity")
        if tuple(
            source.contract.volume_unit for source in sources
        ) != dataset.volume_units or not np.array_equal(
            np.asarray(
                [source.contract.contract_multiplier for source in sources],
                dtype=np.float64,
            ),
            dataset.resolved_array("contract_multipliers"),
        ):
            raise ValueError("base instrument economics differ from identity")
        for source in sources:
            starts, closes = group.clock.interval_bounds(source.raw.timestamps)
            for start, close in zip(
                starts.view(np.int64), closes.view(np.int64), strict=True
            ):
                lo, hi = int(start), int(close)
                if (lo - first) % step or (hi - first) % step or hi - lo <= step:
                    raise ValueError(
                        "native intervals must be larger than and aligned to the base clock"
                    )
            if int(closes[-1].astype(np.int64)) > last:
                raise ValueError("native source close extends beyond base Dataset")
            actual = clocked_native_source_digest(
                source.raw, contract=source.contract, clock=group.clock
            )
            if actual != source.expected_source_digest:
                raise ValueError("native source digest mismatch")

    shape = (dataset.n_bars, dataset.n_symbols, len(specs))
    values = np.zeros(shape, dtype=np.float32)
    available = np.zeros(shape, dtype=np.bool_)
    ages = np.zeros(shape, dtype=np.float32)
    stale = np.ones(shape, dtype=np.float32)
    base_usable = (
        dataset.resolved_array("symbol_active")
        & dataset.resolved_array("information_available")
        & dataset.tradable
        & (dataset.resolved_array("available_at") <= dataset.timestamps[:, None])
    )
    offset = 0
    for group in groups:
        for j, source in enumerate(group.sources):
            for k, spec in enumerate(group.specs, start=offset):
                v, mask, age, staleness = align_clocked_native_feature(
                    spec,
                    source.raw,
                    source.contract,
                    dataset.timestamps,
                    base_usable[:, j],
                    clock=group.clock,
                )
                if (
                    not np.isfinite(v).all()
                    or not np.isfinite(age).all()
                    or np.any(np.abs(v) > np.finfo(np.float32).max)
                    or np.any(np.abs(age) > np.finfo(np.float32).max)
                ):
                    raise ValueError(
                        "overlay feature values or age exceed Dataset representation"
                    )
                values[:, j, k] = v
                available[:, j, k] = mask
                ages[:, j, k] = age
                stale[:, j, k] = staleness
        offset += len(group.specs)

    group_definitions = tuple(
        {
            "clock": payload,
            "specs": tuple(spec.canonical_payload() for spec in group.specs),
        }
        for group, payload in zip(groups, clock_payloads, strict=True)
    )
    definition = {
        "schema": _OVERLAY_SCHEMA,
        "groups": group_definitions,
        "base_feature_config_digest": dataset.feature_config_digest,
    }
    feature_config_digest = content_digest(definition)
    normalization_digest = content_and_arrays_digest(
        {
            "schema": "clocked_native_feature_normalization_v1",
            "feature_config_digest": feature_config_digest,
            "base_normalization_digest": dataset.normalization_digest,
        },
        (
            ("features", values),
            ("feature_available", available),
            ("feature_age_hours", ages),
            ("feature_staleness", stale),
        ),
    )
    augmented = replace(
        dataset,
        dataset_id="0" * 64,
        identity_payload_json=None,
        features=np.concatenate((dataset.features, values), axis=2),
        feature_names=dataset.feature_names + names,
        feature_available=np.concatenate(
            (dataset.feature_available, available), axis=2
        ),
        feature_staleness_hours=np.concatenate(
            (dataset.resolved_array("feature_staleness_hours"), ages), axis=2
        ),
        feature_staleness=np.concatenate(
            (dataset.resolved_array("feature_staleness"), stale), axis=2
        ),
        feature_missing_reason=np.concatenate(
            (
                dataset.resolved_array("feature_missing_reason"),
                (~available).astype(np.int16),
            ),
            axis=2,
        ),
        feature_config_digest=feature_config_digest,
        normalization_digest=normalization_digest,
    )
    # The training-scope contract needs an explicit kind for every feature.
    # Keep the original build config under its unmodified source identity;
    # this is dependency metadata, not a re-serialized MarketBuildConfig.
    dependency_config = {
        "schema_version": "clocked_native_feature_dependencies_v1",
        "features": (*original_specs, *(spec.canonical_payload() for spec in specs)),
        "cross_asset_reference_symbol": config.get("cross_asset_reference_symbol"),
    }
    return augmented.with_content_identity(
        {
            "config": dependency_config,
            "source_dataset": {
                "dataset_id": dataset.dataset_id,
                "identity_payload": source_identity,
            },
            "clocked_native_overlay": {
                **definition,
                "group_source_digests": tuple(
                    tuple(
                        {
                            "symbol": source.contract.symbol,
                            "sha256": source.expected_source_digest,
                        }
                        for source in group.sources
                    )
                    for group in groups
                ),
            },
        }
    )


__all__ = [
    "ClockedNativeFeatureGroup",
    "ClockedNativeSource",
    "append_clocked_native_feature_groups",
    "append_clocked_native_features",
    "clocked_native_source_digest",
]
