from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, TypeVar

from trade_rl.artifacts.hashing import content_digest

UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES = (
    "listing_age_log_days",
    "trailing_30d_quote_notional_log",
    "tick_to_mark_ratio",
    "lot_notional_to_equity",
    "minimum_notional_to_equity",
    "fee_rate",
    "spread_rate",
    "impact_rate",
    "max_participation_rate",
)

_CROSS_ASSET_TOKENS = (
    "relative_return_to_btc",
    "rolling_correlation_to_btc",
    "rolling_beta_to_btc",
    "cross_sectional_momentum_rank",
    "cross_asset_dispersion",
)


class NamedFeature(Protocol):
    name: str


TFeature = TypeVar("TFeature", bound=NamedFeature)


def universal_target_local_features(features: Iterable[TFeature]) -> tuple[TFeature, ...]:
    resolved = tuple(features)
    kept = tuple(
        feature
        for feature in resolved
        if not any(token in feature.name for token in _CROSS_ASSET_TOKENS)
    )
    names = [feature.name for feature in kept]
    if len(names) != len(set(names)):
        raise ValueError("universal feature names must remain unique")
    return kept


def universal_feature_schema_digest(features: Iterable[NamedFeature]) -> str:
    names = tuple(feature.name for feature in features)
    if not names:
        raise ValueError("universal feature schema must not be empty")
    return content_digest(
        {
            "version": "universal_target_local_features_v1",
            "ordered_feature_names": names,
            "instrument_descriptors": UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
        }
    )


@dataclass(frozen=True)
class UniversalObservationContract:
    market_feature_count: int = 206
    account_state_count: int = 7
    instrument_descriptor_count: int = 9
    time_to_go_count: int = 1
    sequence_steps: int = 20

    def validate_market_feature_count(self, count: int) -> None:
        if count != self.market_feature_count:
            raise ValueError(
                f"universal profile requires {self.market_feature_count} market features; got {count}"
            )
