"""Strict JSON configuration for maintained market dataset builds."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    NormalizationMode,
    VolumeUnit,
)


@dataclass(frozen=True, slots=True)
class MarketDatasetBuildRequest:
    source_root: Path
    config: MarketBuildConfig
    instruments: tuple[InstrumentContract, ...]


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, *, field: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _aware_datetime(value: object, *, field: str) -> datetime:
    raw = _string(value, field=field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _reject_unknown(
    value: Mapping[str, object],
    *,
    allowed: set[str],
    field: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {unknown}")


def _feature(value: object, *, index: int) -> FeatureSpec:
    field = f"features[{index}]"
    item = _mapping(value, field=field)
    _reject_unknown(
        item,
        allowed={
            "name",
            "kind",
            "lookback",
            "normalization",
            "normalization_window",
            "min_periods",
            "max_staleness_hours",
        },
        field=field,
    )
    try:
        kind = FeatureKind(_string(item.get("kind"), field=f"{field}.kind"))
        normalization = NormalizationMode(
            _optional_string(
                item.get("normalization"),
                field=f"{field}.normalization",
            )
            or NormalizationMode.NONE.value
        )
    except ValueError as exc:
        raise ValueError(f"{field} contains an unsupported enum value") from exc
    return FeatureSpec(
        name=_string(item.get("name"), field=f"{field}.name"),
        kind=kind,
        lookback=_integer(item.get("lookback"), field=f"{field}.lookback", default=1),
        normalization=normalization,
        normalization_window=_integer(
            item.get("normalization_window"),
            field=f"{field}.normalization_window",
            default=1,
        ),
        min_periods=_integer(
            item.get("min_periods"),
            field=f"{field}.min_periods",
            default=1,
        ),
        max_staleness_hours=_number(
            item.get("max_staleness_hours"),
            field=f"{field}.max_staleness_hours",
            default=24.0,
        ),
    )


def _instrument(value: object, *, index: int) -> InstrumentContract:
    field = f"instruments[{index}]"
    item = _mapping(value, field=field)
    _reject_unknown(
        item,
        allowed={
            "symbol",
            "listed_at",
            "delisted_at",
            "volume_unit",
            "contract_multiplier",
        },
        field=field,
    )
    listed_at = _aware_datetime(item.get("listed_at"), field=f"{field}.listed_at")
    raw_delisted = item.get("delisted_at")
    delisted_at = (
        None
        if raw_delisted is None
        else _aware_datetime(raw_delisted, field=f"{field}.delisted_at")
    )
    try:
        volume_unit = VolumeUnit(
            _optional_string(item.get("volume_unit"), field=f"{field}.volume_unit")
            or VolumeUnit.BASE_ASSET.value
        )
    except ValueError as exc:
        raise ValueError(f"{field}.volume_unit is unsupported") from exc
    return InstrumentContract(
        symbol=_string(item.get("symbol"), field=f"{field}.symbol"),
        listed_at=listed_at,
        delisted_at=delisted_at,
        volume_unit=volume_unit,
        contract_multiplier=_number(
            item.get("contract_multiplier"),
            field=f"{field}.contract_multiplier",
            default=1.0,
        ),
    )


def load_market_build_request(path: str | Path) -> MarketDatasetBuildRequest:
    """Load and validate one JSON build request without permissive coercions."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    root = _mapping(payload, field="market build config")
    _reject_unknown(
        root,
        allowed={
            "source_root",
            "base_timeframe",
            "calendar_kind",
            "session_periods_per_year",
            "features",
            "instruments",
        },
        field="market build config",
    )
    features = tuple(
        _feature(value, index=index)
        for index, value in enumerate(_list(root.get("features"), field="features"))
    )
    instruments = tuple(
        _instrument(value, index=index)
        for index, value in enumerate(
            _list(root.get("instruments"), field="instruments")
        )
    )
    source_root = Path(
        _string(root.get("source_root"), field="source_root")
    ).expanduser()
    if not source_root.is_absolute():
        source_root = config_path.parent / source_root
    return MarketDatasetBuildRequest(
        source_root=source_root.resolve(),
        config=MarketBuildConfig(
            base_timeframe=_string(
                root.get("base_timeframe"),
                field="base_timeframe",
            ),
            features=features,
            calendar_kind=(
                _optional_string(root.get("calendar_kind"), field="calendar_kind")
                or "continuous_24_7"
            ),
            session_periods_per_year=(
                None
                if root.get("session_periods_per_year") is None
                else _integer(
                    root.get("session_periods_per_year"),
                    field="session_periods_per_year",
                    default=0,
                )
            ),
        ),
        instruments=instruments,
    )
