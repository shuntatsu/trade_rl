"""Strict pre-registration contract for canonical M2 study bootstrap."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.build import ExecutionEconomicsProfile
from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.runs import (
    CandidateRunConfig,
    parse_candidate_run_config,
)
from trade_rl.integrations.binance import (
    BinanceMarket,
    binance_interval_milliseconds,
)

_SCHEMA_VERSION_V1 = "canonical_m2_bootstrap_config_v1"
_SCHEMA_VERSION_V2 = "canonical_m2_bootstrap_config_v2"
_TOP_LEVEL_KEYS_V1 = frozenset(
    {
        "schema_version",
        "research_question",
        "market",
        "symbols",
        "base_timeframe",
        "feature_timeframes",
        "data_start",
        "data_stop_exclusive",
        "baseline",
        "ppo_seeds",
        "allowed_factors",
        "max_experiments",
        "n_bootstrap",
        "bootstrap_seed",
    }
)
_TOP_LEVEL_KEYS_V2 = frozenset((*_TOP_LEVEL_KEYS_V1, "execution_economics"))
_BASELINE_FIELDS = (
    "signal_name",
    "feature_names",
    "fit_symbol_names",
    "fit_cutoff",
    "evaluation_start",
    "evaluation_stop_exclusive",
    "rule_entry_threshold",
    "rule_exit_threshold",
    "forecast_entry_threshold",
    "forecast_exit_threshold",
    "ppo_total_timesteps",
    "gross_budget",
    "initial_capital",
)


def _expect_exact_keys(
    raw: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    keys = set(raw)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        raise ValueError(
            f"{field} keys differ from contract: missing={missing}, unknown={unknown}"
        )


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return cast(dict[str, object], value)


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_sequence(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a string array")
    values = tuple(_require_text(item, field=field) for item in value)
    if not allow_empty and not values:
        raise ValueError(f"{field} must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicate values")
    return values


def _string_tuple(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be a string tuple")
    values = tuple(_require_text(item, field=field) for item in value)
    if not allow_empty and not values:
        raise ValueError(f"{field} must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicate values")
    return values


def _int_value(value: object, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _seed_policy(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("ppo_seeds must be an integer array")
    seeds = tuple(_int_value(item, field="ppo_seeds") for item in value)
    if len(seeds) < 2:
        raise ValueError("ppo_seeds must contain at least two seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("ppo_seeds must not contain duplicate values")
    return seeds


def _normalize_source_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_source_datetime(value: object, *, field: str) -> datetime:
    raw = _require_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO datetime") from error
    return _normalize_source_datetime(parsed, field=field)


def _utc_np(value: datetime) -> np.datetime64:
    naive = value.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(naive, "ns")


def _require_native_alignment(
    value: datetime,
    *,
    field: str,
    timeframes: Sequence[str],
) -> None:
    if value.microsecond != 0:
        raise ValueError(f"{field} must align to every native clock")
    epoch_ms = int(value.timestamp()) * 1_000
    for timeframe in timeframes:
        interval_ms = binance_interval_milliseconds(timeframe)
        if epoch_ms % interval_ms != 0:
            raise ValueError(
                f"{field} must align to every native clock; failed {timeframe}"
            )


def _baseline_payload(config: CandidateRunConfig) -> dict[str, object]:
    candidate_payload = config.to_json_payload()
    return {field: candidate_payload[field] for field in _BASELINE_FIELDS}


def _validate_time_contract(
    *,
    data_start: datetime,
    data_stop_exclusive: datetime,
    baseline: CandidateRunConfig,
) -> None:
    if not (
        data_stop_exclusive.day == 1
        and data_stop_exclusive.hour == 0
        and data_stop_exclusive.minute == 0
        and data_stop_exclusive.second == 0
        and data_stop_exclusive.microsecond == 0
    ):
        raise ValueError("data_stop_exclusive must be a UTC month boundary")

    start = _utc_np(data_start)
    stop = _utc_np(data_stop_exclusive)
    if not start < baseline.fit_cutoff:
        raise ValueError("data_start must be before fit_cutoff")
    if baseline.fit_cutoff > baseline.evaluation_start:
        raise ValueError("fit_cutoff must not be after evaluation_start")
    if not baseline.evaluation_start < baseline.evaluation_stop_exclusive:
        raise ValueError("evaluation_start must be before evaluation_stop_exclusive")
    if baseline.evaluation_stop_exclusive > stop:
        raise ValueError(
            "evaluation_stop_exclusive must not exceed data_stop_exclusive"
        )


@dataclass(frozen=True, slots=True)
class CanonicalM2BootstrapConfig:
    """Immutable, normalized pre-registration for one canonical M2 bootstrap."""

    research_question: str
    market: BinanceMarket
    symbols: tuple[str, ...]
    base_timeframe: str
    feature_timeframes: tuple[str, ...]
    data_start: datetime
    data_stop_exclusive: datetime
    baseline: CandidateRunConfig
    ppo_seeds: tuple[int, ...]
    allowed_factors: tuple[ControlledFactor, ...]
    max_experiments: int
    n_bootstrap: int
    bootstrap_seed: int
    execution_economics: ExecutionEconomicsProfile | None = None
    schema_version: str = _SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version not in {_SCHEMA_VERSION_V1, _SCHEMA_VERSION_V2}:
            raise ValueError("schema_version does not match canonical M2 contract")
        if self.schema_version == _SCHEMA_VERSION_V1:
            if self.execution_economics is not None:
                raise ValueError(
                    "v1 bootstrap config must not include execution_economics"
                )
        elif self.execution_economics is None:
            raise ValueError("v2 bootstrap config requires execution_economics")
        if self.execution_economics is not None and not isinstance(
            self.execution_economics, ExecutionEconomicsProfile
        ):
            raise ValueError("execution_economics must be an ExecutionEconomicsProfile")
        if self.market is not BinanceMarket.USDS_M:
            raise ValueError("canonical M2 bootstrap supports only usds-m")
        if not isinstance(self.baseline, CandidateRunConfig):
            raise ValueError("baseline must be a CandidateRunConfig")

        research_question = _require_text(
            self.research_question,
            field="research_question",
        )
        symbols = _string_tuple(self.symbols, field="symbols")
        base_timeframe = _require_text(self.base_timeframe, field="base_timeframe")
        binance_interval_milliseconds(base_timeframe)
        feature_timeframes = _string_tuple(
            self.feature_timeframes,
            field="feature_timeframes",
            allow_empty=True,
        )
        if base_timeframe in feature_timeframes:
            raise ValueError("feature_timeframes must not repeat the base timeframe")
        for timeframe in feature_timeframes:
            binance_interval_milliseconds(timeframe)

        data_start = _normalize_source_datetime(self.data_start, field="data_start")
        data_stop = _normalize_source_datetime(
            self.data_stop_exclusive,
            field="data_stop_exclusive",
        )
        _validate_time_contract(
            data_start=data_start,
            data_stop_exclusive=data_stop,
            baseline=self.baseline,
        )
        timeframes = (base_timeframe, *feature_timeframes)
        _require_native_alignment(data_start, field="data_start", timeframes=timeframes)
        _require_native_alignment(
            data_stop,
            field="data_stop_exclusive",
            timeframes=timeframes,
        )

        if not set(self.baseline.fit_symbol_names).issubset(symbols):
            raise ValueError("fit_symbol_names must be a subset of symbols")
        if not isinstance(self.ppo_seeds, tuple):
            raise ValueError("ppo_seeds must be an integer tuple")
        ppo_seeds = tuple(
            _int_value(seed, field="ppo_seeds") for seed in self.ppo_seeds
        )
        if len(ppo_seeds) < 2:
            raise ValueError("ppo_seeds must contain at least two seeds")
        if len(set(ppo_seeds)) != len(ppo_seeds):
            raise ValueError("ppo_seeds must not contain duplicate values")
        if self.baseline.ppo_seed != ppo_seeds[0]:
            raise ValueError("baseline ppo_seed must equal first ppo_seeds value")

        if not isinstance(self.allowed_factors, tuple) or not self.allowed_factors:
            raise ValueError("allowed_factors must be a non-empty tuple")
        if any(
            not isinstance(factor, ControlledFactor) for factor in self.allowed_factors
        ):
            raise ValueError("allowed_factors must contain ControlledFactor values")
        allowed_factors = tuple(self.allowed_factors)
        if len(set(allowed_factors)) != len(allowed_factors):
            raise ValueError("allowed_factors must not contain duplicate values")

        max_experiments = _int_value(
            self.max_experiments,
            field="max_experiments",
            positive=True,
        )
        n_bootstrap = _int_value(
            self.n_bootstrap,
            field="n_bootstrap",
            positive=True,
        )
        bootstrap_seed = _int_value(self.bootstrap_seed, field="bootstrap_seed")

        object.__setattr__(self, "research_question", research_question)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "base_timeframe", base_timeframe)
        object.__setattr__(self, "feature_timeframes", feature_timeframes)
        object.__setattr__(self, "data_start", data_start)
        object.__setattr__(self, "data_stop_exclusive", data_stop)
        object.__setattr__(self, "ppo_seeds", ppo_seeds)
        object.__setattr__(self, "allowed_factors", allowed_factors)
        object.__setattr__(self, "max_experiments", max_experiments)
        object.__setattr__(self, "n_bootstrap", n_bootstrap)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "research_question": self.research_question,
            "market": self.market.value,
            "symbols": list(self.symbols),
            "base_timeframe": self.base_timeframe,
            "feature_timeframes": list(self.feature_timeframes),
            "data_start": self.data_start.astimezone(UTC).isoformat(),
            "data_stop_exclusive": self.data_stop_exclusive.astimezone(UTC).isoformat(),
            "baseline": _baseline_payload(self.baseline),
            "ppo_seeds": list(self.ppo_seeds),
            "allowed_factors": [factor.value for factor in self.allowed_factors],
            "max_experiments": self.max_experiments,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
        }
        if self.execution_economics is not None:
            payload["execution_economics"] = self.execution_economics.to_payload()
        return payload

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _parse_config(raw: Mapping[str, object]) -> CanonicalM2BootstrapConfig:
    schema_version = _require_text(raw.get("schema_version"), field="schema_version")
    if schema_version == _SCHEMA_VERSION_V1:
        _expect_exact_keys(raw, _TOP_LEVEL_KEYS_V1, field="bootstrap config")
        execution_economics = None
    elif schema_version == _SCHEMA_VERSION_V2:
        _expect_exact_keys(raw, _TOP_LEVEL_KEYS_V2, field="bootstrap config")
        execution_economics = ExecutionEconomicsProfile.from_payload(
            raw.get("execution_economics"),
            field="execution_economics",
        )
    else:
        raise ValueError("schema_version does not match canonical M2 contract")

    market_text = _require_text(raw.get("market"), field="market")
    if market_text != BinanceMarket.USDS_M.value:
        raise ValueError("canonical M2 bootstrap supports only usds-m")
    market = BinanceMarket.USDS_M

    symbols = _string_sequence(raw.get("symbols"), field="symbols")
    base_timeframe = _require_text(raw.get("base_timeframe"), field="base_timeframe")
    binance_interval_milliseconds(base_timeframe)
    feature_timeframes = _string_sequence(
        raw.get("feature_timeframes"),
        field="feature_timeframes",
        allow_empty=True,
    )
    if base_timeframe in feature_timeframes:
        raise ValueError("feature_timeframes must not repeat the base timeframe")
    for timeframe in feature_timeframes:
        binance_interval_milliseconds(timeframe)

    data_start = _parse_source_datetime(raw.get("data_start"), field="data_start")
    data_stop = _parse_source_datetime(
        raw.get("data_stop_exclusive"), field="data_stop_exclusive"
    )
    timeframes = (base_timeframe, *feature_timeframes)
    _require_native_alignment(data_start, field="data_start", timeframes=timeframes)
    _require_native_alignment(
        data_stop,
        field="data_stop_exclusive",
        timeframes=timeframes,
    )

    ppo_seeds = _seed_policy(raw.get("ppo_seeds"))
    baseline_raw = _require_mapping(raw.get("baseline"), field="baseline")
    _expect_exact_keys(baseline_raw, frozenset(_BASELINE_FIELDS), field="baseline")
    candidate_payload = dict(baseline_raw)
    candidate_payload["ppo_seed"] = ppo_seeds[0]
    baseline = parse_candidate_run_config(candidate_payload)
    if not set(baseline.fit_symbol_names).issubset(symbols):
        raise ValueError("fit_symbol_names must be a subset of symbols")

    factor_names = _string_sequence(raw.get("allowed_factors"), field="allowed_factors")
    try:
        allowed_factors = tuple(ControlledFactor(item) for item in factor_names)
    except ValueError as error:
        raise ValueError("controlled factor is not a valid ControlledFactor") from error

    return CanonicalM2BootstrapConfig(
        research_question=_require_text(
            raw.get("research_question"), field="research_question"
        ),
        market=market,
        symbols=symbols,
        base_timeframe=base_timeframe,
        feature_timeframes=feature_timeframes,
        data_start=data_start,
        data_stop_exclusive=data_stop,
        baseline=baseline,
        ppo_seeds=ppo_seeds,
        allowed_factors=allowed_factors,
        max_experiments=_int_value(
            raw.get("max_experiments"), field="max_experiments", positive=True
        ),
        n_bootstrap=_int_value(
            raw.get("n_bootstrap"), field="n_bootstrap", positive=True
        ),
        bootstrap_seed=_int_value(raw.get("bootstrap_seed"), field="bootstrap_seed"),
        execution_economics=execution_economics,
        schema_version=schema_version,
    )


def load_canonical_m2_bootstrap_config(
    path: str | Path,
) -> CanonicalM2BootstrapConfig:
    """Load one strict, regular-file bootstrap JSON contract."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("bootstrap config must be a regular file")
    try:
        decoded = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("bootstrap config is not valid JSON") from error
    raw = _require_mapping(decoded, field="bootstrap config")
    return _parse_config(raw)


__all__ = ["CanonicalM2BootstrapConfig", "load_canonical_m2_bootstrap_config"]
