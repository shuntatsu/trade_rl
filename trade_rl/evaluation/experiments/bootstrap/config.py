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
from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    parse_candidate_run_config,
)
from trade_rl.integrations.binance import (
    BinanceMarket,
    binance_interval_milliseconds,
)

_SCHEMA_VERSION = "canonical_m2_bootstrap_config_v1"
_TOP_LEVEL_KEYS = frozenset(
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
_BASELINE_KEYS = frozenset(
    {
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
    }
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


def _parse_source_datetime(value: object, *, field: str) -> datetime:
    raw = _require_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


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
    return {
        "signal_name": config.signal_name,
        "feature_names": list(config.feature_names),
        "fit_symbol_names": list(config.fit_symbol_names),
        "fit_cutoff": str(np.datetime64(config.fit_cutoff, "ns")),
        "evaluation_start": str(np.datetime64(config.evaluation_start, "ns")),
        "evaluation_stop_exclusive": str(
            np.datetime64(config.evaluation_stop_exclusive, "ns")
        ),
        "rule_entry_threshold": config.rule_entry_threshold,
        "rule_exit_threshold": config.rule_exit_threshold,
        "forecast_entry_threshold": config.forecast_entry_threshold,
        "forecast_exit_threshold": config.forecast_exit_threshold,
        "ppo_total_timesteps": config.ppo_total_timesteps,
        "gross_budget": config.gross_budget,
        "initial_capital": config.initial_capital,
    }


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
        raise ValueError("evaluation_stop_exclusive must not exceed data_stop_exclusive")


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
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("schema_version does not match canonical M2 contract")
        if self.market is not BinanceMarket.USDS_M:
            raise ValueError("canonical M2 bootstrap supports only usds-m")
        if not self.research_question.strip():
            raise ValueError("research_question must be non-empty")
        if not self.symbols or any(not item for item in self.symbols):
            raise ValueError("symbols must be non-empty")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must not contain duplicate values")
        if len(set(self.feature_timeframes)) != len(self.feature_timeframes):
            raise ValueError("feature_timeframes must not contain duplicate values")
        if self.base_timeframe in self.feature_timeframes:
            raise ValueError("feature_timeframes must not repeat the base timeframe")
        if not set(self.baseline.fit_symbol_names).issubset(self.symbols):
            raise ValueError("fit_symbol_names must be a subset of symbols")
        if len(self.ppo_seeds) < 2:
            raise ValueError("ppo_seeds must contain at least two seeds")
        if len(set(self.ppo_seeds)) != len(self.ppo_seeds):
            raise ValueError("ppo_seeds must not contain duplicate values")
        if any(seed < 0 for seed in self.ppo_seeds):
            raise ValueError("ppo_seeds must be non-negative")
        if self.baseline.ppo_seed != self.ppo_seeds[0]:
            raise ValueError("baseline ppo_seed must equal first ppo_seeds value")
        if not self.allowed_factors:
            raise ValueError("allowed_factors must be non-empty")
        if len(set(self.allowed_factors)) != len(self.allowed_factors):
            raise ValueError("allowed_factors must not contain duplicate values")
        _int_value(self.max_experiments, field="max_experiments", positive=True)
        _int_value(self.n_bootstrap, field="n_bootstrap", positive=True)
        _int_value(self.bootstrap_seed, field="bootstrap_seed")
        _validate_time_contract(
            data_start=self.data_start,
            data_stop_exclusive=self.data_stop_exclusive,
            baseline=self.baseline,
        )

    def to_payload(self) -> dict[str, object]:
        return {
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

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _parse_config(raw: Mapping[str, object]) -> CanonicalM2BootstrapConfig:
    _expect_exact_keys(raw, _TOP_LEVEL_KEYS, field="bootstrap config")
    if raw.get("schema_version") != _SCHEMA_VERSION:
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
    _expect_exact_keys(baseline_raw, _BASELINE_KEYS, field="baseline")
    candidate_payload = dict(baseline_raw)
    candidate_payload["ppo_seed"] = ppo_seeds[0]
    baseline = parse_candidate_run_config(candidate_payload)
    if not set(baseline.fit_symbol_names).issubset(symbols):
        raise ValueError("fit_symbol_names must be a subset of symbols")

    factor_names = _string_sequence(
        raw.get("allowed_factors"), field="allowed_factors"
    )
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
