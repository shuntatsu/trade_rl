from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    parse_candidate_run_config,
    resolve_candidate_run_spec,
)


def raw_config() -> dict[str, object]:
    return {
        "signal_name": "signal",
        "feature_names": ["signal", "f1"],
        "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
        "fit_cutoff": "2026-01-01T04:00:00",
        "evaluation_start": "2026-01-01T04:00:00",
        "evaluation_stop_exclusive": "2026-01-01T07:00:00",
        "rule_entry_threshold": 0.10,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 7,
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
    }


def market() -> MarketDataset:
    n = 8
    close = np.full((n, 2), 100.0, dtype=np.float64)
    features = np.zeros((n, 2, 2), dtype=np.float32)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 2), 1_000_000.0),
        funding_rate=np.zeros((n, 2), dtype=np.float64),
        tradable=np.ones((n, 2), dtype=np.bool_),
        feature_available=np.ones((n, 2, 2), dtype=np.bool_),
        feature_names=("signal", "f1"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_parse_candidate_run_config_returns_frozen_semantic_config() -> None:
    config = parse_candidate_run_config(raw_config())

    assert isinstance(config, CandidateRunConfig)
    assert config.feature_names == ("signal", "f1")
    assert config.fit_symbol_names == ("BTCUSDT", "ETHUSDT")
    assert config.ppo_seed == 7
    assert config.fit_cutoff == np.datetime64("2026-01-01T04:00:00", "ns")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.__setitem__("hidden_override", 1), "unknown config keys"),
        (
            lambda raw: raw.__setitem__("feature_names", ["signal", "signal"]),
            "feature_names must not contain duplicates",
        ),
        (
            lambda raw: raw.__setitem__("fit_symbol_names", ["BTCUSDT", "BTCUSDT"]),
            "fit_symbol_names must not contain duplicates",
        ),
        (
            lambda raw: raw.__setitem__("rule_exit_threshold", 0.10),
            "rule exit threshold must be below entry threshold",
        ),
        (
            lambda raw: raw.__setitem__("gross_budget", math.inf),
            "gross_budget must be a finite number",
        ),
        (
            lambda raw: raw.__setitem__("fit_cutoff", "not-a-time"),
            "fit_cutoff must be an ISO datetime",
        ),
        (
            lambda raw: raw.__setitem__("ppo_total_timesteps", 0),
            "ppo_total_timesteps must be a positive integer",
        ),
        (
            lambda raw: raw.__setitem__("ppo_seed", -1),
            "ppo_seed must be a non-negative integer",
        ),
    ],
)
def test_parse_candidate_run_config_rejects_invalid_contract(
    mutate,
    message: str,
) -> None:
    raw = raw_config()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        parse_candidate_run_config(raw)


def test_resolve_candidate_run_spec_binds_dataset_identity_and_indices() -> None:
    dataset = market()
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="d" * 64,
        config=parse_candidate_run_config(raw_config()),
    )

    assert spec.dataset_id == dataset.dataset_id
    assert spec.dataset_artifact_schema == "market_dataset_artifact_v3"
    assert spec.dataset_artifact_digest == "d" * 64
    assert spec.lean_config.signal_index == 0
    assert spec.lean_config.feature_indices == (0, 1)
    assert spec.lean_config.fit_symbol_indices == (0, 1)
    assert spec.evaluation_start_index == 4
    assert spec.evaluation_stop_index == 7


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_names", ["UNKNOWN"], "unknown feature name: UNKNOWN"),
        ("fit_symbol_names", ["UNKNOWN"], "unknown fit symbol name: UNKNOWN"),
        (
            "evaluation_start",
            "2026-01-01T04:30:00",
            "evaluation_start must exactly match one dataset timestamp",
        ),
        (
            "evaluation_stop_exclusive",
            "2026-01-01T07:30:00",
            "evaluation_stop_exclusive must exactly match one dataset timestamp",
        ),
        (
            "evaluation_start",
            "2026-01-01T03:00:00",
            "evaluation must not start before fit_cutoff",
        ),
    ],
)
def test_resolve_candidate_run_spec_rejects_invalid_dataset_resolution(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = raw_config()
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        resolve_candidate_run_spec(
            market(),
            dataset_artifact_schema="market_dataset_artifact_v3",
            dataset_artifact_digest="d" * 64,
            config=parse_candidate_run_config(raw),
        )
