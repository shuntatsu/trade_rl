from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.experiments.bootstrap.config import (
    CanonicalM2BootstrapConfig,
    load_canonical_m2_bootstrap_config,
)


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": "canonical_m2_bootstrap_config_v1",
        "research_question": "Do maintained candidates beat controls?",
        "market": "usds-m",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "base_timeframe": "1h",
        "feature_timeframes": ["4h", "1d"],
        "data_start": "2024-01-01T00:00:00+00:00",
        "data_stop_exclusive": "2025-01-01T00:00:00+00:00",
        "baseline": {
            "signal_name": "1h__log_return_24bar",
            "feature_names": [
                "1h__log_return_24bar",
                "1h__realized_volatility_24bar",
            ],
            "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
            "fit_cutoff": "2024-07-01T00:00:00",
            "evaluation_start": "2024-07-01T00:00:00",
            "evaluation_stop_exclusive": "2025-01-01T00:00:00",
            "rule_entry_threshold": 0.01,
            "rule_exit_threshold": 0.005,
            "forecast_entry_threshold": 0.01,
            "forecast_exit_threshold": 0.005,
            "ppo_total_timesteps": 1_000,
            "gross_budget": 0.5,
            "initial_capital": 100_000.0,
        },
        "ppo_seeds": [0, 1],
        "allowed_factors": ["FEATURE_SET", "FORECAST_THRESHOLDS"],
        "max_experiments": 8,
        "n_bootstrap": 500,
        "bootstrap_seed": 17,
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_config_injects_one_seed_authority_and_normalizes_identity(
    tmp_path: Path,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _valid_payload()))

    assert isinstance(config, CanonicalM2BootstrapConfig)
    assert config.market.value == "usds-m"
    assert config.symbols == ("BTCUSDT", "ETHUSDT")
    assert config.feature_timeframes == ("4h", "1d")
    assert config.ppo_seeds == (0, 1)
    assert config.baseline.ppo_seed == 0
    assert config.allowed_factors == (
        ControlledFactor.FEATURE_SET,
        ControlledFactor.FORECAST_THRESHOLDS,
    )
    payload = config.to_payload()
    assert "ppo_seed" not in payload["baseline"]
    assert payload["data_start"] == "2024-01-01T00:00:00+00:00"
    assert payload["data_stop_exclusive"] == "2025-01-01T00:00:00+00:00"
    assert len(config.digest) == 64


@pytest.mark.parametrize(
    ("location", "key", "value"),
    [
        ("top", "unknown", 1),
        ("baseline", "ppo_seed", 99),
        ("baseline", "unknown", 1),
    ],
)
def test_unknown_keys_are_rejected(
    tmp_path: Path,
    location: str,
    key: str,
    value: object,
) -> None:
    payload = _valid_payload()
    target = payload if location == "top" else payload["baseline"]
    assert isinstance(target, dict)
    target[key] = value

    with pytest.raises(ValueError, match="unknown"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("location", "key"),
    [
        ("top", "research_question"),
        ("top", "symbols"),
        ("top", "ppo_seeds"),
        ("baseline", "signal_name"),
        ("baseline", "fit_cutoff"),
        ("baseline", "gross_budget"),
    ],
)
def test_missing_keys_are_rejected(tmp_path: Path, location: str, key: str) -> None:
    payload = _valid_payload()
    target = payload if location == "top" else payload["baseline"]
    assert isinstance(target, dict)
    del target[key]

    with pytest.raises(ValueError, match="missing"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_non_object_json_and_symlink_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, []))

    real = tmp_path / "real.json"
    real.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="regular file"):
        load_canonical_m2_bootstrap_config(link)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "v2", "schema_version"),
        ("market", "spot", "usds-m"),
        ("symbols", [], "symbols"),
        ("symbols", ["BTCUSDT", "BTCUSDT"], "duplicate"),
        ("feature_timeframes", ["4h", "4h"], "duplicate"),
        ("feature_timeframes", ["1h"], "base timeframe"),
        ("ppo_seeds", [0], "at least two"),
        ("ppo_seeds", [0, 0], "duplicate"),
        ("ppo_seeds", [0, -1], "non-negative"),
        ("allowed_factors", [], "allowed_factors"),
        ("allowed_factors", ["FEATURE_SET", "FEATURE_SET"], "duplicate"),
        ("max_experiments", 0, "max_experiments"),
        ("n_bootstrap", 0, "n_bootstrap"),
        ("bootstrap_seed", -1, "bootstrap_seed"),
    ],
)
def test_top_level_contract_rejects_invalid_values(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _valid_payload()
    payload[field] = value
    with pytest.raises((ValueError, TypeError), match=match):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_unknown_controlled_factor_is_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["allowed_factors"] = ["DO_EVERYTHING"]
    with pytest.raises(ValueError, match="ControlledFactor|controlled factor"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


@pytest.mark.parametrize("field", ["data_start", "data_stop_exclusive"])
def test_source_timestamps_must_be_timezone_aware(tmp_path: Path, field: str) -> None:
    payload = _valid_payload()
    payload[field] = "2024-01-01T00:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_aware_source_timestamp_is_normalized_to_utc(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["data_start"] = "2024-01-01T09:00:00+09:00"
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, payload))
    assert config.to_payload()["data_start"] == "2024-01-01T00:00:00+00:00"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("data_start", "2024-07-01T00:00:00+00:00", "data_start"),
        ("data_stop_exclusive", "2024-06-01T00:00:00+00:00", "evaluation_stop"),
        ("data_stop_exclusive", "2024-12-31T00:00:00+00:00", "month boundary"),
        ("data_start", "2024-01-01T01:00:00+00:00", "native clock"),
    ],
)
def test_source_range_contract_is_fail_closed(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    payload = _valid_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=match):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("fit_cutoff", "2023-12-31T23:00:00", "fit_cutoff"),
        ("evaluation_start", "2024-06-30T23:00:00", "evaluation_start"),
        ("evaluation_stop_exclusive", "2024-06-30T23:00:00", "evaluation_stop"),
        ("evaluation_stop_exclusive", "2025-01-02T00:00:00", "data_stop"),
    ],
)
def test_baseline_time_order_is_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    payload = _valid_payload()
    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    baseline[field] = value
    with pytest.raises(ValueError, match=match):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_fit_symbols_must_be_subset_of_dataset_symbols(tmp_path: Path) -> None:
    payload = _valid_payload()
    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    baseline["fit_symbol_names"] = ["BTCUSDT", "SOLUSDT"]
    with pytest.raises(ValueError, match="fit_symbol_names.*symbols"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_digest_is_key_order_and_whitespace_independent(tmp_path: Path) -> None:
    payload = _valid_payload()
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path_b.write_text(
        json.dumps(dict(reversed(list(payload.items()))), separators=(",", ":")),
        encoding="utf-8",
    )

    assert load_canonical_m2_bootstrap_config(path_a).digest == (
        load_canonical_m2_bootstrap_config(path_b).digest
    )


def test_digest_changes_on_semantic_change(tmp_path: Path) -> None:
    original = _valid_payload()
    changed = deepcopy(original)
    changed["research_question"] = "A different pre-registered question"
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(original), encoding="utf-8")
    path_b.write_text(json.dumps(changed), encoding="utf-8")

    assert load_canonical_m2_bootstrap_config(path_a).digest != (
        load_canonical_m2_bootstrap_config(path_b).digest
    )


def test_baseline_timestamps_remain_existing_candidate_run_semantics(
    tmp_path: Path,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _valid_payload()))
    assert config.baseline.fit_cutoff == np.datetime64("2024-07-01T00:00:00", "ns")
    assert config.baseline.evaluation_start == np.datetime64(
        "2024-07-01T00:00:00", "ns"
    )
