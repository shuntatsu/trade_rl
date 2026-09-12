from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from trade_rl.evaluation.experiments.bootstrap import config as bootstrap_config
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    parse_candidate_run_config,
)

ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG_PATH = ROOT / "trade_rl" / "evaluation" / "runs" / "config.py"

BASELINE_FIELDS = (
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


def _raw_config() -> dict[str, object]:
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


def test_candidate_run_config_owns_explicit_json_field_roster() -> None:
    dataclass_fields = tuple(field.name for field in fields(CandidateRunConfig))

    assert CandidateRunConfig.JSON_FIELDS == dataclass_fields


def test_candidate_run_config_owns_normalized_raw_json_payload() -> None:
    config = parse_candidate_run_config(_raw_config())

    payload = config.to_json_payload()

    assert payload == {
        "signal_name": "signal",
        "feature_names": ["signal", "f1"],
        "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
        "fit_cutoff": "2026-01-01T04:00:00.000000000",
        "evaluation_start": "2026-01-01T04:00:00.000000000",
        "evaluation_stop_exclusive": "2026-01-01T07:00:00.000000000",
        "rule_entry_threshold": 0.10,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 7,
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
    }
    assert parse_candidate_run_config(payload) == config


def test_candidate_run_parser_has_no_second_allowed_key_roster() -> None:
    source = RUN_CONFIG_PATH.read_text(encoding="utf-8")

    assert "_ALLOWED_CONFIG_KEYS" not in source


def test_bootstrap_v1_projects_candidate_payload_through_fixed_schema() -> None:
    config = parse_candidate_run_config(_raw_config())

    assert bootstrap_config._BASELINE_FIELDS == BASELINE_FIELDS
    assert "ppo_seed" not in bootstrap_config._BASELINE_FIELDS
    assert bootstrap_config._baseline_payload(config) == {
        field: config.to_json_payload()[field] for field in BASELINE_FIELDS
    }
