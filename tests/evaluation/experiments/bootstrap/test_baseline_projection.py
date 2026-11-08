from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_config import _valid_payload, _write
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)


def test_direct_bootstrap_canonical_baseline_round_trips_identity(
    tmp_path: Path,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _valid_payload()))

    equivalent = replace(config, baseline=replace(config.baseline))

    assert equivalent.to_payload() == config.to_payload()
    assert equivalent.digest == config.digest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("supervised_forecast_horizon_hours", 48),
        ("forecast_switch_cost", 0.0007),
        ("ppo_settle_terminal_position", True),
    ],
)
def test_direct_bootstrap_rejects_baseline_semantics_outside_digest_projection(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _valid_payload()))
    changed_baseline = replace(config.baseline, **{field: value})
    changed_payload = changed_baseline.to_json_payload()
    canonical_payload = config.baseline.to_json_payload()
    assert changed_payload[field] != canonical_payload[field]

    with pytest.raises(ValueError, match="baseline.*(projection|canonical)"):
        replace(config, baseline=changed_baseline)


@pytest.mark.parametrize(
    "field",
    ("rule_exit_threshold", "forecast_exit_threshold"),
)
def test_direct_bootstrap_rejects_noncanonical_numeric_baseline_encoding(
    tmp_path: Path,
    field: str,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _valid_payload()))
    changed_baseline = replace(config.baseline, **{field: 0})
    assert type(changed_baseline.to_json_payload()[field]) is int

    # Python value equality treats 0 and 0.0 as identical, but canonical
    # JSON bytes (and therefore the persisted digest) distinguish them.
    with pytest.raises(ValueError, match="baseline.*(projection|canonical)"):
        replace(config, baseline=changed_baseline)


@pytest.mark.parametrize(
    "field",
    ("rule_exit_threshold", "forecast_exit_threshold"),
)
def test_direct_bootstrap_accepts_canonical_float_zero_round_trip(
    tmp_path: Path,
    field: str,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _valid_payload()))
    canonical = replace(config, baseline=replace(config.baseline, **{field: 0.0}))
    reloaded = load_canonical_m2_bootstrap_config(
        _write(tmp_path, canonical.to_payload())
    )
    assert canonical.to_payload() == reloaded.to_payload()
    assert canonical.digest == reloaded.digest
