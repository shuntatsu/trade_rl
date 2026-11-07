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
