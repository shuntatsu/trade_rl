import json
from pathlib import Path

import pytest

from trade_rl.evaluation import directional_study


def test_protocol_is_written_before_execution_and_cannot_be_replaced(
    tmp_path: Path,
) -> None:
    assert hasattr(directional_study, "reserve_study")
    root = tmp_path / "study"
    directional_study.reserve_study(
        root, {"maximum_drawdown": 0.2, "roster": ["breakout"]}
    )
    assert json.loads((root / "protocol.json").read_text())["maximum_drawdown"] == 0.2
    with pytest.raises(FileExistsError):
        directional_study.reserve_study(root, {"maximum_drawdown": 0.9})
    assert json.loads((root / "protocol.json").read_text())["maximum_drawdown"] == 0.2


def test_unknown_arm_is_rejected_before_any_training() -> None:
    with pytest.raises(ValueError, match="arm"):
        directional_study.validate_arm("ppo-seed-99")


def test_changed_stress_protocol_is_rejected_even_if_hash_is_rewritten(
    tmp_path: Path,
) -> None:
    from trade_rl.artifacts import canonical_json_bytes, content_digest

    expected = {"stresses": [{"cost_multiplier": 2.0}]}
    directional_study.reserve_study(tmp_path / "study", expected)
    changed = {"stresses": []}
    (tmp_path / "study" / "protocol.json").write_bytes(canonical_json_bytes(changed))
    (tmp_path / "study" / "protocol.digest.json").write_bytes(
        canonical_json_bytes({"digest": content_digest(changed)})
    )
    with pytest.raises(ValueError, match="protocol"):
        directional_study.validate_protocol(tmp_path / "study", expected)


def test_development_clock_requires_complete_exact_calendar_years() -> None:
    from dataclasses import replace

    import numpy as np

    from tests.evaluation.test_shared_cash_replay import _market

    market = _market(np.full((17546, 1), 100.0))
    clock = np.datetime64("2022-12-31T23", "ns") + np.arange(17546) * np.timedelta64(
        1, "h"
    )
    market = replace(market, timestamps=clock, available_at=clock[:, None])
    assert directional_study.development_indices(market) == (1, 17545)
    incomplete = replace(
        market,
        timestamps=clock + np.timedelta64(2, "h"),
        available_at=(clock + np.timedelta64(2, "h"))[:, None],
    )
    with pytest.raises(ValueError, match="clock"):
        directional_study.development_indices(incomplete)
