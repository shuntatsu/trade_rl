"""Frozen artificial measurement scope must not inherit real-data identity."""

import importlib
import json
from pathlib import Path

import pytest

from tools.tier_d_scaling.fixture import write_full_artificial_dataset


def test_full_plan_and_all_measurement_cases_are_declared_together(tmp_path: Path):
    module = importlib.import_module("tools.tier_d_scaling.spec")
    root = tmp_path / "dataset"
    dataset = write_full_artificial_dataset(root)
    protocol = module.profile_protocol(dataset, root)
    serialized = json.dumps(protocol)
    assert all(name not in serialized for name in ("BTCUSDT", "ETHUSDT", "Task18"))
    assert protocol["dataset"]["dimensions"] == [9504, 2, 6]
    assert protocol["dataset"]["dataset_id"] == dataset.dataset_id
    assert protocol["actor"]["vintage_starts"] == [2903, 5087, 7295]
    assert protocol["actor"]["stop"] == 9430
    assert protocol["terminal"]["stop_inclusive"] == 9502
    assert protocol["budget"]["advances_per_arm"] == 6599
    assert protocol["scenarios"] == ["BASEL0"]
    assert len(protocol["configurations"]) == 1
    cases = protocol["measurement"]["cases"]
    assert [(x["name"], x["target"], x["profile"]) for x in cases] == [
        ("n100", 100, False),
        ("n400", 400, False),
        ("n1600", 1600, False),
        ("n1600-profile", 1600, True),
    ]
    assert protocol["measurement"]["financial_authority"] is False
    assert protocol["measurement"]["limits"]["wall_seconds"] == 1200
    assert protocol["measurement"]["limits"]["rss_bytes"] == 2 * 1024**3
    assert module.case_by_name(protocol, "n100") == cases[0]
    for name in ("../n100", "n25", "N100", True):
        with pytest.raises(ValueError):
            module.case_by_name(protocol, name)


def test_wrong_dataset_shape_is_rejected_before_any_measurement(tmp_path: Path):
    from tools.tier_d_nonrl.tier_d_fixture import write_artificial_dataset

    module = importlib.import_module("tools.tier_d_scaling.spec")
    dataset = write_artificial_dataset(tmp_path / "tiny")
    with pytest.raises(ValueError, match="full artificial"):
        module.profile_protocol(dataset, tmp_path / "tiny")
