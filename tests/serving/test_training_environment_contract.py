from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from trade_rl.serving import package as serving_package
from trade_rl.simulation.execution import ExecutionCostConfig

ROOT = Path(__file__).resolve().parents[2]


def _payload() -> dict[str, object]:
    return {
        "environment": {"execution_cost": asdict(ExecutionCostConfig())},
        "schema_version": "training_environment_v2",
    }


def _write_environment(root: Path, payload: object) -> None:
    root.mkdir()
    (root / "environment.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_execution_cost_round_trips_complete_training_artifact(tmp_path: Path) -> None:
    root = tmp_path / "training"
    expected = ExecutionCostConfig(
        fee_rate=0.0007,
        maker_fee_rate=0.0001,
        taker_fee_rate=0.0008,
        order_latency_bars=2,
        order_type="limit",
        trigger_volume_fractions=(0.9, 0.6, 0.3, 0.1),
    )
    payload = _payload()
    environment = payload["environment"]
    assert isinstance(environment, dict)
    environment["execution_cost"] = asdict(expected)
    _write_environment(root, payload)

    assert serving_package._execution_cost(root) == expected


def test_execution_cost_rejects_unsupported_training_environment_schema(
    tmp_path: Path,
) -> None:
    root = tmp_path / "training"
    payload = _payload()
    payload["schema_version"] = "training_environment_v1"
    _write_environment(root, payload)

    with pytest.raises(ValueError, match="schema"):
        serving_package._execution_cost(root)


def test_execution_cost_rejects_missing_canonical_field(tmp_path: Path) -> None:
    root = tmp_path / "training"
    payload = _payload()
    environment = payload["environment"]
    assert isinstance(environment, dict)
    execution = environment["execution_cost"]
    assert isinstance(execution, dict)
    execution.pop("path_mode")
    _write_environment(root, payload)

    with pytest.raises(ValueError, match="missing.*path_mode"):
        serving_package._execution_cost(root)


def test_execution_cost_rejects_unknown_canonical_field(tmp_path: Path) -> None:
    root = tmp_path / "training"
    payload = _payload()
    environment = payload["environment"]
    assert isinstance(environment, dict)
    execution = environment["execution_cost"]
    assert isinstance(execution, dict)
    execution["future_default"] = 1
    _write_environment(root, payload)

    with pytest.raises(ValueError, match="unknown.*future_default"):
        serving_package._execution_cost(root)


def test_execution_cost_rejects_non_sequence_trigger_fractions(tmp_path: Path) -> None:
    root = tmp_path / "training"
    payload = _payload()
    environment = payload["environment"]
    assert isinstance(environment, dict)
    execution = environment["execution_cost"]
    assert isinstance(execution, dict)
    execution["trigger_volume_fractions"] = 1.0
    _write_environment(root, payload)

    with pytest.raises(ValueError, match="trigger_volume_fractions"):
        serving_package._execution_cost(root)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "training environment must be a mapping"),
        (
            {"schema_version": "training_environment_v2", "environment": []},
            "environment must be a mapping",
        ),
        (
            {
                "schema_version": "training_environment_v2",
                "environment": {"execution_cost": []},
            },
            "execution_cost must be a mapping",
        ),
    ],
)
def test_execution_cost_rejects_malformed_mappings(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    root = tmp_path / "training"
    _write_environment(root, payload)

    with pytest.raises(ValueError, match=message):
        serving_package._execution_cost(root)


def test_serving_package_delegates_execution_artifact_decoding() -> None:
    source = inspect.getsource(serving_package)

    assert "load_training_execution_cost(" in source
    assert "ExecutionCostConfig(" not in source
    assert "defaults = ExecutionCostConfig()" not in source


def test_serving_packaging_has_per_file_critical_branch_ratchets() -> None:
    configuration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"trade_rl/serving/package.py" = 90.0' in configuration
    assert '"trade_rl/serving/training_environment.py" = 100.0' in configuration
