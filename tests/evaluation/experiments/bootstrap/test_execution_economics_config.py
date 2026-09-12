from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_config import _valid_payload
from trade_rl.data.build import ExecutionEconomicsProfile
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)

_V1_DIGEST = "0e27cd7608b12261051db0f37ca72a90584ba32a786be93d44de1f6ca820bc0c"


def _profile(*, fee_rate: float = 0.0005) -> ExecutionEconomicsProfile:
    return ExecutionEconomicsProfile(
        name="canonical_m2_research_assumption_v1",
        fee_rate=fee_rate,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )


def _write(tmp_path: Path, payload: object, *, name: str = "bootstrap.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _v2_payload(*, fee_rate: float = 0.0005) -> dict[str, object]:
    payload = deepcopy(_valid_payload())
    payload["schema_version"] = "canonical_m2_bootstrap_config_v2"
    payload["execution_economics"] = _profile(fee_rate=fee_rate).to_payload()
    return payload


def test_v1_payload_digest_remains_compatible_without_execution_economics(
    tmp_path: Path,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _valid_payload()))

    assert config.schema_version == "canonical_m2_bootstrap_config_v1"
    assert config.digest == _V1_DIGEST
    assert "execution_economics" not in config.to_payload()


def test_v1_rejects_execution_economics_field(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["execution_economics"] = _profile().to_payload()

    with pytest.raises(ValueError, match="unknown.*execution_economics"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_v2_requires_execution_economics(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["schema_version"] = "canonical_m2_bootstrap_config_v2"

    with pytest.raises(ValueError, match="missing.*execution_economics"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_v2_round_trips_execution_economics_and_binds_digest(tmp_path: Path) -> None:
    base = load_canonical_m2_bootstrap_config(
        _write(tmp_path, _v2_payload(), name="base.json")
    )
    changed = load_canonical_m2_bootstrap_config(
        _write(tmp_path, _v2_payload(fee_rate=0.0006), name="changed.json")
    )

    assert base.schema_version == "canonical_m2_bootstrap_config_v2"
    assert base.execution_economics == _profile()
    assert base.to_payload()["execution_economics"] == _profile().to_payload()
    assert base.digest != changed.digest
    assert base.digest != _V1_DIGEST


def test_v2_rejects_unknown_execution_economics_field(tmp_path: Path) -> None:
    payload = _v2_payload()
    economics = payload["execution_economics"]
    assert isinstance(economics, dict)
    economics["unexpected"] = True

    with pytest.raises(ValueError, match="execution_economics contains unknown fields"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))
