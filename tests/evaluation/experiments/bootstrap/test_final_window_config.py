from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_execution_economics_config import (
    _v2_payload,
)
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)


def _write(tmp_path: Path, payload: object, *, name: str = "bootstrap.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _v3_payload() -> dict[str, object]:
    payload = deepcopy(_v2_payload())
    payload["schema_version"] = "canonical_m2_bootstrap_config_v3"
    payload["final_evaluation_start"] = _FINAL_START
    payload["final_evaluation_stop_exclusive"] = _FINAL_STOP
    return payload


def test_v3_binds_preregistered_final_window_into_bootstrap_digest(
    tmp_path: Path,
) -> None:
    base = load_canonical_m2_bootstrap_config(
        _write(tmp_path, _v3_payload(), name="base.json")
    )
    changed_payload = _v3_payload()
    changed_payload["final_evaluation_stop_exclusive"] = "2025-05-01T00:00:00+00:00"
    changed = load_canonical_m2_bootstrap_config(
        _write(tmp_path, changed_payload, name="changed.json")
    )

    assert base.schema_version == "canonical_m2_bootstrap_config_v3"
    assert base.to_payload()["final_evaluation_start"] == _FINAL_START
    assert base.to_payload()["final_evaluation_stop_exclusive"] == _FINAL_STOP
    assert base.digest != changed.digest


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("final_evaluation_start", None, "missing|final_evaluation_start"),
        ("final_evaluation_stop_exclusive", None, "missing|final_evaluation_stop"),
        (
            "final_evaluation_start",
            "2024-12-31T23:00:00+00:00",
            "data_stop|development",
        ),
        (
            "final_evaluation_stop_exclusive",
            _FINAL_START,
            "strictly later|stop",
        ),
    ),
)
def test_v3_rejects_invalid_final_window(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _v3_payload()
    if value is None:
        del payload[field]
    else:
        payload[field] = value

    with pytest.raises(ValueError, match=match):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_v2_rejects_final_window_fields_as_unknown(tmp_path: Path) -> None:
    payload = _v2_payload()
    payload["final_evaluation_start"] = _FINAL_START
    payload["final_evaluation_stop_exclusive"] = _FINAL_STOP

    with pytest.raises(ValueError, match="unknown"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))
