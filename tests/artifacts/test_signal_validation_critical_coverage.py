from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import trade_rl.artifacts.signals as signals

SHA = "a" * 64


def _signal_manifest(**overrides: object) -> signals.SignalArrayManifest:
    values: dict[str, object] = {
        "artifact_digest": "1" * 64,
        "arrays_digest": "2" * 64,
        "dataset_id": "3" * 64,
        "fit_start": 0,
        "fit_stop": 1,
        "prediction_start": 1,
        "prediction_stop": 4,
        "generator_config_digest": "4" * 64,
        "generator_code_digest": "5" * 64,
        "kind": "alpha",
        "names": ("BTC",),
        "shape": (4, 1),
        "dtype": "<f8",
        "available_at_dtype": "<i8",
    }
    values.update(overrides)
    return signals.SignalArrayManifest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"kind": "unknown"},
        {"fit_start": -1},
        {"fit_stop": 0},
        {"prediction_start": 0},
        {"prediction_stop": 1},
        {"names": ()},
        {"names": ("BTC", "BTC")},
        {"shape": ()},
        {"shape": (0, 1)},
        {"prediction_stop": 5},
        {"schema_version": "signal_v0"},
    ],
)
def test_signal_manifest_rejects_invalid_semantics(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _signal_manifest(**override)


def _array_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "kind": "alpha",
        "names": ("BTC",),
        "values": np.zeros((4, 1), dtype=np.float64),
        "fit_stop": 1,
        "prediction_start": 1,
        "valid": None,
        "valid_mask": None,
        "available_at": None,
        "knowledge_cutoff": None,
    }
    values.update(overrides)
    return values


def test_signal_array_validation_rejects_shape_availability_and_cutoff_errors() -> None:
    invalid_cases = (
        ({"values": np.zeros(4)}, "rank-2"),
        ({"values": np.array([["x"]] * 4)}, "numeric"),
        ({"values": np.array([[0.0], [0.0], [np.nan], [0.0]])}, "finite"),
        ({"values": np.zeros((4, 2))}, "alpha values"),
        ({"kind": "factor", "values": np.zeros((4, 1))}, "factor values"),
        (
            {
                "valid": np.ones((4, 1), dtype=np.bool_),
                "valid_mask": np.ones(4, dtype=np.bool_),
            },
            "only one",
        ),
        ({"valid": np.ones((4, 2), dtype=np.bool_)}, "validity shape"),
        ({"valid_mask": np.ones(3, dtype=np.bool_)}, "bar count"),
        ({"available_at": np.arange(3)}, "one value"),
        (
            {"available_at": np.array(["NaT"] * 4, dtype="datetime64[ns]")},
            "NaT",
        ),
        ({"available_at": np.array([-1, 0, 1, 2])}, "non-negative"),
        ({"available_at": np.arange(4, dtype=np.float64)}, "datetime64"),
        ({"knowledge_cutoff": np.array([-1, 0, 1])}, "bar count"),
        (
            {
                "valid": np.array([[True], [True], [False], [True]]),
                "knowledge_cutoff": np.array([-1, 0, 1, 2]),
            },
            "invalid signal rows",
        ),
        ({"knowledge_cutoff": np.array([-1, 1, 1, 2])}, "strictly precede"),
    )
    for override, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            signals._validate_arrays(**_array_kwargs(**override))  # type: ignore[arg-type]

    datetime_arrays = signals._validate_arrays(
        **_array_kwargs(
            available_at=np.datetime64("2026-01-01", "ns")
            + np.arange(4) * np.timedelta64(1, "h"),
            valid_mask=np.array([False, True, True, True]),
        )
    )
    assert datetime_arrays.valid_mask.tolist() == [False, True, True, True]

    factor_arrays = signals._validate_arrays(
        **_array_kwargs(
            kind="factor",
            names=("momentum",),
            values=np.zeros((4, 1, 2)),
        )
    )
    assert factor_arrays.shape == (4, 1, 2)


def test_signal_file_closure_and_write_preconditions(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        signals._verify_file_closure(tmp_path / "missing")

    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="closure"):
        signals._verify_file_closure(extra)

    values = np.zeros((4, 1))
    with pytest.raises(ValueError, match="prediction range"):
        signals.write_signal_artifact(
            tmp_path / "early",
            kind="alpha",
            dataset_id=SHA,
            fit_start=0,
            fit_stop=2,
            prediction_start=1,
            names=("BTC",),
            values=values,
        )
    with pytest.raises(ValueError, match="non-empty"):
        signals.write_signal_artifact(
            tmp_path / "empty-range",
            kind="alpha",
            dataset_id=SHA,
            fit_start=0,
            fit_stop=1,
            prediction_start=2,
            prediction_stop=2,
            names=("BTC",),
            values=values,
        )
    with pytest.raises(ValueError, match="generator_digest"):
        signals.write_signal_artifact(
            tmp_path / "bad-generator",
            kind="alpha",
            dataset_id=SHA,
            fit_start=0,
            fit_stop=1,
            names=("BTC",),
            values=values,
            generator_digest="bad",
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "file").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        signals.write_signal_artifact(
            occupied,
            kind="alpha",
            dataset_id=SHA,
            fit_start=0,
            fit_stop=1,
            names=("BTC",),
            values=values,
        )
