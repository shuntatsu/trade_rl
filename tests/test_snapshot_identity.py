import numpy as np
import pytest

from mars_lite.serving.snapshot_identity import compute_snapshot_id


def _payload(**overrides):
    values = {
        "bundle_digest": "bundle-digest",
        "base_timeframe": "1h",
        "timestamps": np.asarray(
            ["2026-07-12T08:00:00", "2026-07-12T09:00:00"],
            dtype="datetime64[ns]",
        ),
        "symbols": ("BTCUSDT",),
        "feature_names": ("ret", "vol"),
        "global_feature_names": ("hour_sin",),
        "feature_history": np.asarray([[[1.5, 2.0]], [[2.5, 4.0]]], dtype=np.float64),
        "global_features": np.asarray([0.5], dtype=np.float64),
        "close_history": np.asarray([[100.0], [101.0]], dtype=np.float64),
    }
    values.update(overrides)
    return values


def test_snapshot_id_is_stable_across_equivalent_array_representations() -> None:
    base = _payload()
    native = compute_snapshot_id(**base)
    represented = compute_snapshot_id(
        **_payload(
            feature_history=np.asarray(base["feature_history"], dtype=">f4"),
            global_features=np.asarray(base["global_features"], dtype=">f4"),
            close_history=np.asfortranarray(
                np.asarray(base["close_history"], dtype=np.float32)
            ),
        )
    )

    assert represented == native


def test_snapshot_id_changes_when_one_feature_value_changes() -> None:
    mutated = np.asarray(_payload()["feature_history"]).copy()
    mutated[1, 0, 0] += 1.0

    assert compute_snapshot_id(**_payload()) != compute_snapshot_id(
        **_payload(feature_history=mutated)
    )


def test_snapshot_id_changes_when_close_history_changes() -> None:
    mutated = np.asarray(_payload()["close_history"]).copy()
    mutated[-1, 0] += 1.0

    assert compute_snapshot_id(**_payload()) != compute_snapshot_id(
        **_payload(close_history=mutated)
    )


def test_snapshot_id_changes_when_ordered_schema_changes() -> None:
    assert compute_snapshot_id(**_payload()) != compute_snapshot_id(
        **_payload(feature_names=("vol", "ret"))
    )


@pytest.mark.parametrize(
    "field",
    ["feature_history", "global_features", "close_history"],
)
def test_snapshot_id_rejects_non_finite_values(field: str) -> None:
    value = np.asarray(_payload()[field]).copy()
    value.reshape(-1)[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_snapshot_id(**_payload(**{field: value}))
