"""Raw admission must bound allocation before parsing caller-selected files."""

import pytest

from tools.market_replay_oracle import DatasetContentPin


def _fields() -> dict[str, object]:
    return {
        "dataset_id": "a" * 64,
        "manifest_sha256": "b" * 64,
        "arrays_sha256": "c" * 64,
        "manifest_bytes": 128,
        "arrays_bytes": 1024,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "a" * 63),
        ("manifest_sha256", "B" * 64),
        ("arrays_sha256", "z" * 64),
        ("manifest_bytes", True),
        ("manifest_bytes", 0),
        ("manifest_bytes", 1024 * 1024 + 1),
        ("arrays_bytes", False),
        ("arrays_bytes", -1),
        ("arrays_bytes", 64 * 1024 * 1024 + 1),
    ],
)
def test_invalid_pin_is_rejected_before_any_file_access(
    field: str, value: object
) -> None:
    fields = _fields()
    fields[field] = value
    with pytest.raises(ValueError):
        DatasetContentPin(**fields)  # type: ignore[arg-type]


def test_exact_resource_limits_are_admissible_pin_values() -> None:
    fields = _fields()
    fields.update(manifest_bytes=1024 * 1024, arrays_bytes=64 * 1024 * 1024)
    pin = DatasetContentPin(**fields)  # type: ignore[arg-type]
    assert pin.manifest_bytes == 1024 * 1024
    assert pin.arrays_bytes == 64 * 1024 * 1024
