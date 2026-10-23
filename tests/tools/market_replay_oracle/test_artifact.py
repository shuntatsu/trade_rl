from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from tools.market_replay_oracle import DatasetContentPin, load_pinned_v4
from trade_rl.data import write_market_dataset_files
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset

EXPECTED_ARRAYS = set(
    """
    timestamps available_at information_available features feature_available
    feature_staleness feature_staleness_hours feature_missing_reason global_features
    global_feature_available global_feature_staleness_hours global_feature_missing_reason
    open high low close volume funding_rate funding_price_rate funding_event_count
    tradable symbol_active fee_rate maker_fee_rate taker_fee_rate spread_rate
    max_participation_rate minimum_notional lot_size tick_size borrow_available
    borrow_rate funding_due buy_allowed sell_allowed mark_price index_price dividend
    split_factor delisting_recovery cash_rate contract_multipliers asset_active
    """.split()
)


def _market() -> MarketDataset:
    bars = 6
    shape = (bars, 2)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        bars
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(bars * 2).reshape(shape)
    funding_due = np.zeros(shape, dtype=bool)
    funding_due[2, 0] = True
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        features=np.zeros((bars, 2, 2), dtype=np.float32),
        global_features=np.zeros((bars, 2), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(shape, 1_000.0),
        funding_rate=np.zeros(shape),
        funding_due=funding_due,
        funding_event_count=np.zeros(shape, dtype=np.int32),
        tradable=np.ones(shape, dtype=bool),
        feature_available=np.ones((bars, 2, 2), dtype=bool),
        feature_names=("momentum", "volatility"),
        global_feature_names=("market_mean", "market_dispersion"),
        periods_per_year=8_760,
        volume_units=(VolumeUnit.BASE_ASSET, VolumeUnit.CONTRACTS),
        contract_multipliers=np.array([1.0, 10.0], dtype=np.float64),
    ).with_content_identity({"fixture": "market-replay-content-v4"})


def _write_fixture(root: Path) -> MarketDataset:
    dataset = _market()
    write_market_dataset_files(root, dataset)
    return dataset


def _pin(root: Path, dataset_id: str) -> DatasetContentPin:
    manifest = (root / "manifest.json").read_bytes()
    arrays = (root / "arrays.npz").read_bytes()
    return DatasetContentPin(
        dataset_id=dataset_id,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        arrays_sha256=hashlib.sha256(arrays).hexdigest(),
        manifest_bytes=len(manifest),
        arrays_bytes=len(arrays),
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _rewrite_artifact(
    root: Path,
    dataset_id: str,
    *,
    array_changes: Mapping[str, np.ndarray] | None = None,
    dataset_changes: Mapping[str, object] | None = None,
    compression: int = zipfile.ZIP_STORED,
    duplicate_member: str | None = None,
    extra_fields: bool = False,
) -> DatasetContentPin:
    """Mutate only this synthetic fixture and refresh its outer file pins."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    with np.load(
        io.BytesIO((root / "arrays.npz").read_bytes()), allow_pickle=False
    ) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays.update(array_changes or {})

    archive_buffer = io.BytesIO()
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(archive_buffer, "w", compression=compression) as output:
        for name, array in sorted(arrays.items()):
            member_buffer = io.BytesIO()
            np.lib.format.write_array(
                member_buffer,
                np.asarray(array),
                allow_pickle=np.asarray(array).dtype.hasobject,
            )
            member = f"{name}.npy"
            members[member] = member_buffer.getvalue()
            info = zipfile.ZipInfo(member)
            info.compress_type = compression
            if extra_fields:
                info.extra = b"\xfe\xca\x64\x00" + b"x" * 100
            output.writestr(info, members[member])
        if duplicate_member is not None:
            member = f"{duplicate_member}.npy"
            info = zipfile.ZipInfo(member)
            info.compress_type = compression
            output.writestr(info, members[member])
    arrays_raw = archive_buffer.getvalue()
    (root / "arrays.npz").write_bytes(arrays_raw)

    descriptors = {}
    for name, array in arrays.items():
        value = np.asarray(array)
        descriptors[name] = {"dtype": value.dtype.str, "shape": list(value.shape)}
    manifest["arrays"] = descriptors
    manifest["arrays_digest"] = hashlib.sha256(arrays_raw).hexdigest()
    if dataset_changes:
        manifest["dataset"].update(dataset_changes)
    payload = {
        key: value for key, value in manifest.items() if key != "artifact_digest"
    }
    manifest["artifact_digest"] = hashlib.sha256(_canonical(payload)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))
    return _pin(root, dataset_id)


def _pin_after_manifest_bytes(
    root: Path, dataset_id: str, raw: bytes
) -> DatasetContentPin:
    (root / "manifest.json").write_bytes(raw)
    return _pin(root, dataset_id)


def _install_malformed_archive(root: Path, dataset_id: str) -> DatasetContentPin:
    raw = b"not-a-zip"
    (root / "arrays.npz").write_bytes(raw)
    path = root / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["arrays_digest"] = hashlib.sha256(raw).hexdigest()
    payload = {
        key: value for key, value in manifest.items() if key != "artifact_digest"
    }
    manifest["artifact_digest"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path.write_bytes(_canonical(manifest))
    return _pin(root, dataset_id)


def _assert_rejected(root: Path, pin: DatasetContentPin, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_pinned_v4(root, pin)


def test_loads_v4_content_with_exact_arrays_and_deep_immutability(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    assert original.dataset_id == (
        "d6669590441c27e0200a3f6af9cb24840d372b6cc7932c45cf6d3224360434af"
    )

    loaded = load_pinned_v4(root, _pin(root, original.dataset_id))

    assert loaded.dataset_id == (
        "d6669590441c27e0200a3f6af9cb24840d372b6cc7932c45cf6d3224360434af"
    )
    assert set(loaded.arrays) == EXPECTED_ARRAYS
    assert loaded.dataset["dataset_id"] == original.dataset_id
    assert loaded.dataset["symbols"] == ("BTCUSDT", "ETHUSDT")
    assert loaded.dataset["volume_units"] == ("base_asset", "contracts")
    assert loaded.arrays["funding_due"][2, 0]
    assert loaded.arrays["funding_event_count"][2, 0] == 0
    np.testing.assert_array_equal(loaded.arrays["close"], original.close)

    assert isinstance(loaded.arrays, Mapping)
    assert isinstance(loaded.dataset, Mapping)
    with pytest.raises(TypeError):
        loaded.arrays["close"] = loaded.arrays["open"]  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded.dataset["symbols"] = ("OTHER", "ETHUSDT")  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded.dataset["symbols"][0] = "OTHER"  # type: ignore[index]
    for array in loaded.arrays.values():
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE|writeable|writable"):
            array.setflags(write=True)


@pytest.mark.parametrize(
    ("array_name", "mutate"),
    [
        ("close", lambda value: value.__setitem__((0, 0), value[0, 0] + 0.25)),
        (
            "contract_multipliers",
            lambda value: value.__setitem__(1, value[1] + 0.5),
        ),
    ],
)
def test_rehashed_array_change_cannot_keep_the_old_dataset_identity(
    tmp_path: Path, array_name: str, mutate: object
) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    with np.load(
        io.BytesIO((root / "arrays.npz").read_bytes()), allow_pickle=False
    ) as archive:
        changed = archive[array_name].copy()
    mutate(changed)  # type: ignore[operator]

    pin = _rewrite_artifact(
        root, original.dataset_id, array_changes={array_name: changed}
    )

    _assert_rejected(root, pin, "identity")


def test_changed_volume_unit_cannot_reuse_the_stored_v7_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    pin = _rewrite_artifact(
        root,
        original.dataset_id,
        dataset_changes={"volume_units": ["base_asset", "quote_notional"]},
    )

    _assert_rejected(root, pin, "contract")


def test_activity_alias_mismatch_is_rejected_after_outer_hashes_are_refreshed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    with np.load(
        io.BytesIO((root / "arrays.npz").read_bytes()), allow_pickle=False
    ) as archive:
        changed = archive["asset_active"].copy()
        changed[0, 0] = not changed[0, 0]
    pin = _rewrite_artifact(
        root, original.dataset_id, array_changes={"asset_active": changed}
    )

    _assert_rejected(root, pin, "activity aliases")


def test_nonregular_timestamp_clock_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    with np.load(
        io.BytesIO((root / "arrays.npz").read_bytes()), allow_pickle=False
    ) as archive:
        changed = archive["timestamps"].copy()
    changed[0] = changed[1]
    pin = _rewrite_artifact(
        root, original.dataset_id, array_changes={"timestamps": changed}
    )

    _assert_rejected(root, pin, "timestamps")


@pytest.mark.parametrize(
    "malformation",
    ["duplicate_json_key", "nonfinite_json_number"],
)
def test_manifest_parser_rejects_duplicate_or_nonfinite_json(
    tmp_path: Path, malformation: str
) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    raw = (root / "manifest.json").read_bytes()
    if malformation == "duplicate_json_key":
        raw = raw.replace(b'{"arrays":', b'{"arrays":{},"arrays":', 1)
    else:
        raw = raw.replace(b'"periods_per_year":8760', b'"periods_per_year":NaN', 1)
        assert b'"periods_per_year":NaN' in raw
    pin = _pin_after_manifest_bytes(root, original.dataset_id, raw)

    _assert_rejected(root, pin, "duplicate|nonfinite|JSON|overflow")


def test_npz_member_roster_is_checked_before_any_numpy_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    with pytest.warns(UserWarning, match="Duplicate name"):
        pin = _rewrite_artifact(root, original.dataset_id, duplicate_member="close")
    import tools.market_replay_oracle.artifact as artifact_module

    calls: list[object] = []
    original_frombuffer = artifact_module.np.frombuffer

    def observe_frombuffer(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(args[0] if args else None)
        return original_frombuffer(*args, **kwargs)

    monkeypatch.setattr(artifact_module.np, "frombuffer", observe_frombuffer)

    _assert_rejected(root, pin, "43 safe NPY members|central directory")
    assert calls == []


def test_malformed_npz_is_rejected_before_constructing_zip_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    pin = _install_malformed_archive(root, original.dataset_id)
    import tools.market_replay_oracle.artifact as artifact_module

    def forbidden_zip_reader(*args: object, **kwargs: object) -> None:
        raise AssertionError("ZipFile constructed before bounded archive preflight")

    monkeypatch.setattr(artifact_module.zipfile, "ZipFile", forbidden_zip_reader)
    _assert_rejected(root, pin, "NPZ|ZIP|archive")


@pytest.mark.parametrize("malformation", ["compressed", "object"])
def test_compressed_or_object_member_is_rejected_before_any_numpy_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformation: str
) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    changes: dict[str, np.ndarray] | None = None
    if malformation == "object":
        with np.load(
            io.BytesIO((root / "arrays.npz").read_bytes()), allow_pickle=False
        ) as archive:
            unsafe = np.zeros(archive["features"].shape, dtype=object)
        changes = {"features": unsafe}
    pin = _rewrite_artifact(
        root,
        original.dataset_id,
        array_changes=changes,
        compression=zipfile.ZIP_DEFLATED
        if malformation == "compressed"
        else zipfile.ZIP_STORED,
        extra_fields=malformation == "compressed",
    )
    import tools.market_replay_oracle.artifact as artifact_module

    calls: list[object] = []
    original_frombuffer = artifact_module.np.frombuffer

    def observe_frombuffer(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(args[0] if args else None)
        return original_frombuffer(*args, **kwargs)

    monkeypatch.setattr(artifact_module.np, "frombuffer", observe_frombuffer)

    message = (
        "stored regular members"
        if malformation == "compressed"
        else "unsupported NPY dtype"
    )
    _assert_rejected(root, pin, message)
    assert calls == []


def test_artifact_root_rejects_undeclared_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    original = _write_fixture(root)
    (root / "notes.txt").write_text("not part of the artifact", encoding="utf-8")

    _assert_rejected(
        root, _pin(root, original.dataset_id), "exactly manifest.json and arrays.npz"
    )
