"""Bounded v4/v7 content verification, without native Dataset construction."""

from __future__ import annotations

import ast
import hashlib
import io
import math
import re
import stat
import struct
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tools.market_replay_oracle.identity import (
    ARRAY_FIELDS,
    CONTRACT_FIELDS,
    canonical,
    content_id,
    json_object,
)

_FILES = {"manifest.json", "arrays.npz"}
_NAMES = {*ARRAY_FIELDS, "asset_active"}
_MANIFEST_FIELDS = {
    "arrays",
    "arrays_digest",
    "arrays_file",
    "artifact_digest",
    "dataset",
    "dataset_id",
    "schema_version",
}
_BOOL = {
    "information_available",
    "feature_available",
    "global_feature_available",
    "tradable",
    "symbol_active",
    "asset_active",
    "borrow_available",
    "funding_due",
    "buy_allowed",
    "sell_allowed",
}
_INT = {
    "feature_missing_reason",
    "global_feature_missing_reason",
    "funding_event_count",
}
_LOCAL = {
    "features",
    "feature_available",
    "feature_staleness",
    "feature_staleness_hours",
    "feature_missing_reason",
}
_GLOBAL = {
    "global_features",
    "global_feature_available",
    "global_feature_staleness_hours",
    "global_feature_missing_reason",
}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("lowercase SHA-256 required")
    return value


@dataclass(frozen=True, slots=True)
class DatasetContentPin:
    """Trusted caller's exact file/content pins, not research authorization."""

    dataset_id: str
    manifest_sha256: str
    arrays_sha256: str
    manifest_bytes: int
    arrays_bytes: int

    def __post_init__(self) -> None:
        for value in (self.dataset_id, self.manifest_sha256, self.arrays_sha256):
            _digest(value)
        for size, maximum in (
            (self.manifest_bytes, 1024 * 1024),
            (self.arrays_bytes, 64 * 1024 * 1024),
        ):
            if type(size) is not int or not 0 < size <= maximum:
                raise ValueError("raw file size is outside bounded content limits")


@dataclass(frozen=True, slots=True)
class VerifiedDatasetContent:
    """Immutable content view; no policy, execution or economic admission."""

    dataset_id: str
    arrays: Mapping[str, NDArray[Any]]
    dataset: Mapping[str, object]


def _regular(path: Path, *, directory: bool = False) -> None:
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & 0x400
    ):
        raise ValueError(
            "artifact requires regular files without links or reparse points"
        )


def _read(path: Path, size: int, digest: str) -> bytes:
    _regular(path)
    if path.stat().st_size != size:
        raise ValueError("raw source byte count mismatch")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    _regular(path)
    if len(raw) != size or _sha(raw) != digest:
        raise ValueError("raw source SHA-256 mismatch")
    return raw


def _header(raw: bytes) -> tuple[str, tuple[int, ...], int]:
    if raw[:6] != b"\x93NUMPY" or raw[6:8] not in {b"\x01\x00", b"\x02\x00"}:
        raise ValueError("unsupported NPY format")
    start = 10 if raw[6] == 1 else 12
    length = int.from_bytes(raw[8:start], "little")
    end = start + length
    if not 0 < length <= 4096 or end > len(raw) or end % 16:
        raise ValueError("unbounded or malformed NPY header")
    try:
        header = ast.literal_eval(raw[start:end].decode("latin1"))
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("invalid NPY header") from error
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header["fortran_order"] is not False
        or not raw[start:end].endswith(b"\n")
    ):
        raise ValueError("only primitive C-order NPY arrays are supported")
    dtype, shape = header["descr"], header["shape"]
    allowed = {"|b1", "|i1", "<i2", "<i4", "<i8", "<f4", "<f8", "<M8[ns]"}
    if not isinstance(dtype, str) or dtype not in allowed:
        raise ValueError("unsupported NPY dtype or byte order")
    if (
        not isinstance(shape, tuple)
        or not 1 <= len(shape) <= 3
        or any(type(n) is not int or n < 0 or n > 64 * 1024 * 1024 for n in shape)
    ):
        raise ValueError("invalid NPY shape")
    if math.prod(shape) * np.dtype(dtype).itemsize != len(raw) - end:
        raise ValueError("NPY payload length mismatch")
    return dtype, shape, end


def _zip_directory_bound(raw: bytes) -> None:
    # ZipFile builds its whole central-entry list during construction. Bound that
    # allocation first, using the closed stored-only, comment-free NPZ format.
    if len(raw) < 22 or raw[-22:-18] != b"PK\x05\x06":
        raise ValueError("NPZ requires an ordinary comment-free ZIP end record")
    _, disk, directory_disk, disk_count, count, size, offset, comment = struct.unpack(
        "<4s4H2LH", raw[-22:]
    )
    if (
        disk != 0
        or directory_disk != 0
        or disk_count != len(_NAMES)
        or count != len(_NAMES)
        or comment != 0
        or not 0 < size <= 64 * 1024
        or offset + size != len(raw) - 22
    ):
        raise ValueError("NPZ central directory is outside bounded v4 framing")


def _arrays(raw: bytes, descriptions: object) -> dict[str, NDArray[Any]]:
    if not isinstance(descriptions, dict) or set(descriptions) != _NAMES:
        raise ValueError("v4 array descriptor roster mismatch")
    _zip_directory_bound(raw)
    parsed: dict[str, tuple[bytes, str, tuple[int, ...], int]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if (
                len(members) != len(_NAMES)
                or any(i.orig_filename != i.filename for i in members)
                or {i.filename for i in members}
                != {name + ".npy" for name in _NAMES}
            ):
                raise ValueError("NPZ requires exactly the unique 43 safe NPY members")
            if sum(i.file_size for i in members) > len(raw):
                raise ValueError(
                    "NPZ declared allocation exceeds verified stored bytes"
                )
            for info in members:
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or info.compress_size != info.file_size
                    or info.flag_bits & 1
                    or stat.S_IFMT(info.external_attr >> 16) not in {0, stat.S_IFREG}
                ):
                    raise ValueError("NPZ requires unencrypted stored regular members")
                name = info.filename[:-4]
                member = archive.read(info)  # ZIP verifies CRC before any NumPy view.
                dtype, shape, offset = _header(member)
                description = descriptions[name]
                if (
                    not isinstance(description, dict)
                    or set(description) != {"dtype", "shape"}
                    or description["dtype"] != dtype
                    or not isinstance(description["shape"], list)
                    or any(type(n) is not int for n in description["shape"])
                    or description["shape"] != list(shape)
                ):
                    raise ValueError(
                        "manifest array descriptor differs from NPY header"
                    )
                parsed[name] = (member, dtype, shape, offset)
    except (zipfile.BadZipFile, EOFError) as error:
        raise ValueError("invalid stored NPZ archive") from error
    # Every member/header has passed before creating the first array view.
    return {
        name: np.frombuffer(member, dtype=dtype, offset=offset).reshape(shape)
        for name, (member, dtype, shape, offset) in parsed.items()
    }


def _names(value: object, *, allow_empty: bool = False) -> list[str]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(x, str) or not x.strip() for x in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("invalid or duplicate dataset names")
    return value


def _contract(
    dataset: dict[str, Any], arrays: dict[str, NDArray[Any]]
) -> dict[str, Any]:
    symbols = _names(dataset["symbols"])
    features = _names(dataset["feature_names"], allow_empty=True)
    globals_ = _names(dataset["global_feature_names"], allow_empty=True)
    units = dataset["volume_units"]
    if (
        not isinstance(units, list)
        or len(units) != len(symbols)
        or any(
            not isinstance(unit, str)
            or unit not in {"base_asset", "contracts", "quote_notional"}
            for unit in units
        )
    ):
        raise ValueError("invalid volume units")
    if dataset["calendar_kind"] != "continuous_24_7":
        raise ValueError("only continuous content clocks are supported")
    for key in ("feature_config_digest", "normalization_digest"):
        _digest(dataset[key])
    n = arrays["timestamps"].shape[0]
    if n < 3:
        raise ValueError("at least three market rows required")
    for name, array in arrays.items():
        shape: tuple[int, ...] = (n, len(symbols))
        kind = "b" if name in _BOOL else "i" if name in _INT else "f"
        if name in {"timestamps", "available_at"}:
            kind = "M"
        if name in {"timestamps", "cash_rate"}:
            shape = (n,)
        elif name == "contract_multipliers":
            shape = (len(symbols),)
        elif name in _LOCAL:
            shape = (n, len(symbols), len(features))
        elif name in _GLOBAL:
            shape = (n, len(globals_))
        if array.shape != shape or array.dtype.kind != kind:
            raise ValueError("dataset array shape or primitive type mismatch")
        if kind == "f" and not np.isfinite(array).all():
            raise ValueError("nonfinite array values")
        if kind == "M" and np.isnat(array).any():
            raise ValueError("NaT content clock")
    times = arrays["timestamps"].astype("<i8")
    step = int(times[1]) - int(times[0])
    if step <= 0 or any(int(b) - int(a) != step for a, b in zip(times, times[1:])):
        raise ValueError(
            "continuous timestamps must be strictly increasing and regular"
        )
    hours = step / 3_600_000_000_000
    nominal, periods = dataset["nominal_bar_hours"], dataset["periods_per_year"]
    if (
        (
            nominal is not None
            and (type(nominal) not in {int, float} or nominal != hours)
        )
        or type(periods) is not int
        or periods <= 0
        or not math.isclose(8760 / hours, periods, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("dataset scalar cadence differs from raw timestamps")
    if (arrays["available_at"] < arrays["timestamps"][:, None]).any():
        raise ValueError("available_at precedes source row")
    if not np.array_equal(arrays["asset_active"], arrays["symbol_active"]):
        raise ValueError("dataset activity aliases differ")
    # Continuous Dataset writers may retain an unspecified constructor value.
    # Its identity binds the resolved cadence derived from timestamps instead.
    return {
        **{key: dataset[key] for key in CONTRACT_FIELDS},
        "nominal_bar_hours": hours,
    }


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def load_pinned_v4(root: Path, pin: DatasetContentPin) -> VerifiedDatasetContent:
    """Recompute pinned content; caller separately authorizes protocol and scope."""
    root = Path(root).absolute()
    for ancestor in (root, *root.parents):
        _regular(ancestor, directory=True)
    if {path.name for path in root.iterdir()} != _FILES:
        raise ValueError("artifact requires exactly manifest.json and arrays.npz")
    manifest_raw = _read(
        root / "manifest.json", pin.manifest_bytes, pin.manifest_sha256
    )
    arrays_raw = _read(root / "arrays.npz", pin.arrays_bytes, pin.arrays_sha256)
    manifest = json_object(manifest_raw)
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("v4 manifest field closure mismatch")
    if (
        manifest["schema_version"] != "market_dataset_artifact_v4"
        or manifest["arrays_file"] != "arrays.npz"
        or manifest["arrays_digest"] != pin.arrays_sha256
        or manifest["dataset_id"] != pin.dataset_id
    ):
        raise ValueError("manifest identities differ from pinned v4 content")
    payload = {
        key: value for key, value in manifest.items() if key != "artifact_digest"
    }
    if _sha(canonical(payload)) != manifest["artifact_digest"]:
        raise ValueError("manifest artifact digest mismatch")
    dataset = manifest["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        *CONTRACT_FIELDS,
        "dataset_id",
        "identity_payload_json",
    }:
        raise ValueError("dataset scalar field closure mismatch")
    if dataset["dataset_id"] != pin.dataset_id or not isinstance(
        dataset["identity_payload_json"], str
    ):
        raise ValueError("dataset identity payload missing or mismatched")
    identity = json_object(dataset["identity_payload_json"].encode("utf-8"))
    arrays = _arrays(arrays_raw, manifest["arrays"])
    contract = _contract(dataset, arrays)
    if identity.get("schema") != "market_dataset_identity_v7" or canonical(
        identity.get("dataset_contract")
    ) != canonical(contract):
        raise ValueError("v7 identity scalar contract mismatch")
    if content_id(identity, arrays) != pin.dataset_id:
        raise ValueError("independent dataset content identity mismatch")
    return VerifiedDatasetContent(
        pin.dataset_id, MappingProxyType(arrays), _freeze(dataset)
    )
