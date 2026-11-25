from __future__ import annotations

import io
import struct
import zipfile
import zlib

import numpy as np
import pytest

from trade_rl.evaluation.runs import artifact as candidate_artifact


# Regression for the real zlib.error observed by baseline artifact falsification run 34616635445.
# Keep the unit oracle at the Candidate artifact boundary where backend errors are normalized.
def test_load_returns_normalizes_zlib_decompression_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_decompression(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise zlib.error("corrupt compressed NPZ member")

    monkeypatch.setattr(candidate_artifact.np, "load", fail_decompression)

    payload = io.BytesIO()
    np.savez(payload, symbol_0_strategy_0=np.asarray([0.0]))
    with pytest.raises(ValueError, match="malformed candidate returns archive"):
        candidate_artifact._load_returns(
            payload.getvalue(),
            expected_keys=frozenset({"symbol_0_strategy_0"}),
        )


@pytest.mark.parametrize(
    ("header_offset", "directory_offset", "value"),
    [
        (8, 10, 99),  # Unknown ZIP compression method.
        (6, 8, 1),  # Encrypted member; candidate artifacts have no password.
        (6, 8, 1 << 5),  # Compressed patched data, unsupported by zipfile.
        (6, 8, 1 << 6),  # Strong encryption, unsupported by zipfile.
    ],
)
def test_malformed_archive_member_features_raise_candidate_input_error(
    header_offset: int,
    directory_offset: int,
    value: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = io.BytesIO()
    np.savez(payload, cell=np.asarray([0.1, 0.2]))
    malformed = bytearray(payload.getvalue())
    directory = malformed.index(b"PK\x01\x02")
    struct.pack_into("<H", malformed, header_offset, value)
    struct.pack_into("<H", malformed, directory + directory_offset, value)

    def forbidden_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unsupported ZIP member reached array decoding")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden_decode)
    with pytest.raises(ValueError, match="malformed candidate returns archive"):
        candidate_artifact._load_returns(
            bytes(malformed), expected_keys=frozenset({"cell"})
        )


@pytest.mark.parametrize(
    "compression",
    [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA],
)
def test_valid_zip_compression_retains_array_dtype_and_bytes(compression: int) -> None:
    value = np.asarray([0.1, 0.2], dtype=np.float32)
    member = io.BytesIO()
    np.save(member, value, allow_pickle=False)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=compression) as archive:
        archive.writestr("cell.npy", member.getvalue())
    result = candidate_artifact._load_returns(
        payload.getvalue(), expected_keys=frozenset({"cell"})
    )["cell"]
    assert result.dtype == value.dtype
    assert result.tobytes() == value.tobytes()


def test_unsupported_zip_extraction_version_raises_candidate_input_error() -> None:
    payload = io.BytesIO()
    np.savez(payload, cell=np.asarray([0.1, 0.2]))
    malformed = bytearray(payload.getvalue())
    directory = malformed.index(b"PK\x01\x02")
    struct.pack_into("<H", malformed, 4, 255)
    struct.pack_into("<H", malformed, directory + 6, 255)
    with pytest.raises(ValueError, match="malformed candidate returns archive"):
        candidate_artifact._load_returns(
            bytes(malformed), expected_keys=frozenset({"cell"})
        )


def test_corrupt_lzma_member_raises_candidate_input_error() -> None:
    member = io.BytesIO()
    np.save(member, np.asarray([0.1, 0.2]), allow_pickle=False)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_LZMA) as archive:
        archive.writestr("cell.npy", member.getvalue())
    malformed = bytearray(payload.getvalue())
    filename_bytes, extra_bytes = struct.unpack_from("<HH", malformed, 26)
    compressed_start = 30 + filename_bytes + extra_bytes
    # A ZIP LZMA stream's properties follow its 4-byte version/size header.
    malformed[compressed_start + 4] = 255
    with pytest.raises(ValueError, match="malformed candidate returns archive"):
        candidate_artifact._load_returns(
            bytes(malformed), expected_keys=frozenset({"cell"})
        )


@pytest.mark.parametrize(
    ("shape", "body"),
    [
        ((1,), b"\0" * 8 + b"extra"),
        ((0,), b"extra"),
        ((2,), b"\0" * 8),
        ((-1,), b""),
        ((2**63,), b"\0" * 8),
    ],
)
def test_npy_extent_is_checked_before_numpy_materializes_array(
    shape: tuple[int, ...], body: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        member, {"descr": "<f8", "fortran_order": False, "shape": shape}
    )
    member.write(body)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cell.npy", member.getvalue())

    def forbidden_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("malformed NPY extent reached array materialization")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden_decode)
    with pytest.raises(ValueError, match="candidate return.*(extent|shape|size)"):
        candidate_artifact._load_returns(
            payload.getvalue(), expected_keys=frozenset({"cell"})
        )


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("dtype", [">i4", "<f8", "uint64"])
@pytest.mark.parametrize("length", [0, 3])
def test_npy_exact_extent_preserves_supported_numeric_headers(
    version: tuple[int, int], dtype: str, length: int
) -> None:
    value = np.arange(length, dtype=dtype)
    member = io.BytesIO()
    np.lib.format.write_array(member, value, version=version, allow_pickle=False)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cell.npy", member.getvalue())
    result = candidate_artifact._load_returns(
        payload.getvalue(), expected_keys=frozenset({"cell"})
    )["cell"]
    assert result.dtype == value.dtype
    assert result.tobytes() == value.tobytes()


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_oversized_npy_header_is_refused_before_reading_or_materializing(
    version: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    width = 2 if version == (1, 0) else 4
    member = (
        np.lib.format.magic(*version) + (10001).to_bytes(width, "little") + b" " * 10001
    )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cell.npy", member)
    original_read = zipfile.ZipExtFile.read

    def bounded_read(handle: zipfile.ZipExtFile, n: int = -1) -> bytes:
        assert 0 <= n <= 10000, "oversized header was requested"
        return original_read(handle, n)

    def forbidden_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("oversized NPY header reached materialization")

    monkeypatch.setattr(zipfile.ZipExtFile, "read", bounded_read)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden_decode)
    with pytest.raises(ValueError, match="invalid NPY header extent"):
        candidate_artifact._load_returns(
            payload.getvalue(), expected_keys=frozenset({"cell"})
        )


def test_all_member_extents_are_checked_before_materializing_any_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = io.BytesIO()
    np.save(good, np.asarray([0.1]), allow_pickle=False)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("a.npy", good.getvalue())
        archive.writestr("z.npy", good.getvalue() + b"extra")

    def forbidden_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("array decoded before the complete member preflight")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden_decode)
    with pytest.raises(ValueError, match="byte extent"):
        candidate_artifact._load_returns(
            payload.getvalue(), expected_keys=frozenset({"a", "z"})
        )
